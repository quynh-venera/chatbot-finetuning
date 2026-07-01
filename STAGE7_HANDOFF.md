# Venera AI Chatbot — Handoff Addendum: Stage 7

Append this to `venera_handoff_stages1to6.md`, or keep alongside it. Covers
monitoring and stress testing, built on top of the Stage 6 GKE deployment.

**Status: code and manifests complete, not yet deployed** — the GKE cluster
was deleted after Stage 6 testing (see Stage 6 known issues). Everything
below is ready to `kubectl apply` once the cluster is recreated.

---

## What changed

### `serve/app.py`
Added Prometheus instrumentation:
- `chatbot_requests_total{endpoint,method,status}` — counter
- `chatbot_request_latency_seconds{endpoint}` — histogram, buckets tuned for
  CPU inference latency (up to 60s) rather than typical web-request buckets
- `chatbot_inference_errors_total` — counter, incremented only on actual
  `llm()` call failures, not on validation errors (empty question, etc.)
- `chatbot_tokens_generated` — histogram of `tokens_used` per `/chat` response
- `chatbot_requests_in_progress` — gauge, useful for spotting queueing under load
- `chatbot_model_loaded` / `chatbot_model_load_seconds` — gauges set in the
  `lifespan` handler

New `GET /metrics` endpoint, excluded from its own middleware instrumentation
(scraping shouldn't inflate the counters it's reading).

`serve/requirements-serve.txt`: added `prometheus-client>=0.20.0`.

### `k8s/deployment.yaml`
Added pod annotations (`prometheus.io/scrape`, `port`, `path`) so the new
Prometheus deployment discovers the pod automatically via
`kubernetes_sd_configs` — no hardcoded target list to maintain.

### `k8s/monitoring/` (new)
Self-hosted Prometheus + Alertmanager + Grafana, plain manifests (no Helm —
kept inspectable given this is a small prototype cluster):
- `namespace.yaml` — separate `monitoring` namespace
- `prometheus-rbac.yaml` — ServiceAccount + ClusterRole for pod discovery
- `prometheus-configmap.yaml` — scrape config + alert rules (latency, error
  rate, model-not-loaded, no-traffic)
- `prometheus-deployment.yaml` — Prometheus itself, `emptyDir` storage
  (**data is lost on pod restart** — fine for a prototype, swap for a PVC
  before this matters)
- `alertmanager.yaml` — routes alerts; ships with a placeholder webhook
  receiver, **not wired to anything real yet**
- `grafana-dashboard.json` + `grafana.yaml` — Grafana with the Prometheus
  datasource and a starter dashboard pre-provisioned (request rate, p50/p95/p99
  latency, tokens/response, in-progress requests, error rate, model-loaded stat)
- `setup_cloud_monitoring_alerts.sh` — separate from the above; wires up
  Cloud Monitoring alert policies for pod restarts and OOM risk

### `stress_test/` (new)
- `locustfile.py` — hits `/chat` with realistic Venera AI questions,
  low-weight `/health` canary task to check event-loop starvation under load
- `requirements-stress.txt`

---

## Key decision: split infra alerts from app alerts

Two alerting paths, deliberately not merged into one system:

| Signal | Where it lives | Why |
|---|---|---|
| Pod restarts, OOM risk | **Cloud Monitoring** (GKE exports these natively) | No extra component needed — `kube-state-metrics` would duplicate what GCP already tracks for free |
| Request latency, error rate, model-loaded state | **Prometheus + Alertmanager** | These are app-level signals only the FastAPI process knows about; Cloud Monitoring has no visibility into them without a custom exporter, which is what `/metrics` already is |

If this were a real production service, both would probably route to the
same alerting destination (PagerDuty/Slack) — for the prototype they're two
separate, unwired placeholders. See "Not yet done" below.

---

## Deploy steps (once cluster is recreated)

```bash
# 1. Recreate the cluster (Stage 6 known issue #3)
gcloud container clusters create venera-chatbot-cluster \
  --zone us-east1-b --num-nodes 1 --machine-type e2-standard-2
gcloud container clusters get-credentials venera-chatbot-cluster --zone us-east1-b

# 2. Deploy the chatbot (as before, now with scrape annotations)
kubectl create secret generic venera-secrets --from-literal=hf-token=YOUR_HF_TOKEN
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 3. Deploy the monitoring stack
kubectl apply -f k8s/monitoring/namespace.yaml
kubectl apply -f k8s/monitoring/prometheus-rbac.yaml
kubectl apply -f k8s/monitoring/prometheus-configmap.yaml
kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
kubectl apply -f k8s/monitoring/alertmanager.yaml
kubectl apply -f k8s/monitoring/grafana.yaml

# 4. Get Grafana's external IP (LoadBalancer, same pattern as the chatbot service)
kubectl get service grafana -n monitoring
# Login: admin / admin (GF_SECURITY_ADMIN_PASSWORD in grafana.yaml) — change this
# if the cluster stays up longer than a quick test.

# 5. Confirm Prometheus is scraping the chatbot pod
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# open http://localhost:9090/targets — should show the venera-chatbot pod as UP

# 6. Optional: wire up infra alerts
GCP_PROJECT_ID=venera-chatbot ./k8s/monitoring/setup_cloud_monitoring_alerts.sh
# edit YOUR_EMAIL_HERE in the script first, or edit the channel in Console after
```

## Running the stress test

```bash
pip install -r stress_test/requirements-stress.txt

# Interactive UI (recommended first pass — watch Grafana panels live alongside it)
locust -f stress_test/locustfile.py --host http://<chatbot-loadbalancer-ip>

# Headless, e.g. ramp to 20 users over 60s, run 5 minutes, save a CSV report
locust -f stress_test/locustfile.py --host http://<chatbot-loadbalancer-ip> \
  --headless -u 20 -r 1 -t 5m --csv=stress_test/results/run1
```

**Expected outcome, not yet measured**: single replica, CPU-only inference,
`N_THREADS=4` on an `e2-standard-2` node. Throughput should fall off sharply
somewhere around 2-4 concurrent in-flight requests, since llama.cpp
serializes on available CPU threads. The test is meant to *find* that
ceiling, not to prove the prototype handles real traffic — it doesn't yet on
one replica with no autoscaling. Results should go back into this doc once
the cluster is up and the test has actually run.

---

## Not yet done / open questions for Stage 7 wrap-up

1. **Alertmanager receiver is a placeholder** — `alertmanager.yaml` points
   at an invalid webhook URL. Needs a real Slack webhook or similar before
   any alert actually reaches anyone.
2. **Cloud Monitoring notification channel needs manual verification** — 
   `setup_cloud_monitoring_alerts.sh` creates an email channel but GCP
   requires interactive email verification; can't be scripted end-to-end.
3. **Prometheus storage is ephemeral** (`emptyDir`) — acceptable for a
   prototype, but metrics history is lost on every Prometheus pod restart.
   Add a PVC if longer retention matters.
4. **Stress test hasn't been run against a live pod yet** — cluster is down.
   Run it after redeploying and record actual p95/p99 numbers + the
   concurrency level where latency starts climbing, here in this doc.
5. **No horizontal pod autoscaling** — `k8s/deployment.yaml` is still
   `replicas: 1`. Worth deciding whether Stage 7's stress test results
   justify an HPA before moving to Stage 8, or whether that's out of
   scope for a prototype.
6. **Grafana admin password is hardcoded to `admin`** in `grafana.yaml` —
   fine for a short-lived prototype cluster, not fine if this cluster stays
   up unattended.
