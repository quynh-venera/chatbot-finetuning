"""
stress_test/locustfile.py

Load test for the Venera AI chatbot's /chat endpoint.

Run against the live GKE deployment:
    locust -f stress_test/locustfile.py --host http://34.139.132.166

Or headless, e.g. ramp to 20 concurrent users over 60s and run for 5 minutes:
    locust -f stress_test/locustfile.py --host http://34.139.132.166 \
        --headless -u 20 -r 1 -t 5m --csv=stress_test/results/run1

Notes specific to this deployment:
- Single replica, e2-standard-2 node (2 vCPU / 8GB), N_THREADS=4 CPU inference
  (see k8s/deployment.yaml). Expect throughput to fall off sharply once
  concurrent requests exceed ~2-4, since llama.cpp on CPU serializes on
  available threads. This test is meant to find that ceiling, not to prove
  the prototype can handle production traffic — it can't yet, on 1 replica.
- Watch the Grafana "p50/p95/p99 Latency" and "In-Progress Requests" panels
  live while this runs — they'll show the ceiling more clearly than the
  locust UI's aggregate stats.
- /chat has no timeout configured server-side, so a saturated pod will just
  queue requests rather than fail fast. If you see p99 latency climbing
  without 5xx errors, that's the queue growing, not stability.
"""

import random

from locust import HttpUser, between, task

# Mirrors the kind of questions the model was actually fine-tuned on
# (see data/processed/test.jsonl for the real held-out set).
QUESTIONS = [
    "What is Venera AI's health platform?",
    "How does Venera AI synchronize biological information?",
    "What data sources does the Venera AI platform support?",
    "Does Venera AI provide medical advice?",
    "How do I get started with Venera AI?",
    "What is the main function of the Venera AI platform?",
    "How does Venera AI handle user privacy?",
    "Can I integrate wearable devices with Venera AI?",
    "What is the difference between Venera AI's free and paid plans?",
    "How does Venera AI support healthcare providers?",
]


class ChatbotUser(HttpUser):
    # Gap between a simulated user's requests — real chat usage isn't a tight
    # loop, so don't hammer /chat back-to-back per user.
    wait_time = between(2, 8)

    @task(9)
    def ask_question(self):
        question = random.choice(QUESTIONS)
        with self.client.post(
            "/chat",
            json={"question": question},
            catch_response=True,
            timeout=60,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}: {resp.text[:200]}")
            elif not resp.json().get("answer"):
                resp.failure("200 but empty answer field")

    @task(1)
    def check_health(self):
        # Low-weight background check — confirms /health stays responsive
        # even while /chat is under load (it shouldn't share the inference
        # lock, so this is a useful canary for event-loop starvation).
        self.client.get("/health")
