#!/usr/bin/env bash
# k8s/monitoring/setup_cloud_monitoring_alerts.sh
#
# Infra-level alerts (pod restarts, OOM kills) don't need kube-state-metrics
# or any extra in-cluster component — GKE already exports these to Cloud
# Monitoring by default. This just wires up alert policies against metrics
# that already exist. App-level alerts (latency, error rate) live in
# Prometheus/Alertmanager instead — see prometheus-configmap.yaml.
#
# Requires: gcloud CLI authenticated, GCP_PROJECT_ID set below or exported.
#
# Usage:
#   GCP_PROJECT_ID=venera-chatbot ./setup_cloud_monitoring_alerts.sh
#
# This creates a notification channel stub you must finish configuring
# (email/SMS/Slack) in the Cloud Console before alerts will actually reach
# anyone — gcloud can create the policies but the notification channel
# needs an interactive step to verify the destination.

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"

echo "Creating notification channel (you'll need to verify it in the Console)..."
gcloud alpha monitoring channels create \
  --project="$PROJECT_ID" \
  --display-name="Venera Chatbot Alerts" \
  --type=email \
  --channel-labels=email_address=YOUR_EMAIL_HERE \
  --description="Placeholder — replace YOUR_EMAIL_HERE and re-run, or edit in Console."

CHANNEL_ID=$(gcloud alpha monitoring channels list \
  --project="$PROJECT_ID" \
  --filter='displayName="Venera Chatbot Alerts"' \
  --format='value(name)' | head -n1)

echo "Notification channel: $CHANNEL_ID"

# Pod restart count — venera-chatbot deployment specifically.
gcloud alpha monitoring policies create \
  --project="$PROJECT_ID" \
  --display-name="Venera Chatbot: Pod Restarts" \
  --condition-display-name="Restart count increasing" \
  --condition-filter='resource.type="k8s_container" AND resource.labels.container_name="venera-chatbot" AND metric.type="kubernetes.io/container/restart_count"' \
  --condition-threshold-value=0 \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-duration=0s \
  --notification-channels="$CHANNEL_ID"

# OOM kills — surfaces as a restart with OOMKilled reason; GKE exposes this
# via the same restart_count metric plus a memory-limit-utilization signal.
# Using memory utilization near 100% as the leading indicator, since it fires
# before the kill happens rather than after.
gcloud alpha monitoring policies create \
  --project="$PROJECT_ID" \
  --display-name="Venera Chatbot: Memory Near Limit (OOM risk)" \
  --condition-display-name="Memory utilization > 90% of limit" \
  --condition-filter='resource.type="k8s_container" AND resource.labels.container_name="venera-chatbot" AND metric.type="kubernetes.io/container/memory/limit_utilization"' \
  --condition-threshold-value=0.9 \
  --condition-threshold-comparison=COMPARISON_GT \
  --condition-threshold-duration=120s \
  --notification-channels="$CHANNEL_ID"

echo "Done. Go verify the email notification channel in the Console:"
echo "  https://console.cloud.google.com/monitoring/alerting/notifications?project=$PROJECT_ID"
