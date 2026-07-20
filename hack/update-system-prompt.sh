#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-physical-ai}"
DEPLOYMENT="${DEPLOYMENT:-platform-agent}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-120s}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGMAP_FILE="${CONFIGMAP_FILE:-${SCRIPT_DIR}/../platform/base/agent/configmap.yaml}"

usage() {
  echo "Usage: $0"
  echo ""
  echo "Push the local platform-agent-config ConfigMap (system prompt and"
  echo "other agent settings) straight to the cluster, then restart the"
  echo "platform-agent deployment so the pod picks up the new value --"
  echo "no image rebuild needed."
  echo ""
  echo "NOTE: this applies your local file directly to the cluster, ahead"
  echo "of git. If the ArgoCD app for this overlay has self-heal enabled,"
  echo "it may revert an uncommitted change on its next reconcile -- commit"
  echo "and push once you're happy with the wording."
  echo ""
  echo "Environment variables:"
  echo "  NAMESPACE        Agent namespace (default: physical-ai)"
  echo "  DEPLOYMENT       Deployment name (default: platform-agent)"
  echo "  CONFIGMAP_FILE   Path to the ConfigMap manifest"
  echo "                   (default: platform/base/agent/configmap.yaml)"
  echo "  ROLLOUT_TIMEOUT  Max wait for rollout (default: 120s)"
  exit 1
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || usage
[[ $# -eq 0 ]] || usage

if [[ ! -f "$CONFIGMAP_FILE" ]]; then
  echo "ERROR: ConfigMap file not found: ${CONFIGMAP_FILE}" >&2
  exit 1
fi

echo "Applying ${CONFIGMAP_FILE} to the cluster..."
oc apply -f "$CONFIGMAP_FILE"

echo "Restarting deployment/${DEPLOYMENT} in ${NAMESPACE} to pick up the new prompt..."
oc rollout restart "deployment/${DEPLOYMENT}" -n "$NAMESPACE"

echo "Waiting for rollout to finish (timeout: ${ROLLOUT_TIMEOUT})..."
oc rollout status "deployment/${DEPLOYMENT}" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"

echo ""
echo "Done. ${DEPLOYMENT} is running with the updated ConfigMap."
