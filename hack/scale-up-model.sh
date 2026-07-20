#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-physical-ai-models}"
KEDA_PROXY="keda-add-ons-http-interceptor-proxy.openshift-keda.svc.cluster.local:8080"
POLL_INTERVAL=15
TIMEOUT=3600

usage() {
  echo "Usage: $0 <model-name>"
  echo ""
  echo "Scale up a model from zero by sending a request through the KEDA"
  echo "HTTP interceptor proxy, then wait until the predictor pod is ready."
  echo ""
  echo "Examples:"
  echo "  $0 pi05"
  echo "  $0 dreamzero"
  echo "  $0 cosmos3-nano"
  echo ""
  echo "Environment variables:"
  echo "  NAMESPACE      Target namespace (default: physical-ai-models)"
  echo "  TIMEOUT        Max wait in seconds (default: 3600)"
  exit 1
}

[[ $# -eq 1 ]] || usage
MODEL="$1"
PREDICTOR="${MODEL}-predictor"
HOST="${PREDICTOR}.${NAMESPACE}.svc.cluster.local"

# Check if already running
READY=$(oc get pods -n "$NAMESPACE" \
  -l "serving.kserve.io/inferenceservice=${MODEL}" \
  -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || true)

if [[ "$READY" == "true" ]]; then
  echo "${MODEL} is already up and ready."
  exit 0
fi

echo "Triggering scale-up for ${MODEL}..."
oc delete pod "scale-trigger-${MODEL}" -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
oc run "scale-trigger-${MODEL}" \
  --rm -i --restart=Never \
  --image=curlimages/curl \
  -n "$NAMESPACE" \
  -- -s -o /dev/null -w "" \
  -H "Host: ${HOST}" \
  "http://${KEDA_PROXY}/" &>/dev/null &
TRIGGER_PID=$!

echo "Waiting for ${PREDICTOR} pod to become ready (timeout: ${TIMEOUT}s)..."
ELAPSED=0
while (( ELAPSED < TIMEOUT )); do
  STATUS=$(oc get pods -n "$NAMESPACE" \
    -l "serving.kserve.io/inferenceservice=${MODEL}" \
    -o jsonpath='{.items[0].status.phase}' 2>/dev/null || true)
  READY=$(oc get pods -n "$NAMESPACE" \
    -l "serving.kserve.io/inferenceservice=${MODEL}" \
    -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || true)
  RESTARTS=$(oc get pods -n "$NAMESPACE" \
    -l "serving.kserve.io/inferenceservice=${MODEL}" \
    -o jsonpath='{.items[0].status.containerStatuses[0].restartCount}' 2>/dev/null || true)

  if [[ "$READY" == "true" ]]; then
    echo ""
    echo "${MODEL} is ready! (${ELAPSED}s elapsed, ${RESTARTS:-0} restarts)"
    kill "$TRIGGER_PID" 2>/dev/null || true
    wait "$TRIGGER_PID" 2>/dev/null || true
    exit 0
  fi

  if [[ -n "$RESTARTS" ]] && (( RESTARTS > 3 )); then
    echo ""
    echo "ERROR: ${MODEL} has restarted ${RESTARTS} times — likely crash-looping."
    echo "Check logs: oc logs -n ${NAMESPACE} -l serving.kserve.io/inferenceservice=${MODEL}"
    kill "$TRIGGER_PID" 2>/dev/null || true
    wait "$TRIGGER_PID" 2>/dev/null || true
    exit 1
  fi

  printf "\r  %3ds  status=%-20s ready=%-5s restarts=%s" \
    "$ELAPSED" "${STATUS:-pending}" "${READY:-false}" "${RESTARTS:-0}"
  sleep "$POLL_INTERVAL"
  ELAPSED=$(( ELAPSED + POLL_INTERVAL ))
done

echo ""
echo "ERROR: Timed out after ${TIMEOUT}s waiting for ${MODEL} to become ready."
kill "$TRIGGER_PID" 2>/dev/null || true
wait "$TRIGGER_PID" 2>/dev/null || true
exit 1
