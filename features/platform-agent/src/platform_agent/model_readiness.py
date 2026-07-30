from kubernetes import client

from platform_agent.config import settings


def _get_k8s_client():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CustomObjectsApi(), client.CoreV1Api()


CRASH_REASONS = {"CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "InvalidImageName"}


def _fetch_model_state(model_name: str) -> dict | None:
    """K8s fetch backing get_model_readiness(). Returns None if the
    InferenceService doesn't exist.
    """
    custom_api, core_api = _get_k8s_client()

    try:
        isvc = custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=settings.models_namespace,
            plural="inferenceservices",
            name=model_name,
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return None
        raise

    conditions = isvc.get("status", {}).get("conditions", [])
    ready = next(
        (c["status"] for c in conditions if c["type"] == "Ready"),
        "Unknown",
    )

    pods = core_api.list_namespaced_pod(
        namespace=settings.models_namespace,
        label_selector=f"serving.kserve.io/inferenceservice={model_name}",
    )

    pod_infos = []
    for pod in pods.items:
        statuses = pod.status.container_statuses or []
        restarts = sum(cs.restart_count for cs in statuses)
        waiting_reason = next(
            (cs.state.waiting.reason for cs in statuses if cs.state and cs.state.waiting),
            None,
        )
        pod_ready = bool(statuses) and all(cs.ready for cs in statuses)
        pod_infos.append(
            {
                "name": pod.metadata.name,
                "phase": pod.status.phase,
                "restarts": restarts,
                "waiting_reason": waiting_reason,
                "container_ready": pod_ready,
            }
        )

    output_kind = isvc["metadata"].get("annotations", {}).get(
        "physical-ai.io/output-kind", "chat"
    )

    return {
        "conditions": conditions,
        # NOTE: the InferenceService's own "Ready" condition is misleading for
        # external-autoscaler/scale-to-zero models — KServe reports Ready=True
        # even at zero replicas, since scaled-to-zero is a valid steady state
        # for that autoscaler class. Actual readiness must come from pod
        # container status instead (see "pods" below).
        "isvc_ready_condition": ready == "True",
        "pods": pod_infos,
        "output_kind": output_kind,
    }


def get_model_readiness(model_name: str) -> dict:
    """Plain, non-LLM-dependent readiness check. Safe to call before the
    model (and therefore the agent's own LLM calls) is up — used by
    GET /api/model/status.

    Returns {"state": ..., "detail": ...} where state is one of:
    "ready", "starting", "not_started", "error".
    """
    state = _fetch_model_state(model_name)
    if state is None:
        return {"state": "error", "detail": f"InferenceService '{model_name}' not found."}

    for pod in state["pods"]:
        if pod["waiting_reason"] in CRASH_REASONS:
            return {
                "state": "error",
                "detail": f"Pod {pod['name']} is in {pod['waiting_reason']}.",
            }

    if not state["pods"]:
        return {"state": "not_started", "detail": "Scaled to zero."}

    if any(p["container_ready"] for p in state["pods"]):
        return {"state": "ready", "detail": "Model is up and ready."}

    phases = ", ".join(p["phase"] for p in state["pods"])
    return {"state": "starting", "detail": f"Model is starting up (pod phase: {phases})."}


def resume_scaling(model_name: str) -> None:
    """Clear a KEDA paused-replicas annotation left over from a previous
    scale_model(..., 0) shutdown, so HTTP traffic can trigger scale-from-zero
    again. Without this, a paused ScaledObject ignores incoming requests and
    stays pinned at 0 forever. Safe to call on models without an
    HTTPScaledObject (404s ignored).
    """
    custom_api, _ = _get_k8s_client()
    scaler_name = f"{model_name}-http-scaler"
    try:
        custom_api.patch_namespaced_custom_object(
            group="keda.sh",
            version="v1alpha1",
            namespace=settings.models_namespace,
            plural="scaledobjects",
            name=scaler_name,
            body={"metadata": {"annotations": {"autoscaling.keda.sh/paused-replicas": None}}},
        )
    except client.exceptions.ApiException:
        pass
