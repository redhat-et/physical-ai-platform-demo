from langchain_core.tools import tool
from kubernetes import client
from kubernetes.client import AppsV1Api

from platform_agent.config import settings


def _get_k8s_client():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CustomObjectsApi(), client.CoreV1Api()


@tool
def list_models() -> str:
    """List all model InferenceServices deployed on the platform with their status."""
    custom_api, _ = _get_k8s_client()
    items = custom_api.list_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=settings.models_namespace,
        plural="inferenceservices",
    )

    results = []
    for isvc in items.get("items", []):
        name = isvc["metadata"]["name"]
        conditions = isvc.get("status", {}).get("conditions", [])
        ready = next(
            (c["status"] for c in conditions if c["type"] == "Ready"),
            "Unknown",
        )
        replicas = isvc.get("spec", {}).get("predictor", {}).get("minReplicas", "?")
        url = isvc.get("status", {}).get("url", "N/A")
        results.append(
            f"- {name}: ready={ready}, minReplicas={replicas}, url={url}"
        )

    if not results:
        return "No InferenceServices found in the models namespace."
    return "Models deployed:\n" + "\n".join(results)


CRASH_REASONS = {"CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "InvalidImageName"}


def _fetch_model_state(model_name: str) -> dict | None:
    """Shared K8s fetch used by both the get_model_status LangChain tool and
    get_model_readiness(). Returns None if the InferenceService doesn't exist.
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


@tool
def get_model_status(model_name: str) -> str:
    """Get detailed status of a specific model including its InferenceService conditions and pod health.

    Args:
        model_name: The name of the InferenceService to check.
    """
    state = _fetch_model_state(model_name)
    if state is None:
        return f"InferenceService '{model_name}' not found."

    cond_lines = [
        f"  {c['type']}: {c['status']} (reason={c.get('reason', 'N/A')})"
        for c in state["conditions"]
    ]
    pod_lines = [
        f"  {p['name']}: phase={p['phase']}, restarts={p['restarts']}"
        for p in state["pods"]
    ]

    output = f"InferenceService: {model_name}\n"
    output += f"Output kind: {state['output_kind']}\n"
    output += "Conditions:\n" + "\n".join(cond_lines) + "\n" if cond_lines else ""
    output += f"Pods ({len(state['pods'])}):\n" + "\n".join(pod_lines) if pod_lines else "Pods: none (likely scaled to zero)"
    return output


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


@tool
def scale_model(model_name: str, min_replicas: int) -> str:
    """Scale a model by setting its minReplicas. Use 1 to bring a model up
    and keep it running. Use 0 to shut it down immediately.

    Args:
        model_name: The name of the InferenceService to scale.
        min_replicas: Desired minimum replicas (0 = shut down, 1+ = keep running).
    """
    custom_api, core_api = _get_k8s_client()

    try:
        custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=settings.models_namespace,
            plural="inferenceservices",
            name=model_name,
            body={"spec": {"predictor": {"minReplicas": min_replicas}}},
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"InferenceService '{model_name}' not found."
        return f"Failed to scale '{model_name}': {e.reason}"

    scaler_name = f"{model_name}-http-scaler"
    try:
        custom_api.patch_namespaced_custom_object(
            group="http.keda.sh",
            version="v1alpha1",
            namespace=settings.models_namespace,
            plural="httpscaledobjects",
            name=scaler_name,
            body={"spec": {"replicas": {"min": min_replicas}}},
        )
    except client.exceptions.ApiException:
        pass

    # The HTTPScaledObject's own scaledownPeriod (idle cooldown, often ~1hr)
    # otherwise keeps its generated ScaledObject/HPA pinned at minReplicas=1
    # while "active", fighting any direct scale-down below. KEDA's pause
    # annotation is the documented way to force an exact replica count
    # regardless of triggers/cooldown; not all models have this scaler
    # (always-on models like mocklm/qwen25-cpu don't), so 404s are expected.
    try:
        if min_replicas == 0:
            custom_api.patch_namespaced_custom_object(
                group="keda.sh",
                version="v1alpha1",
                namespace=settings.models_namespace,
                plural="scaledobjects",
                name=scaler_name,
                body={"metadata": {"annotations": {"autoscaling.keda.sh/paused-replicas": "0"}}},
            )
        else:
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

    if min_replicas == 0:
        apps_api = AppsV1Api()
        deploy_name = f"{model_name}-predictor"
        try:
            apps_api.patch_namespaced_deployment_scale(
                name=deploy_name,
                namespace=settings.models_namespace,
                body={"spec": {"replicas": 0}},
            )
        except client.exceptions.ApiException:
            pass
        pods = core_api.list_namespaced_pod(
            namespace=settings.models_namespace,
            label_selector=f"serving.kserve.io/inferenceservice={model_name}",
        )
        deleted = 0
        for pod in pods.items:
            core_api.delete_namespaced_pod(
                name=pod.metadata.name,
                namespace=settings.models_namespace,
            )
            deleted += 1
        return (
            f"Shut down '{model_name}' — deleted {deleted} pod(s). "
            f"Model is now scaled to zero."
        )

    return f"Scaled '{model_name}' to minReplicas={min_replicas}. Model will start up shortly."
