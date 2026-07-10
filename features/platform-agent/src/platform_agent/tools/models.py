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


@tool
def get_model_status(model_name: str) -> str:
    """Get detailed status of a specific model including its InferenceService conditions and pod health.

    Args:
        model_name: The name of the InferenceService to check.
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
            return f"InferenceService '{model_name}' not found."
        raise

    conditions = isvc.get("status", {}).get("conditions", [])
    cond_lines = [
        f"  {c['type']}: {c['status']} (reason={c.get('reason', 'N/A')})"
        for c in conditions
    ]

    pods = core_api.list_namespaced_pod(
        namespace=settings.models_namespace,
        label_selector=f"serving.kserve.io/inferenceservice={model_name}",
    )

    pod_lines = []
    for pod in pods.items:
        restarts = sum(
            cs.restart_count for cs in (pod.status.container_statuses or [])
        )
        pod_lines.append(
            f"  {pod.metadata.name}: phase={pod.status.phase}, restarts={restarts}"
        )

    output_kind = isvc["metadata"].get("annotations", {}).get(
        "physical-ai.io/output-kind", "chat"
    )

    output = f"InferenceService: {model_name}\n"
    output += f"Output kind: {output_kind}\n"
    output += "Conditions:\n" + "\n".join(cond_lines) + "\n" if cond_lines else ""
    output += f"Pods ({len(pods.items)}):\n" + "\n".join(pod_lines) if pod_lines else "Pods: none (likely scaled to zero)"
    return output


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
