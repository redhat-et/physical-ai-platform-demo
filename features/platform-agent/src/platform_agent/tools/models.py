from langchain_core.tools import tool
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

    output = f"InferenceService: {model_name}\n"
    output += "Conditions:\n" + "\n".join(cond_lines) + "\n" if cond_lines else ""
    output += f"Pods ({len(pods.items)}):\n" + "\n".join(pod_lines) if pod_lines else "Pods: none (likely scaled to zero)"
    return output
