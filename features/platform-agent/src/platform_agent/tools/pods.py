from langchain_core.tools import tool
from kubernetes import client

from platform_agent.config import settings


def _get_core_client():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api()


@tool
def get_pod_logs(model_name: str, tail_lines: int = 50) -> str:
    """Get recent logs from a model's pod. Returns logs from the kserve-container.

    Args:
        model_name: The name of the InferenceService whose pod logs to retrieve.
        tail_lines: Number of log lines to return (default 50).
    """
    core_api = _get_core_client()

    pods = core_api.list_namespaced_pod(
        namespace=settings.models_namespace,
        label_selector=f"serving.kserve.io/inferenceservice={model_name}",
    )

    if not pods.items:
        return f"No pods found for model '{model_name}'. It may be scaled to zero."

    pod = pods.items[0]
    try:
        logs = core_api.read_namespaced_pod_log(
            name=pod.metadata.name,
            namespace=settings.models_namespace,
            container="kserve-container",
            tail_lines=tail_lines,
        )
    except client.exceptions.ApiException as e:
        return f"Could not read logs from {pod.metadata.name}: {e.reason}"

    return f"Logs from {pod.metadata.name} (last {tail_lines} lines):\n{logs}"
