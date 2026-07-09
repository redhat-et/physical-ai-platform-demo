import threading

import httpx
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


def _trigger_scale_up(model_name: str):
    """Fire a request to the MaaS proxy to trigger KEDA scale-up."""
    url = (
        f"{settings.maas_proxy_url}/physical-ai-models/"
        f"{model_name}/v1/chat/completions"
    )
    try:
        with httpx.Client(verify=False, timeout=300.0) as http_client:
            http_client.post(
                url,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                headers={"Authorization": "Bearer unused"},
            )
    except Exception:
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
        custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=settings.models_namespace,
            plural="inferenceservices",
            name=model_name,
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"InferenceService '{model_name}' not found."
        return f"Failed to look up '{model_name}': {e.reason}"

    if min_replicas == 0:
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
            f"KEDA will keep it at zero until the next request."
        )

    threading.Thread(
        target=_trigger_scale_up, args=(model_name,), daemon=True
    ).start()
    return (
        f"Triggered scale-up for '{model_name}'. A request has been sent through "
        f"MaaS to wake the model — it may take a few minutes to become ready."
    )
