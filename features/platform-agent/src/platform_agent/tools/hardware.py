import math

from langchain_core.tools import tool
from kubernetes import client

from platform_agent.config import settings

# VRAM per GPU product, in GB. Deliberately small and explicit rather than
# guessed — add an entry here when a new GPU type is added to the cluster.
GPU_VRAM_GB = {
    "NVIDIA-L40S": 48,
}

BYTES_PER_PARAM = {
    "F32": 4, "FP32": 4,
    "F16": 2, "FP16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "FP8": 1,
    "I4": 0.5, "INT4": 0.5,
}

# Rough overhead multiplier for activations/KV-cache on top of raw weight
# size. A heuristic, not a guarantee — always leave headroom beyond this.
FOOTPRINT_OVERHEAD_FACTOR = 1.2

# Fraction of a GPU's VRAM assumed usable once framework/runtime overhead is
# accounted for, when sizing tensor-parallel-size.
GPU_UTILIZATION_HEADROOM = 0.85


def _get_core_client():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api()


@tool
def list_cluster_gpus() -> str:
    """List GPU capacity on the cluster, grouped by GPU product type: total
    GPUs, how many are currently in use by running model pods, and VRAM per
    GPU where known. Use this before recommending hardware for a new model
    deployment.
    """
    core_api = _get_core_client()

    nodes = core_api.list_node()
    capacity = {}
    node_counts = {}
    for node in nodes.items:
        labels = node.metadata.labels or {}
        product = labels.get("nvidia.com/gpu.product")
        if not product:
            continue
        gpu_count = int((node.status.allocatable or {}).get("nvidia.com/gpu", "0"))
        if gpu_count == 0:
            continue
        capacity[product] = capacity.get(product, 0) + gpu_count
        node_counts[product] = node_counts.get(product, 0) + 1

    if not capacity:
        return "No GPU nodes found on the cluster (no nodes with an nvidia.com/gpu.product label)."

    pods = core_api.list_namespaced_pod(namespace=settings.models_namespace)
    in_use = {}
    for pod in pods.items:
        if pod.status.phase not in ("Running", "Pending"):
            continue
        product = (pod.spec.node_selector or {}).get("nvidia.com/gpu.product")
        if not product:
            continue
        gpu_count = 0
        for container in pod.spec.containers:
            requests = (container.resources.requests or {}) if container.resources else {}
            gpu_count += int(requests.get("nvidia.com/gpu", "0"))
        if gpu_count:
            in_use[product] = in_use.get(product, 0) + gpu_count

    lines = []
    for product, total in sorted(capacity.items()):
        vram = GPU_VRAM_GB.get(product)
        vram_str = f"{vram}GB VRAM" if vram else "VRAM unknown"
        used = in_use.get(product, 0)
        lines.append(
            f"- {product}: {vram_str}, {total} total on {node_counts[product]} "
            f"node(s), {used} in use, {total - used} free"
        )
    return "GPU capacity:\n" + "\n".join(lines)


@tool
def estimate_model_footprint(
    hf_repo_id: str,
    dtype: str = "auto",
    gpu_product: str = "NVIDIA-L40S",
) -> str:
    """Estimate the GPU memory footprint of a Hugging Face model and how many
    of a given GPU type it would need. Reads parameter count from the
    model's safetensors metadata — no weights are downloaded. Call this
    before generate_model_manifests to pick a tensor_parallel_size, and call
    list_cluster_gpus first to know what GPU types/capacity are actually
    available.

    Args:
        hf_repo_id: Hugging Face repo id, e.g. 'Qwen/Qwen3-8B'.
        dtype: Weight dtype to size for, e.g. 'BF16', 'FP8', 'INT4'. Defaults
            to 'auto', which uses whichever dtype the model's safetensors
            metadata reports the most parameters in.
        gpu_product: GPU type to size against — see list_cluster_gpus for
            what's actually available on this cluster. Defaults to
            'NVIDIA-L40S', the only GPU type currently on this cluster.
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(hf_repo_id)
    except Exception as e:
        return f"Could not fetch model info for '{hf_repo_id}' from Hugging Face: {e}"

    if not info.safetensors or not info.safetensors.parameters:
        return (
            f"'{hf_repo_id}' has no safetensors metadata to size from — it "
            f"may not be in safetensors format, or may be gated/private."
        )

    param_map = info.safetensors.parameters
    if dtype == "auto":
        dtype_used, total_params = max(param_map.items(), key=lambda kv: kv[1])
    else:
        dtype_used = dtype.upper()
        total_params = param_map.get(dtype_used) or sum(param_map.values())

    bytes_per_param = BYTES_PER_PARAM.get(dtype_used.upper())
    if bytes_per_param is None:
        return (
            f"Unrecognized dtype '{dtype_used}' for '{hf_repo_id}' — known "
            f"dtypes are {sorted(BYTES_PER_PARAM)}. Pass an explicit `dtype`."
        )

    estimated_vram_gb = total_params * bytes_per_param * FOOTPRINT_OVERHEAD_FACTOR / 1e9

    gpu_vram_gb = GPU_VRAM_GB.get(gpu_product)
    if gpu_vram_gb is None:
        return (
            f"~{total_params / 1e9:.1f}B params ({dtype_used}), estimated "
            f"{estimated_vram_gb:.1f}GB VRAM needed. VRAM for '{gpu_product}' "
            f"isn't in the known GPU table — add it to GPU_VRAM_GB to get a "
            f"tensor_parallel_size recommendation, or pass a known gpu_product."
        )

    usable_vram_gb = gpu_vram_gb * GPU_UTILIZATION_HEADROOM
    recommended_tp = max(1, math.ceil(estimated_vram_gb / usable_vram_gb))

    tier_note = ""
    if len(GPU_VRAM_GB) <= 1:
        tier_note = (
            f" Note: this cluster only has one known GPU type "
            f"({gpu_product}), so there's no cost/latency tier tradeoff to "
            f"weigh yet — tensor_parallel_size is the main lever."
        )

    return (
        f"'{hf_repo_id}': ~{total_params / 1e9:.1f}B params ({dtype_used}), "
        f"estimated {estimated_vram_gb:.1f}GB VRAM (includes a ~20% overhead "
        f"margin for activations/KV-cache — a rough estimate, not exact). "
        f"Recommended tensor_parallel_size={recommended_tp} on {gpu_product} "
        f"({gpu_vram_gb}GB VRAM each).{tier_note}"
    )
