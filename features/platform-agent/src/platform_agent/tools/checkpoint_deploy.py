"""Deploys a finished fine-tuning checkpoint as a live, scale-to-zero
InferenceService for side-by-side comparison against the base model -- e.g.
in the robotics playground (registering it there is a manual follow-up, not
done by these tools).

submit_finetune_run's checkpoint PVC lives in settings.datasets_namespace
(where the KFP/DSPA pipeline that produced it actually runs -- see
finetune_pipeline.py's DSPA_NAMESPACE), but serving happens in
settings.models_namespace (see platform/base/models/pi05/inferenceservice.yaml).
A Pod can never mount a PVC from a different namespace, so the checkpoint has
to be copied into a new PVC that actually lives in the models namespace
before anything can serve it. That copy is a small export/import Job pair
(mirroring datasets.py's pull_dataset/get_dataset_job_status Job-then-poll
idiom) rather than a blocking operation inside one tool call, since a full
checkpoint copy can take a couple of minutes: an export Job in the datasets
namespace mounts the checkpoint PVC read-only and serves it via
`python -m http.server`; an import Job in the models namespace mounts the new
destination PVC and crawls that HTTP listing with stdlib urllib (no wget --
python:3.11-slim doesn't ship it, and installing it via apt adds an
unnecessary external dependency).

Once the checkpoint's own PVC is populated, the InferenceService reuses the
existing openpi-runtime ServingRuntime as-is (platform/base/models/pi05/
servingruntime.yaml) -- it already just loads whatever's mounted at
/mnt/models, so no new runtime is needed, only a new storageUri.

Only 'pi05' is supported, same restriction as finetune_recipes.get_recipe --
this only makes sense once a second fine-tuning recipe exists.
"""

from kubernetes import client
from langchain_core.tools import tool

from platform_agent.config import settings
from platform_agent.tools.finetune import FINETUNE_EXP_LABEL, _checkpoint_pvc_name
from platform_agent.tools.finetune_pipeline import GPU_NODE_SELECTOR_KEY, GPU_NODE_SELECTOR_VALUE
from platform_agent.tools.finetune_recipes import CHECKPOINT_MOUNT_PATH, _checkpoint_dir
from platform_agent.tools.models import _live_pod_status

CHECKPOINT_DEPLOYMENT_LABEL = "physical-ai.io/checkpoint-deployment"
CHECKPOINT_EXPORT_PORT = 8080
EXPORT_JOB_TIMEOUT_SECONDS = 900  # safety net in case the import side never shows up

_SUPPORTED_MODELS = ("pi05",)


def _get_clients():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api(), client.BatchV1Api(), client.CustomObjectsApi()


def _require_pi05(model_name: str) -> str | None:
    if model_name not in _SUPPORTED_MODELS:
        return f"No checkpoint-deployment support for '{model_name}' -- only {_SUPPORTED_MODELS} so far."
    return None


def _isvc_name(model_name: str, exp_name: str) -> str:
    return f"{model_name}-ft-{exp_name}"


def _export_job_name(exp_name: str) -> str:
    return f"checkpoint-export-{exp_name}"


def _import_job_name(isvc_name: str) -> str:
    return f"checkpoint-import-{isvc_name}"


def _model_cache_pvc_name(isvc_name: str) -> str:
    return f"{isvc_name}-model-cache"


def _triton_cache_pvc_name(isvc_name: str) -> str:
    return f"{isvc_name}-triton-cache"


def _scaler_name(isvc_name: str) -> str:
    return f"{isvc_name}-http-scaler"


@tool
def deploy_checkpoint_model(exp_name: str, model_name: str = "pi05") -> str:
    """Start deploying a finished fine-tuning checkpoint as a live, callable
    model endpoint, separate from the base model, for side-by-side testing.

    This only starts the process -- it copies the checkpoint from the
    fine-tuning checkpoint PVC into a new PVC in the models namespace (a Pod
    can never mount a PVC from a different namespace, so this copy can't be
    skipped). Call get_checkpoint_deployment_status(exp_name) afterward,
    repeatedly, to advance through the copy and finish standing up the
    InferenceService -- unlike get_finetune_run_status, there's no pipeline
    driving this forward on its own.

    Only call this for a checkpoint whose fine-tuning run has actually
    succeeded (check get_finetune_run_status first) -- this only checks that
    the checkpoint PVC exists, not that training actually finished cleanly.

    Args:
        exp_name: The exp_name passed to submit_finetune_run.
        model_name: Which model this checkpoint is for. Only 'pi05' is
            supported so far.
    """
    err = _require_pi05(model_name)
    if err:
        return err

    isvc_name = _isvc_name(model_name, exp_name)
    if len(isvc_name) + len("-predictor") > 63:
        return (
            f"'{exp_name}' is too long -- the resulting resource name "
            f"'{isvc_name}-predictor' would exceed Kubernetes' 63-character name limit."
        )

    core_api, batch_api, _ = _get_clients()

    checkpoint_pvc_name = _checkpoint_pvc_name(exp_name)
    try:
        core_api.read_namespaced_persistent_volume_claim(
            name=checkpoint_pvc_name, namespace=settings.datasets_namespace
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return (
                f"No fine-tuning checkpoint found for '{exp_name}' -- has submit_finetune_run "
                f"been run for it? Check get_finetune_run_status('{exp_name}')."
            )
        return f"Could not read checkpoint PVC '{checkpoint_pvc_name}': {e.reason}"

    model_cache_pvc = _model_cache_pvc_name(isvc_name)
    labels = {FINETUNE_EXP_LABEL: exp_name, CHECKPOINT_DEPLOYMENT_LABEL: "true"}
    try:
        core_api.create_namespaced_persistent_volume_claim(
            namespace=settings.models_namespace,
            body={
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": model_cache_pvc, "labels": labels},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": "30Gi"}},
                    "storageClassName": "gp3-csi",
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to create destination PVC '{model_cache_pvc}': {e.reason}"

    export_job_name = _export_job_name(exp_name)
    export_script = f"""\
set -e
timeout {EXPORT_JOB_TIMEOUT_SECONDS} python3 -m http.server {CHECKPOINT_EXPORT_PORT} --directory {_checkpoint_dir(exp_name)}
"""
    try:
        batch_api.create_namespaced_job(
            namespace=settings.datasets_namespace,
            body={
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": export_job_name, "labels": labels},
                "spec": {
                    "backoffLimit": 0,
                    "activeDeadlineSeconds": EXPORT_JOB_TIMEOUT_SECONDS + 60,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "export",
                                    "image": "python:3.11-slim",
                                    "command": ["/bin/bash", "-c", export_script],
                                    "volumeMounts": [
                                        {
                                            "name": "checkpoint",
                                            "mountPath": CHECKPOINT_MOUNT_PATH,
                                            "readOnly": True,
                                        }
                                    ],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "checkpoint",
                                    "persistentVolumeClaim": {"claimName": checkpoint_pvc_name},
                                }
                            ],
                        },
                    },
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to start checkpoint export Job '{export_job_name}': {e.reason}"

    return (
        f"Started copying checkpoint '{exp_name}' into the models namespace as '{isvc_name}'. "
        f"Call get_checkpoint_deployment_status('{exp_name}') to advance and check progress -- "
        f"it needs to be called repeatedly until it reports the model deployed."
    )


@tool
def get_checkpoint_deployment_status(exp_name: str, model_name: str = "pi05") -> str:
    """Check progress of a checkpoint deployment started by
    deploy_checkpoint_model, and advance it to the next stage if the current
    one has finished.

    Call this repeatedly until it reports the model deployed -- unlike
    get_finetune_run_status, nothing else drives this forward automatically.
    Each call both checks status and, if ready, starts the next stage
    (creating the import Job once the export Job's pod is reachable, then
    the InferenceService once the import Job succeeds).

    Args:
        exp_name: The exp_name passed to deploy_checkpoint_model.
        model_name: Which model this checkpoint is for. Only 'pi05' is
            supported so far.
    """
    err = _require_pi05(model_name)
    if err:
        return err

    isvc_name = _isvc_name(model_name, exp_name)
    core_api, batch_api, custom_api = _get_clients()

    try:
        custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=settings.models_namespace,
            plural="inferenceservices",
            name=isvc_name,
        )
        already_deployed = True
    except client.exceptions.ApiException as e:
        if e.status != 404:
            return f"Could not read InferenceService '{isvc_name}': {e.reason}"
        already_deployed = False

    if already_deployed:
        pods = core_api.list_namespaced_pod(
            namespace=settings.models_namespace,
            label_selector=f"serving.kserve.io/inferenceservice={isvc_name}",
        )
        status = _live_pod_status(pods.items)
        return (
            f"'{isvc_name}' is deployed (scale-to-zero) at "
            f"{isvc_name}-predictor.{settings.models_namespace}.svc.cluster.local -- {status}. "
            f"Call scale_model('{isvc_name}', 1) to warm it up for testing."
        )

    import_job_name = _import_job_name(isvc_name)
    try:
        import_job = batch_api.read_namespaced_job(name=import_job_name, namespace=settings.models_namespace)
    except client.exceptions.ApiException as e:
        if e.status != 404:
            return f"Could not read import Job '{import_job_name}': {e.reason}"
        import_job = None

    if import_job is not None:
        if import_job.status.succeeded:
            try:
                batch_api.delete_namespaced_job(
                    name=_export_job_name(exp_name),
                    namespace=settings.datasets_namespace,
                    propagation_policy="Background",
                )
            except client.exceptions.ApiException:
                pass
            return _create_checkpoint_inference_service(custom_api, core_api, exp_name, isvc_name)
        if import_job.status.failed:
            return (
                f"Checkpoint copy for '{exp_name}' failed (import Job '{import_job_name}'). "
                f"Check get_pod_logs for its pod, then call deploy_checkpoint_model again after "
                f"fixing the issue."
            )
        return f"Copying checkpoint '{exp_name}' into the models namespace ('{import_job_name}' still running)."

    export_job_name = _export_job_name(exp_name)
    export_pods = core_api.list_namespaced_pod(
        namespace=settings.datasets_namespace, label_selector=f"job-name={export_job_name}"
    )
    export_pod = next(
        (p for p in export_pods.items if p.status.phase == "Running" and p.status.pod_ip), None
    )
    if export_pod is None:
        return f"Export Job '{export_job_name}' hasn't started serving yet -- try again shortly."

    return _start_import_job(batch_api, exp_name, isvc_name, export_pod.status.pod_ip)


def _start_import_job(batch_api, exp_name: str, isvc_name: str, export_pod_ip: str) -> str:
    import_job_name = _import_job_name(isvc_name)
    model_cache_pvc = _model_cache_pvc_name(isvc_name)
    labels = {FINETUNE_EXP_LABEL: exp_name, CHECKPOINT_DEPLOYMENT_LABEL: "true"}

    base_url = f"http://{export_pod_ip}:{CHECKPOINT_EXPORT_PORT}/"
    import_script = f"""\
set -e
python3 << 'PYEOF'
import os
import re
import urllib.parse
import urllib.request

BASE_URL = "{base_url}"
DEST_DIR = "/mnt/models"


def crawl(url, dest):
    with urllib.request.urlopen(url, timeout=30) as resp:
        html = resp.read().decode()
    for href in re.findall(r'href="([^"]+)"', html):
        if href in ("../", "./"):
            continue
        name = urllib.parse.unquote(href)
        child_url = url + href
        child_dest = os.path.join(dest, name.rstrip("/"))
        if href.endswith("/"):
            os.makedirs(child_dest, exist_ok=True)
            crawl(child_url, child_dest)
        else:
            os.makedirs(os.path.dirname(child_dest) or ".", exist_ok=True)
            urllib.request.urlretrieve(child_url, child_dest)


crawl(BASE_URL, DEST_DIR)
print("Checkpoint copy complete.")
PYEOF
"""
    try:
        batch_api.create_namespaced_job(
            namespace=settings.models_namespace,
            body={
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": import_job_name, "labels": labels},
                "spec": {
                    "backoffLimit": 2,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "import",
                                    "image": "python:3.11-slim",
                                    "command": ["/bin/bash", "-c", import_script],
                                    "volumeMounts": [{"name": "model-cache", "mountPath": "/mnt/models"}],
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "model-cache",
                                    "persistentVolumeClaim": {"claimName": model_cache_pvc},
                                }
                            ],
                        },
                    },
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to start checkpoint import Job '{import_job_name}': {e.reason}"

    return f"Export reachable -- copying checkpoint '{exp_name}' into the models namespace now ('{import_job_name}')."


def _create_checkpoint_inference_service(custom_api, core_api, exp_name: str, isvc_name: str) -> str:
    triton_cache_pvc = _triton_cache_pvc_name(isvc_name)
    model_cache_pvc = _model_cache_pvc_name(isvc_name)
    labels = {FINETUNE_EXP_LABEL: exp_name, CHECKPOINT_DEPLOYMENT_LABEL: "true"}

    try:
        core_api.create_namespaced_persistent_volume_claim(
            namespace=settings.models_namespace,
            body={
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": triton_cache_pvc, "labels": labels},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": "1Gi"}},
                    "storageClassName": "gp3-csi",
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to create triton-cache PVC '{triton_cache_pvc}': {e.reason}"

    isvc_body = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": settings.models_namespace,
            "labels": {
                "opendatahub.io/dashboard": "true",
                "opendatahub.io/genai-asset": "true",
                **labels,
            },
            "annotations": {
                "serving.kserve.io/deploymentMode": "RawDeployment",
                "serving.kserve.io/autoscalerClass": "external",
                "sidecar.istio.io/inject": "false",
                "physical-ai.io/output-kind": "unsupported",
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": 0,
                "deploymentStrategy": {"type": "Recreate"},
                "nodeSelector": {GPU_NODE_SELECTOR_KEY: GPU_NODE_SELECTOR_VALUE},
                "volumes": [
                    {"name": "triton-cache", "persistentVolumeClaim": {"claimName": triton_cache_pvc}}
                ],
                "model": {
                    "modelFormat": {"name": "pytorch"},
                    "runtime": "openpi-runtime",
                    "storageUri": f"pvc://{model_cache_pvc}",
                    "resources": {
                        "requests": {"cpu": "2", "memory": "24Gi", "nvidia.com/gpu": "1"},
                        "limits": {"cpu": "4", "memory": "48Gi", "nvidia.com/gpu": "1"},
                    },
                },
            },
        },
    }
    try:
        custom_api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=settings.models_namespace,
            plural="inferenceservices",
            body=isvc_body,
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to create InferenceService '{isvc_name}': {e.reason}"

    scaler_name = _scaler_name(isvc_name)
    scaler_body = {
        "apiVersion": "http.keda.sh/v1alpha1",
        "kind": "HTTPScaledObject",
        "metadata": {"name": scaler_name, "namespace": settings.models_namespace, "labels": labels},
        "spec": {
            "hosts": [f"{isvc_name}-predictor.{settings.models_namespace}.svc.cluster.local"],
            "targetPendingRequests": 1,
            "scaleTargetRef": {
                "name": f"{isvc_name}-predictor",
                "kind": "Deployment",
                "apiVersion": "apps/v1",
                "service": f"{isvc_name}-predictor",
                "port": 80,
            },
            "replicas": {"min": 0, "max": 1},
            "scaledownPeriod": 3600,
        },
    }
    try:
        custom_api.create_namespaced_custom_object(
            group="http.keda.sh",
            version="v1alpha1",
            namespace=settings.models_namespace,
            plural="httpscaledobjects",
            body=scaler_body,
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"InferenceService '{isvc_name}' created, but failed to create its HTTPScaledObject: {e.reason}"

    return (
        f"Deployed '{isvc_name}' -- scale-to-zero, currently at 0 replicas. "
        f"Predictor: {isvc_name}-predictor.{settings.models_namespace}.svc.cluster.local. "
        f"Call scale_model('{isvc_name}', 1) to warm it up for testing, and "
        f"takedown_checkpoint_model('{exp_name}') when you're done comparing."
    )


@tool
def takedown_checkpoint_model(exp_name: str, model_name: str = "pi05") -> str:
    """Tear down a checkpoint deployment created by deploy_checkpoint_model:
    its InferenceService, HTTPScaledObject, and the PVCs/Jobs this feature
    created for it. Frees the GPU it was holding (if scaled up) and the
    storage the checkpoint was copied into.

    Does NOT touch the original fine-tuning checkpoint PVC
    ('finetune-<exp_name>-checkpoint-pvc') -- that's still owned by the
    fine-tuning tools (see list_finetune_runs), so the same checkpoint can be
    redeployed later with deploy_checkpoint_model.

    Args:
        exp_name: The exp_name passed to deploy_checkpoint_model.
        model_name: Which model this checkpoint is for. Only 'pi05' is
            supported so far.
    """
    err = _require_pi05(model_name)
    if err:
        return err

    isvc_name = _isvc_name(model_name, exp_name)
    core_api, batch_api, custom_api = _get_clients()
    removed = []

    for group, version, plural, name in (
        ("http.keda.sh", "v1alpha1", "httpscaledobjects", _scaler_name(isvc_name)),
        ("serving.kserve.io", "v1beta1", "inferenceservices", isvc_name),
    ):
        try:
            custom_api.delete_namespaced_custom_object(
                group=group, version=version, namespace=settings.models_namespace, plural=plural, name=name
            )
            removed.append(name)
        except client.exceptions.ApiException as e:
            if e.status != 404:
                return f"Failed to delete {plural} '{name}': {e.reason}"

    for pvc_name in (_model_cache_pvc_name(isvc_name), _triton_cache_pvc_name(isvc_name)):
        try:
            core_api.delete_namespaced_persistent_volume_claim(name=pvc_name, namespace=settings.models_namespace)
            removed.append(pvc_name)
        except client.exceptions.ApiException as e:
            if e.status != 404:
                return f"Failed to delete PVC '{pvc_name}': {e.reason}"

    for namespace, job_name in (
        (settings.datasets_namespace, _export_job_name(exp_name)),
        (settings.models_namespace, _import_job_name(isvc_name)),
    ):
        try:
            batch_api.delete_namespaced_job(name=job_name, namespace=namespace, propagation_policy="Background")
            removed.append(job_name)
        except client.exceptions.ApiException as e:
            if e.status != 404:
                return f"Failed to delete Job '{job_name}': {e.reason}"

    if not removed:
        return f"No checkpoint deployment found for '{exp_name}' -- nothing to take down."
    return f"Took down checkpoint deployment '{isvc_name}': removed {', '.join(removed)}."


@tool
def list_checkpoint_deployments() -> str:
    """List fine-tuned checkpoints currently deployed as live comparison
    endpoints (via deploy_checkpoint_model), with real-time status. Separate
    from list_models (permanent catalog models) and list_finetune_runs
    (checkpoints that exist but may not be deployed anywhere)."""
    core_api, _, custom_api = _get_clients()
    items = custom_api.list_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=settings.models_namespace,
        plural="inferenceservices",
        label_selector=CHECKPOINT_DEPLOYMENT_LABEL,
    )

    isvcs = items.get("items", [])
    if not isvcs:
        return "No checkpoint deployments found."

    all_pods = core_api.list_namespaced_pod(namespace=settings.models_namespace)
    pods_by_isvc: dict[str, list] = {}
    for pod in all_pods.items:
        name = (pod.metadata.labels or {}).get("serving.kserve.io/inferenceservice")
        if name:
            pods_by_isvc.setdefault(name, []).append(pod)

    lines = []
    for isvc in isvcs:
        name = isvc["metadata"]["name"]
        exp_name = isvc["metadata"].get("labels", {}).get(FINETUNE_EXP_LABEL, "unknown")
        status = _live_pod_status(pods_by_isvc.get(name, []))
        lines.append(f"- {name} (exp_name={exp_name}): {status}")

    return "Checkpoint deployments:\n" + "\n".join(lines)
