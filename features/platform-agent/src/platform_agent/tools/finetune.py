from langchain_core.tools import tool
from kubernetes import client

from platform_agent.config import settings
from platform_agent.tools.datasets import DATASET_REPO_LABEL
from platform_agent.tools.finetune_recipes import CHECKPOINT_MOUNT_PATH, DATASET_MOUNT_ROOT, get_recipe, get_requirements

FINETUNE_EXP_LABEL = "physical-ai.io/finetune-exp"
FINETUNE_MODEL_LABEL = "physical-ai.io/finetune-model"
FINETUNE_DATASET_REPO_LABEL = "physical-ai.io/finetune-dataset-repo"
FINETUNE_DATASET_PVC_LABEL = "physical-ai.io/finetune-dataset-pvc"
FINETUNE_STAGE_LABEL = "physical-ai.io/finetune-stage"
FINETUNE_STAGE_INDEX_LABEL = "physical-ai.io/finetune-stage-index"

GPU_PRODUCT = "NVIDIA-L40S"


def _get_clients():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api(), client.BatchV1Api()


def _stage_job_name(exp_name: str, stage_name: str) -> str:
    return f"finetune-{exp_name}-{stage_name}"


def _checkpoint_pvc_name(exp_name: str) -> str:
    return f"finetune-{exp_name}-checkpoint-pvc"


def _create_stage_job(
    batch_api, exp_name: str, model_name: str, dataset_repo_id: str, dataset_pvc_name: str, stage: dict, stage_index: int
):
    job_name = _stage_job_name(exp_name, stage["name"])
    labels = {
        FINETUNE_EXP_LABEL: exp_name,
        FINETUNE_MODEL_LABEL: model_name,
        FINETUNE_DATASET_REPO_LABEL: dataset_repo_id.replace("/", "--"),
        FINETUNE_DATASET_PVC_LABEL: dataset_pvc_name,
        FINETUNE_STAGE_LABEL: stage["name"],
        FINETUNE_STAGE_INDEX_LABEL: str(stage_index),
    }

    dataset_mount_path = f"{DATASET_MOUNT_ROOT}/{dataset_repo_id}"
    volume_mounts = [
        {"name": "dataset", "mountPath": dataset_mount_path, "readOnly": True},
        {"name": "checkpoint", "mountPath": CHECKPOINT_MOUNT_PATH},
    ]
    volumes = [
        {"name": "dataset", "persistentVolumeClaim": {"claimName": dataset_pvc_name}},
        {"name": "checkpoint", "persistentVolumeClaim": {"claimName": _checkpoint_pvc_name(exp_name)}},
    ]

    resources = {"requests": {"cpu": "4", "memory": "16Gi"}, "limits": {"cpu": "8", "memory": "32Gi"}}
    node_selector = {}
    if stage["gpu"] > 0:
        resources["requests"]["nvidia.com/gpu"] = str(stage["gpu"])
        resources["limits"]["nvidia.com/gpu"] = str(stage["gpu"])
        node_selector = {"nvidia.com/gpu.product": GPU_PRODUCT}

    pod_spec = {
        "restartPolicy": "Never",
        "containers": [
            {
                "name": "stage",
                "image": stage["image"],
                "command": stage["command"],
                "env": [{"name": "HF_HOME", "value": "/tmp/hf_home"}],
                "volumeMounts": volume_mounts,
                "resources": resources,
            }
        ],
        "volumes": volumes,
    }
    if node_selector:
        pod_spec["nodeSelector"] = node_selector

    batch_api.create_namespaced_job(
        namespace=settings.datasets_namespace,
        body={
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_name, "labels": labels},
            "spec": {"backoffLimit": 0, "template": {"metadata": {"labels": labels}, "spec": pod_spec}},
        },
    )
    return job_name


@tool
def get_finetune_requirements(model_name: str = "pi05") -> str:
    """Get a model's fine-tuning dataset requirements -- robot embodiment,
    camera/state layout, dataset format, and a suggested search
    query/filter -- BEFORE searching for a dataset. Call this first, then
    pass its search_query_hint and robot_type into
    search_compatible_lerobot_datasets, instead of searching for the model
    name itself: a model-name keyword (e.g. 'pi05') returns datasets for
    ANY embodiment anyone used with that model (simulated benchmarks,
    humanoids, custom rigs with different cameras), not specifically the
    embodiment THIS recipe's data config expects. This is derived from the
    same recipe definition submit_finetune_run actually trains against, so
    it can't drift out of sync with what training really needs.

    Args:
        model_name: Which fine-tuning recipe to look up. Only 'pi05' exists so far.
    """
    try:
        req = get_requirements(model_name)
    except ValueError as e:
        return str(e)

    return (
        f"Fine-tuning dataset requirements for '{model_name}':\n"
        f"Dataset format: {req['dataset_format']}\n"
        f"Robot embodiment: {req['robot_type']}\n"
        f"Camera views expected: {req['expected_exterior_cameras']} exterior, "
        f"{req['expected_wrist_cameras']} wrist\n"
        f"State/action: {req['state_action']}\n\n"
        f"Suggested search: search_compatible_lerobot_datasets(query='{req['search_query_hint']}', "
        f"expected_robot_type='{req['robot_type']}')\n"
        f"{req['search_note']}\n\n"
        f"To check a specific candidate, call validate_lerobot_dataset(dataset_repo_id=..., "
        f"model_name='{model_name}') -- pass model_name, do NOT re-type "
        f"expected_exterior_cameras/expected_wrist_cameras yourself from this text; "
        f"model_name looks the same numbers up directly with no copying step to get wrong."
    )


@tool
def submit_finetune_run(dataset_pvc_name: str, exp_name: str, model_name: str = "pi05") -> str:
    """Start a fine-tuning run for a model against an already-staged dataset.

    This runs as a sequence of Kubernetes Jobs (per the model's recipe --
    e.g. train, evaluate for pi05), consuming real GPU-hours on the shared
    cluster for potentially hours. Only call this after you've discussed
    the recipe (which model, which dataset, roughly how long it'll take)
    with the user and they've explicitly said to proceed -- never call this
    speculatively. This tool only creates the *first* stage's Job; call
    get_finetune_run_status repeatedly afterward both to check progress and
    to advance the pipeline to its next stage once the current one succeeds.

    Args:
        dataset_pvc_name: The PVC name of an already-pull_dataset-staged
            dataset (e.g. 'dataset-my-droid-set-pvc').
        exp_name: Short experiment name, lowercase alphanumeric and hyphens
            (used as the K8s resource name prefix and the training run's
            own job/experiment name).
        model_name: Which fine-tuning recipe to use. Only 'pi05' exists so far.
    """
    core_api, batch_api = _get_clients()

    try:
        pvc = core_api.read_namespaced_persistent_volume_claim(
            name=dataset_pvc_name, namespace=settings.datasets_namespace
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"Dataset PVC '{dataset_pvc_name}' not found in '{settings.datasets_namespace}'. Pull it first with pull_dataset."
        return f"Could not read PVC '{dataset_pvc_name}': {e.reason}"

    dataset_repo_label = (pvc.metadata.labels or {}).get(DATASET_REPO_LABEL)
    if not dataset_repo_label:
        return f"PVC '{dataset_pvc_name}' isn't labeled as a dataset cache — was it created by pull_dataset?"
    dataset_repo_id = dataset_repo_label.replace("--", "/")

    try:
        stages = get_recipe(model_name, dataset_repo_id, exp_name)
    except ValueError as e:
        return str(e)

    try:
        core_api.create_namespaced_persistent_volume_claim(
            namespace=settings.datasets_namespace,
            body={
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": _checkpoint_pvc_name(exp_name), "labels": {FINETUNE_EXP_LABEL: exp_name}},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": "100Gi"}},
                    "storageClassName": "gp3-csi",
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to create checkpoint PVC: {e.reason}"

    first_stage = stages[0]
    try:
        job_name = _create_stage_job(
            batch_api, exp_name, model_name, dataset_repo_id, dataset_pvc_name, first_stage, stage_index=0
        )
    except client.exceptions.ApiException as e:
        if e.status == 409:
            return f"A fine-tuning run named '{exp_name}' already exists. Check get_finetune_run_status('{exp_name}')."
        return f"Failed to create stage Job: {e.reason}"

    return (
        f"Started fine-tuning '{model_name}' as experiment '{exp_name}' — stage "
        f"'{first_stage['name']}' running as Job '{job_name}' in "
        f"'{settings.datasets_namespace}'. This recipe has {len(stages)} stage(s): "
        f"{', '.join(s['name'] for s in stages)}. Call "
        f"get_finetune_run_status('{exp_name}') to check progress and advance to "
        f"the next stage once this one succeeds."
    )


@tool
def get_finetune_run_status(exp_name: str) -> str:
    """Check the status of a fine-tuning run started by submit_finetune_run.

    Reports the current stage's Job status. If that stage has succeeded and
    the recipe has a next stage, this call also creates the next stage's
    Job — call this tool repeatedly to both monitor and advance the run.

    Args:
        exp_name: The exp_name passed to submit_finetune_run.
    """
    core_api, batch_api = _get_clients()

    jobs = batch_api.list_namespaced_job(
        namespace=settings.datasets_namespace,
        label_selector=f"{FINETUNE_EXP_LABEL}={exp_name}",
    )
    if not jobs.items:
        return f"No fine-tuning run found for exp_name '{exp_name}' — has submit_finetune_run been called?"

    jobs_by_index = sorted(jobs.items, key=lambda j: int(j.metadata.labels.get(FINETUNE_STAGE_INDEX_LABEL, 0)))
    current_job = jobs_by_index[-1]
    current_labels = current_job.metadata.labels
    model_name = current_labels.get(FINETUNE_MODEL_LABEL, "pi05")
    dataset_repo_id = current_labels.get(FINETUNE_DATASET_REPO_LABEL, "").replace("--", "/")
    dataset_pvc_name = current_labels.get(FINETUNE_DATASET_PVC_LABEL, "")
    stage_name = current_labels.get(FINETUNE_STAGE_LABEL, "unknown")
    stage_index = int(current_labels.get(FINETUNE_STAGE_INDEX_LABEL, 0))

    status = current_job.status
    if status.succeeded:
        state = "succeeded"
    elif status.failed:
        state = "failed"
    elif status.active:
        state = "running"
    else:
        state = "pending"

    result = f"Fine-tuning run '{exp_name}' ({model_name}): stage '{stage_name}' is {state}."

    if state == "failed":
        pods = core_api.list_namespaced_pod(
            namespace=settings.datasets_namespace,
            label_selector=f"job-name={current_job.metadata.name}",
        )
        if pods.items:
            try:
                logs = core_api.read_namespaced_pod_log(
                    name=pods.items[0].metadata.name,
                    namespace=settings.datasets_namespace,
                    tail_lines=50,
                )
                result += f"\nLast 50 log lines:\n{logs}"
            except client.exceptions.ApiException:
                pass
        return result

    if state != "succeeded":
        return result

    try:
        stages = get_recipe(model_name, dataset_repo_id, exp_name)
    except ValueError as e:
        return f"{result} (Could not resolve recipe to check for a next stage: {e})"

    next_index = stage_index + 1
    if next_index >= len(stages):
        pods = core_api.list_namespaced_pod(
            namespace=settings.datasets_namespace,
            label_selector=f"job-name={current_job.metadata.name}",
        )
        eval_output = ""
        if pods.items:
            try:
                eval_output = core_api.read_namespaced_pod_log(
                    name=pods.items[0].metadata.name,
                    namespace=settings.datasets_namespace,
                    tail_lines=20,
                )
            except client.exceptions.ApiException:
                pass
        return (
            f"{result} All {len(stages)} stages complete. Checkpoint PVC: "
            f"'{_checkpoint_pvc_name(exp_name)}'.\nFinal stage log tail:\n{eval_output}"
        )

    next_stage = stages[next_index]
    try:
        job_name = _create_stage_job(
            batch_api, exp_name, model_name, dataset_repo_id, dataset_pvc_name, next_stage, stage_index=next_index
        )
    except client.exceptions.ApiException as e:
        if e.status == 409:
            return f"{result} Next stage '{next_stage['name']}' Job already exists — check again shortly."
        return f"{result} Failed to advance to next stage '{next_stage['name']}': {e.reason}"

    return f"{result} Advanced to stage '{next_stage['name']}' (Job '{job_name}')."
