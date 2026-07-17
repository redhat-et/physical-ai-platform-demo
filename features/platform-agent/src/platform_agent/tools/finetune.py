from langchain_core.tools import tool
from kubernetes import client

from platform_agent.config import settings
from platform_agent.tools.datasets import DATASET_REPO_LABEL
from platform_agent.tools.finetune_pipeline import get_pipeline_run_status, submit_pipeline_run
from platform_agent.tools.finetune_recipes import CHECKPOINT_MOUNT_PATH, dataset_mount_path, get_recipe, get_requirements

FINETUNE_EXP_LABEL = "physical-ai.io/finetune-exp"
FINETUNE_RUN_ID_ANNOTATION = "physical-ai.io/kfp-run-id"


def _get_core_api():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api()


def _checkpoint_pvc_name(exp_name: str) -> str:
    return f"finetune-{exp_name}-checkpoint-pvc"


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

    This runs as a real KFP pipeline (train -> evaluate for pi05) against
    RHOAI's Data Science Pipelines, consuming real GPU-hours on the shared
    cluster for potentially hours. Only call this after you've discussed
    the recipe (which model, which dataset, roughly how long it'll take)
    with the user and they've explicitly said to proceed -- never call this
    speculatively. The pipeline advances through its own stages on its
    own (no manual "create the next stage" step needed); call
    get_finetune_run_status afterward to check progress.

    Args:
        dataset_pvc_name: The PVC name of an already-pull_dataset-staged
            dataset (e.g. 'dataset-my-droid-set-pvc').
        exp_name: Short experiment name, lowercase alphanumeric and hyphens
            (used as the K8s resource name prefix and the pipeline run's name).
        model_name: Which fine-tuning recipe to use. Only 'pi05' exists so far.
    """
    core_api = _get_core_api()

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

    checkpoint_pvc_name = _checkpoint_pvc_name(exp_name)
    try:
        core_api.create_namespaced_persistent_volume_claim(
            namespace=settings.datasets_namespace,
            body={
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": checkpoint_pvc_name, "labels": {FINETUNE_EXP_LABEL: exp_name}},
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
        existing_pvc = core_api.read_namespaced_persistent_volume_claim(
            name=checkpoint_pvc_name, namespace=settings.datasets_namespace
        )
        existing_run_id = (existing_pvc.metadata.annotations or {}).get(FINETUNE_RUN_ID_ANNOTATION)
        if existing_run_id:
            return (
                f"A fine-tuning run named '{exp_name}' already exists (pipeline run "
                f"'{existing_run_id}'). Check get_finetune_run_status('{exp_name}')."
            )

    try:
        run_id, dashboard_url = submit_pipeline_run(
            exp_name=exp_name,
            model_name=model_name,
            stages=stages,
            dataset_pvc_name=dataset_pvc_name,
            checkpoint_pvc_name=checkpoint_pvc_name,
            dataset_mount_path=dataset_mount_path(dataset_repo_id),
            checkpoint_mount_path=CHECKPOINT_MOUNT_PATH,
        )
    except Exception as e:
        return f"Failed to submit fine-tuning pipeline: {e}"

    try:
        core_api.patch_namespaced_persistent_volume_claim(
            name=checkpoint_pvc_name,
            namespace=settings.datasets_namespace,
            body={"metadata": {"annotations": {FINETUNE_RUN_ID_ANNOTATION: run_id}}},
        )
    except client.exceptions.ApiException:
        pass

    return (
        f"Started fine-tuning '{model_name}' as experiment '{exp_name}' — pipeline run "
        f"'{run_id}' submitted to Data Science Pipelines with {len(stages)} stage(s): "
        f"{', '.join(s['name'] for s in stages)}. View it in the RHOAI dashboard at "
        f"{dashboard_url}, or call get_finetune_run_status('{exp_name}') to check progress."
    )


@tool
def get_finetune_run_status(exp_name: str) -> str:
    """Check the status of a fine-tuning run started by submit_finetune_run.

    Reports the pipeline run's overall state and per-stage state. The
    pipeline advances through its own stages on its own -- this is a
    read-only status check, not something that needs to be called
    repeatedly to make progress happen (unlike the old raw-Job version).

    Args:
        exp_name: The exp_name passed to submit_finetune_run.
    """
    core_api = _get_core_api()

    checkpoint_pvc_name = _checkpoint_pvc_name(exp_name)
    try:
        pvc = core_api.read_namespaced_persistent_volume_claim(
            name=checkpoint_pvc_name, namespace=settings.datasets_namespace
        )
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"No fine-tuning run found for exp_name '{exp_name}' — has submit_finetune_run been called?"
        return f"Could not read checkpoint PVC '{checkpoint_pvc_name}': {e.reason}"

    run_id = (pvc.metadata.annotations or {}).get(FINETUNE_RUN_ID_ANNOTATION)
    if not run_id:
        return (
            f"Checkpoint PVC for '{exp_name}' exists but has no pipeline run recorded — "
            f"submit_finetune_run may have failed partway through."
        )

    status = get_pipeline_run_status(run_id)
    return f"{status}\nCheckpoint PVC: '{checkpoint_pvc_name}'."
