from langchain_core.tools import tool
from kubernetes import client

from platform_agent.config import settings
from platform_agent.tools.datasets import dataset_repo_id_from_pvc
from platform_agent.tools.finetune_pipeline import (
    get_finetune_eval_metrics,
    get_pipeline_run_state,
    get_pipeline_run_status,
    log_finetune_run_params,
    submit_pipeline_run,
)
from platform_agent.tools.finetune_recipes import CHECKPOINT_MOUNT_PATH, dataset_mount_path, get_recipe

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

    Resubmitting under an exp_name that already has a run is normally
    refused -- but if that prior run's state is FAILED (e.g. after fixing a
    dataset-format error with convert_dataset_to_v3), this proceeds and
    reuses the same checkpoint PVC, overwriting the failed attempt's output.

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

    dataset_repo_id = dataset_repo_id_from_pvc(pvc)
    if not dataset_repo_id:
        return f"PVC '{dataset_pvc_name}' isn't labeled as a dataset cache — was it created by pull_dataset?"

    try:
        stages, recipe_params = get_recipe(model_name, dataset_repo_id, exp_name)
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
        if existing_run_id and get_pipeline_run_state(existing_run_id) != "FAILED":
            return (
                f"A fine-tuning run named '{exp_name}' already exists (pipeline run "
                f"'{existing_run_id}'). Check get_finetune_run_status('{exp_name}')."
            )
        # existing_run_id's prior run reached the terminal FAILED state (or
        # was never recorded at all) -- allow retrying under the same
        # exp_name. Reuse the checkpoint PVC as-is rather than deleting and
        # recreating it: the failed run's stage pods are deliberately left
        # running for debugging (see submit_pipeline_run's ttl_seconds
        # docstring), so they'd still be mounting this PVC and a delete would
        # just hang in Terminating behind the pvc-protection finalizer.
        # lerobot-train's own --output_dir semantics already overwrite a
        # prior run's contents on a fresh run, so this is safe.

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

    log_finetune_run_params(exp_name, recipe_params, kfp_run_id=run_id)

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
        f"{', '.join(s['name'] for s in stages)}. Find it by name ('{exp_name}') in the "
        f"RHOAI dashboard's pipeline runs list at {dashboard_url}, or call "
        f"get_finetune_run_status('{exp_name}') to check progress."
    )


@tool
def get_finetune_run_status(exp_name: str) -> str:
    """Check the status of a fine-tuning run started by submit_finetune_run.

    Reports the pipeline run's overall state and per-stage state. The
    pipeline advances through its own stages on its own -- this is a
    read-only status check, not something that needs to be called
    repeatedly to make progress happen (unlike the old raw-Job version).
    Also includes the resolved recipe (dataset, step count, batch size,
    episode split, ...) logged to MLflow at submission time, plus mean and
    per-episode action-MSE eval results once the evaluate stage has logged
    them there too -- eval results are omitted if that stage hasn't run yet
    or hasn't finished; the recipe itself is omitted only if the
    submission-time MLflow logging failed.

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
    result = f"{status}\nCheckpoint PVC: '{checkpoint_pvc_name}'."

    eval_results = get_finetune_eval_metrics(exp_name)
    if eval_results:
        recipe_params = eval_results.get("params")
        if recipe_params:
            result += "\nRecipe: " + ", ".join(f"{k}={v}" for k, v in recipe_params.items())

        metrics = eval_results["metrics"]
        per_episode = sorted(
            ((k[len("action_mse_ep") :], v) for k, v in metrics.items() if k.startswith("action_mse_ep")),
            key=lambda kv: int(kv[0]),
        )
        result += f"\nEval results (MLflow run '{exp_name}', {eval_results['status']}):"
        if "mean_action_mse" in metrics:
            result += f"\n  Mean action MSE: {float(metrics['mean_action_mse']):.4f}"
        if per_episode:
            per_episode_str = ", ".join(f"{ep}={float(mse):.4f}" for ep, mse in per_episode)
            result += f"\n  Per-episode MSE: {per_episode_str}"

    return result


@tool
def list_finetune_runs() -> str:
    """List fine-tuning experiments started on this cluster. Call this when
    asked about fine-tuning runs in general (e.g. "what's running",
    "any fine-tunes in progress") or when the exact exp_name isn't known --
    get_finetune_run_status requires the exact exp_name and has no other
    way to look one up, so without this, a forgotten exp_name is
    unrecoverable.

    Shows each experiment's exp_name and pipeline run state (best-effort --
    "no pipeline run recorded" if submit_finetune_run failed partway
    through, "status unavailable" if the pipeline run can't be reached).
    """
    core_api = _get_core_api()
    pvcs = core_api.list_namespaced_persistent_volume_claim(
        namespace=settings.datasets_namespace,
        label_selector=FINETUNE_EXP_LABEL,
    )

    if not pvcs.items:
        return "No fine-tuning runs found."

    lines = []
    for pvc in pvcs.items:
        exp_name = (pvc.metadata.labels or {}).get(FINETUNE_EXP_LABEL, "unknown")
        run_id = (pvc.metadata.annotations or {}).get(FINETUNE_RUN_ID_ANNOTATION)
        if not run_id:
            state = "no pipeline run recorded"
        else:
            try:
                state = get_pipeline_run_status(run_id).splitlines()[0]
            except Exception:
                state = "status unavailable"
        lines.append(f"- {exp_name}: {state}")

    return "Fine-tuning runs:\n" + "\n".join(lines)
