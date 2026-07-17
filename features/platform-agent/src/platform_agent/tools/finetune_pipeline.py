"""Submits/polls fine-tuning runs as real KFP v2 pipelines against RHOAI's
Data Science Pipelines (DSPA), so they show up as Pipeline Runs in the RHOAI
dashboard instead of being invisible raw batch/v1 Jobs.

Auth: confirmed live that the DSPA's kube-rbac-proxy sidecar authorizes
purely via a K8s SubjectAccessReview -- get/create on
datasciencepipelinesapplications/api (resourceName "dspa") in this
namespace -- NOT an OpenShift OAuth browser flow, despite `enableOauth:
true` on the DSPA spec (that flag is for the external route used by human
dashboard users, not this in-cluster service). The platform-agent
ServiceAccount's own auto-mounted token, bound to the Data Science
Pipelines operator's own "ds-pipeline-user-access-dspa" Role
(platform/base/agent/rbac.yaml), is accepted directly as a kfp.Client
bearer token -- confirmed via a raw SubjectAccessReview call before writing
any of this.

DSPA reports dspVersion: v2 (odh-ml-pipelines-api-server-v2-rhel9 image) --
this uses the kfp>=2.0 SDK/DSL, not the older v1 kfp-tekton-based one RHOAI
used before it moved Data Science Pipelines onto native Kubeflow (Argo
Workflows) execution.
"""

import os
import tempfile
from pathlib import Path

import httpx
import kfp
from kfp import dsl
from kfp import kubernetes as kfp_kubernetes
from kfp.dsl import PipelineConfig

DSPA_HOST = "https://ds-pipeline-dspa.physical-ai.svc.cluster.local:8443"
DSPA_NAMESPACE = "physical-ai"
SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"

DASHBOARD_BASE_URL = os.environ.get("DASHBOARD_URL", "https://rh-ai.apps.emerg.pcbk.p1.openshiftapps.com")
DASHBOARD_RUNS_PATH = "develop-train/pipelines/runs"

GPU_NODE_SELECTOR_KEY = "nvidia.com/gpu.product"
GPU_NODE_SELECTOR_VALUE = "NVIDIA-L40S"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_URL", "https://mlflow.redhat-ods-applications.svc:8443")
MLFLOW_WORKSPACE = DSPA_NAMESPACE
MLFLOW_EXPERIMENT_NAME = "fine-tuning"


def _dspa_client() -> kfp.Client:
    with open(SA_TOKEN_PATH) as f:
        token = f.read().strip()
    return kfp.Client(host=DSPA_HOST, existing_token=token, namespace=DSPA_NAMESPACE, verify_ssl=False)


def _stage_component(stage: dict):
    """Wraps one of finetune_recipes.get_recipe()'s stage dicts (image,
    command, gpu) as a KFP v2 container component. A fresh component
    definition per stage (image/command baked into the closure) rather
    than one generic parameterized component -- these are already fully
    resolved Python strings by the time get_recipe() returns them, exactly
    like the raw Job body finetune.py used to build directly.

    dsl.container_component names the compiled template after the wrapped
    function's __name__, so the inner function is renamed to the stage's
    own name before decorating -- otherwise every stage would compile to
    the same generic "component" template name.
    """
    image = stage["image"]
    command = stage["command"]

    def component() -> dsl.ContainerSpec:
        return dsl.ContainerSpec(image=image, command=command)

    component.__name__ = stage["name"]
    component.__qualname__ = stage["name"]
    return dsl.container_component(component)


def submit_pipeline_run(
    exp_name: str,
    model_name: str,
    stages: list[dict],
    dataset_pvc_name: str,
    checkpoint_pvc_name: str,
    dataset_mount_path: str,
    checkpoint_mount_path: str,
) -> tuple[str, str]:
    """Compiles `stages` into a KFP v2 pipeline (one step per stage, in
    order, each depending on the previous) and submits a run. Returns
    (run_id, dashboard_url).

    Mounts are the same PVCs/paths the raw-Job version used -- dataset
    read-write (kfp.kubernetes.mount_pvc has no read-only flag in this SDK
    version; the training/eval scripts never write to it in practice, so
    this is a minor loss of defense-in-depth, not a functional gap) and
    checkpoint read-write. GPU stages get the same NVIDIA-L40S node
    selector the raw Jobs used.

    ttl_seconds_after_success (Argo's ttlStrategy.secondsAfterSuccess) has
    Argo delete the whole completed Workflow -- pods included -- 15 minutes
    after a successful finish, independent of the KFP run/execution/
    artifact records a Run's history lives in (confirmed live: deleting a
    run's pods doesn't affect its visibility in the dashboard). Left unset
    for failures so a failed run's pods/logs stick around to debug.
    """

    @dsl.pipeline(name=exp_name, pipeline_config=PipelineConfig(ttl_seconds_after_success=900))
    def pipeline():
        previous_task = None
        for stage in stages:
            component = _stage_component(stage)
            task = component()
            task.set_display_name(stage["name"])
            task.set_caching_options(False)
            task.set_env_variable("HF_HOME", "/tmp/hf_home")
            task.set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
            task.set_env_variable("MLFLOW_WORKSPACE", MLFLOW_WORKSPACE)
            kfp_kubernetes.use_secret_as_env(
                task, secret_name="huggingface-token", secret_key_to_env={"HF_TOKEN": "HF_TOKEN"}, optional=True
            )
            kfp_kubernetes.mount_pvc(task, pvc_name=dataset_pvc_name, mount_path=dataset_mount_path)
            kfp_kubernetes.mount_pvc(task, pvc_name=checkpoint_pvc_name, mount_path=checkpoint_mount_path)
            if stage["gpu"] > 0:
                task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(stage["gpu"])
                kfp_kubernetes.add_node_selector(task, label_key=GPU_NODE_SELECTOR_KEY, label_value=GPU_NODE_SELECTOR_VALUE)
            if previous_task is not None:
                task.after(previous_task)
            previous_task = task

    client = _dspa_client()
    with tempfile.TemporaryDirectory() as tmp_dir:
        ir_path = Path(tmp_dir) / "pipeline.yaml"
        kfp.compiler.Compiler().compile(pipeline_func=pipeline, package_path=str(ir_path))
        result = client.create_run_from_pipeline_package(
            pipeline_file=str(ir_path),
            run_name=exp_name,
            experiment_name="fine-tuning",
            namespace=DSPA_NAMESPACE,
            enable_caching=False,
        )
        _register_pipeline_version(client, ir_path, model_name, exp_name)

    dashboard_url = f"{DASHBOARD_BASE_URL}/{DASHBOARD_RUNS_PATH}/{DSPA_NAMESPACE}/runs"
    return result.run_id, dashboard_url


def _register_pipeline_version(client: kfp.Client, ir_path: Path, model_name: str, exp_name: str) -> None:
    """Registers this run's compiled IR under a per-model Pipeline (e.g.
    "pi05-finetune") so it's browsable under the dashboard's Pipelines tab,
    not just as a one-off Run. Decoupled from the run submitted above --
    that still runs from the raw IR file directly -- so a failure here
    (e.g. a version-name collision) can't break an actual fine-tuning
    submission over a discoverability nice-to-have.
    """
    pipeline_name = f"{model_name}-finetune"
    try:
        pipeline_id = client.get_pipeline_id(pipeline_name)
        if pipeline_id is None:
            client.upload_pipeline(pipeline_package_path=str(ir_path), pipeline_name=pipeline_name)
        else:
            client.upload_pipeline_version(
                pipeline_package_path=str(ir_path), pipeline_version_name=exp_name, pipeline_id=pipeline_id
            )
    except Exception:
        pass


def get_pipeline_run_status(run_id: str) -> str:
    """Human-readable status for a run started by submit_pipeline_run --
    overall state plus per-task state, matching the level of detail the
    old raw-Job get_finetune_run_status reported.
    """
    client = _dspa_client()
    try:
        run = client.get_run(run_id)
    except Exception as e:
        return f"Could not read pipeline run '{run_id}': {e}"

    state = run.state
    result = f"Pipeline run '{run.display_name}' ({run_id}): {state}."

    task_details = getattr(run.run_details, "task_details", None) if run.run_details else None
    if task_details:
        task_lines = [f"  - {t.display_name}: {t.state}" for t in task_details if t.display_name]
        if task_lines:
            result += "\nStages:\n" + "\n".join(task_lines)

    return result


def get_finetune_eval_metrics(exp_name: str) -> dict | None:
    """Looks up the evaluate stage's logged metrics for this experiment from
    MLflow, by run name -- the evaluate stage (finetune_recipes.py's
    _evaluate_script) logs its own run there under run_name=exp_name, since
    that's the only identifier both sides can agree on without persisting a
    new id anywhere (the same reasoning exp_name/dataset_pvc_name naming
    already relies on elsewhere in this module).

    Returns None (best-effort -- this is a nice-to-have status enrichment,
    never a reason to fail get_finetune_run_status) if MLflow is
    unreachable, or if the "fine-tuning" experiment or this run doesn't
    exist yet (e.g. the evaluate stage hasn't completed, or hasn't run at
    all on a pre-MLflow-logging pipeline).
    """
    try:
        with open(SA_TOKEN_PATH) as f:
            token = f.read().strip()
        headers = {"Authorization": f"Bearer {token}", "X-MLFLOW-WORKSPACE": MLFLOW_WORKSPACE}
        with httpx.Client(timeout=10.0, verify=False, headers=headers) as http:
            exp_resp = http.get(
                f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/get-by-name",
                params={"experiment_name": MLFLOW_EXPERIMENT_NAME},
            )
            if exp_resp.status_code != 200:
                return None
            experiment_id = exp_resp.json()["experiment"]["experiment_id"]

            search_resp = http.post(
                f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search",
                json={
                    "experiment_ids": [experiment_id],
                    "filter": f"tags.\"mlflow.runName\" = '{exp_name}'",
                    "max_results": 1,
                },
            )
            search_resp.raise_for_status()
            runs = search_resp.json().get("runs", [])
            if not runs:
                return None

            run = runs[0]
            metrics = {m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])}
            return {"status": run["info"]["status"], "metrics": metrics}
    except Exception:
        return None
