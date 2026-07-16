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

import tempfile
from pathlib import Path

import kfp
from kfp import dsl
from kfp import kubernetes as kfp_kubernetes

DSPA_HOST = "https://ds-pipeline-dspa.physical-ai.svc.cluster.local:8443"
DSPA_NAMESPACE = "physical-ai"
SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"

# RHOAI dashboard's project-scoped pipeline-run URL pattern -- best-effort
# deep link (not confirmed against every RHOAI version), included as a
# convenience only. "Data Science Projects > physical-ai > Pipelines" in
# the dashboard is the reliable fallback if this exact path is stale.
DASHBOARD_BASE_URL = "https://rhods-dashboard-redhat-ods-applications.apps.emerg.pcbk.p1.openshiftapps.com"

GPU_NODE_SELECTOR_KEY = "nvidia.com/gpu.product"
GPU_NODE_SELECTOR_VALUE = "NVIDIA-L40S"


def _dspa_client() -> kfp.Client:
    with open(SA_TOKEN_PATH) as f:
        token = f.read().strip()
    # verify_ssl=False matches this codebase's existing handling of other
    # in-cluster HTTPS services with self-signed/internal certs (e.g.
    # DefaultHttpxClient(verify=False) in agent.py) -- the DSPA spec's own
    # caBundleFileMountPath is empty, and there's no simpler verified CA
    # path for kube-rbac-proxy's serving cert specifically.
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
    """

    @dsl.pipeline(name=exp_name)
    def pipeline():
        previous_task = None
        for stage in stages:
            component = _stage_component(stage)
            task = component()
            task.set_display_name(stage["name"])
            task.set_caching_options(False)
            # Same env the raw-Job version set (finetune.py's old
            # _create_stage_job) -- HF_TOKEN specifically is not optional in
            # practice: pi0.5's tokenizer processor loads config from
            # PaliGemma's gated HF repo and 401s without it (confirmed live
            # earlier this session). optional=True so stages that don't need
            # it (none currently, but matches the old behavior) don't fail
            # if the secret is ever absent.
            task.set_env_variable("HF_HOME", "/tmp/hf_home")
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

    dashboard_url = f"{DASHBOARD_BASE_URL}/pipelineRuns/{DSPA_NAMESPACE}/pipelineRun/view/{result.run_id}"
    return result.run_id, dashboard_url


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
