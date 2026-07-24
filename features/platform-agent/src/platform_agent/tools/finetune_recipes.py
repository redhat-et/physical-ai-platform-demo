"""Fine-tuning recipes: per-model ordered stage lists for submit_finetune_run.

A recipe's stages run as separate Kubernetes Jobs, in order, each overriding
the same training image's command -- see the platform_agent fine-tuning
plan for why (no Kubeflow/Tekton available on this cluster; Tekton's CRDs
aren't installed despite pre-provisioned RBAC, and KFP/DSPA would need a new
SDK dependency and an unconfirmed auth story). Kept as a plain Python
constant, not a generic multi-architecture schema, until a second recipe
exists to generalize from.

pi0.5's recipe trains via LeRobot's own native `lerobot-train` CLI (pure
PyTorch, https://huggingface.co/docs/lerobot/pi05), not openpi's JAX
scripts -- confirmed that `lerobot/pi05_base` (the checkpoint our
openpi-runtime already serves live) is exactly LeRobot's own
PreTrainedPolicy save format (config.json + model.safetensors +
pre/post-processor json), so a lerobot-train checkpoint should load with
zero conversion via that same serving setup. No custom TrainConfig shim
needed either -- lerobot-train is a normal CLI.
"""

import json

from platform_agent.tools.datasets import _fetch_lerobot_info

LEROBOT_IMAGE = "huggingface/lerobot-gpu:latest"

DATASET_MOUNT_ROOT = "/mnt/lerobot_home"
CHECKPOINT_MOUNT_PATH = "/mnt/checkpoint"

# The base checkpoint to fine-tune from -- same HF repo our pi05
# InferenceService already downloads and serves (platform/base/models/pi05/
# model-download-job.yaml, on the unmerged origin/feat/add-pi05-model
# branch we don't touch). Fine-tuning from this exact checkpoint, in the
# exact same checkpoint format, is what makes the "no conversion needed"
# assumption hold.
PI05_PRETRAINED_PATH = "lerobot/pi05_base"

# Per-model dataset-compatibility requirements (embodiment, camera counts,
# action space, dataset format) live in the `datasets` skill
# (platform_agent/skills/datasets.md), not here -- that content needs to
# express real uncertainty/caveats a Python dict can't, and shouldn't imply
# machine-checked ground truth it isn't.

# Shared between _train_script's actual --policy.normalization_mapping flag
# and get_recipe's logged params -- a single source of truth so the two can't
# drift apart.
NORMALIZATION_MAPPING = '{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'

# Confirmed live via lerobot-train against lerobot/pi05_base with no
# n_action_steps override: PI05Config.validate() reports its own default as
# 50, matching its default chunk_size. Used only to decide whether
# _train_script needs to auto-cap n_action_steps when chunk_size is lowered
# without an explicit n_action_steps -- see _train_script's docstring.
PI05_BASE_DEFAULT_N_ACTION_STEPS = 50


def dataset_mount_path(dataset_repo_id: str) -> str:
    """Where a dataset PVC is mounted in every finetune stage's pod -- single
    source of truth shared between the actual volume mount (kfp.kubernetes
    mount_pvc call in finetune_pipeline.py's submit_pipeline_run) and the
    training/eval scripts below that need to tell lerobot-train/
    LeRobotDataset the same path via --dataset.root / root=.

    That explicit root is not optional: LeRobotDatasetMetadata only trusts a
    local path when it's passed as `root` -- otherwise it checks for
    `<path>/.cache/huggingface/download/`, the marker left by
    snapshot_download(local_dir=...) (exactly what pull_dataset uses), and
    treats its PRESENCE as "old, non-revision-safe download, re-fetch from
    the Hub instead" (confirmed live: without --dataset.root, lerobot-train
    ignored this mount entirely and tried to re-download over the network,
    which then failed anyway since the mount is read-only).
    """
    return f"{DATASET_MOUNT_ROOT}/{dataset_repo_id}"


# How many trailing episodes to reserve for eval. Previously the eval script
# computed its own "held_out = last 5 episodes" at runtime while the train
# script had no episode filter at all -- training used ALL episodes, so
# eval's "held-out" set had already been seen during training. That made the
# eval numbers an in-sample fit check, not a real generalization measure.
NUM_EVAL_EPISODES = 5


def split_episodes(total_episodes: int) -> tuple[list[int], list[int]]:
    """Split a dataset's episodes into a training set and a genuinely
    held-out eval set. The eval episodes get passed to --dataset.episodes
    at training time to exclude them, and the exact same list gets passed
    to the eval script -- one computation, shared by both stages, so they
    can't drift apart the way the old two-independent-computations version
    could.
    """
    num_eval = min(NUM_EVAL_EPISODES, total_episodes - 1) if total_episodes > 1 else 0
    if num_eval < 1:
        raise ValueError(
            f"Dataset has only {total_episodes} episode(s) -- too few to split into a "
            f"non-empty train set and a non-empty eval set."
        )
    train_episodes = list(range(total_episodes - num_eval))
    eval_episodes = list(range(total_episodes - num_eval, total_episodes))
    return train_episodes, eval_episodes


def _checkpoint_dir(exp_name: str) -> str:
    """lerobot-train's own convention: {output_dir}/checkpoints/last/pretrained_model
    always points at the most recent checkpoint (a directory containing
    config.json + model.safetensors + pre/post-processor json -- the same
    layout as lerobot/pi05_base itself)."""
    return f"{CHECKPOINT_MOUNT_PATH}/{exp_name}/checkpoints/last/pretrained_model"


def _train_script(
    dataset_repo_id: str,
    exp_name: str,
    num_train_steps: int,
    batch_size: int,
    train_episodes: list[int],
    chunk_size: int | None = None,
    n_action_steps: int | None = None,
    empty_cameras: int | None = None,
) -> tuple[str, int | None]:
    """Training stage script: runs lerobot-train directly -- a plain CLI, no
    custom Python config-construction shim needed unlike the old openpi-based
    recipe. Uses the MEAN_STD normalization override instead of the
    QUANTILES preprocessing script, for a simpler first pass (see plan's
    "Open risks" re: fine-tune quality tradeoff).

    huggingface/lerobot-gpu:latest already ships lerobot with pi0.5 support
    preinstalled (confirmed live: `import lerobot.policies.pi05` and
    PI05Policy both import with zero extra installs) in a uv-managed venv
    that has no `pip` binary at all -- a prior version of this script ran
    `pip install -q "lerobot[pi]"` here, which failed immediately with
    "pip: command not found" (exit 127) before training ever started. Do
    NOT re-add a pip/uv install line for this -- it's unnecessary.

    Confirmed live (full dry run, actual training steps executing on GPU)
    that three more fixes were needed beyond removing pip install:
    --dataset.root (without it, LeRobotDatasetMetadata ignores the mounted
    PVC entirely -- see dataset_mount_path's docstring -- and tries to
    re-download over the network, failing on the read-only mount);
    --policy.push_to_hub=false (cfg.validate() otherwise demands a
    --policy.repo_id to push the checkpoint to the Hub); and HF_TOKEN in the
    pod env (finetune_pipeline.py's submit_pipeline_run -- pi0.5's tokenizer
    processor loads config from PaliGemma's gated HF repo and 401s without it).

    STILL NEEDS VERIFICATION: whether lerobot-train handles DROID's
    camera/state layout correctly over a full 3000-step run (only the first
    ~10 steps were observed directly), and whether train_expert_only fits a
    single 48GB L40S for the full run (only ~25GB used in early steps).

    train_episodes excludes whatever split_episodes reserved for eval, via
    --dataset.episodes -- confirmed this flag's list-literal CLI syntax
    against LeRobot's own Makefile/CI examples (--dataset.episodes="[0]").
    Without this, the eval stage's "held-out" episodes were actually part of
    the training set the whole time (see split_episodes' docstring).

    chunk_size/empty_cameras are opt-in overrides for datasets that don't
    share DROID's fps or camera count -- confirmed real flags via
    `lerobot-train --policy.type=pi05 --help` on this same image, and a full
    (non-training) config-resolution dry run against a real --dataset.root
    confirmed they parse and resolve correctly together. A different fps
    changes the real-world time a fixed chunk_size covers; a dataset with
    fewer camera views than pi05_base's pretrained input_features needs
    padding via empty_cameras. Left unset (the default), the generated
    command is byte-identical to the pre-existing droid_100-only script,
    which needs neither.

    There used to be a rename_map flag here too, to remap a dataset's own
    camera key names to pi05_base's pretrained naming (e.g. 'world_camera'
    -> 'base_0_rgb'). Removed after confirming live it actively breaks
    training rather than helping: it renames the keys the DataLoader yields
    at batch time, but cfg.input_features (what PI05Policy._preprocess_images
    checks the batch against) gets resolved from the dataset's RAW,
    un-renamed meta/info.json names earlier in argument parsing -- the two
    sides then share zero key names, so every training step failed with
    "All image features are missing from the batch". Confirmed unnecessary
    besides: pretrained weight transfer from lerobot/pi05_base doesn't need
    matching camera key names at all ("Remapped 812 state dict keys / All
    keys loaded successfully" happened fine using a dataset's own raw
    camera names) -- that transfer is positional/structural, not
    name-matched. A dataset's own camera keys should just be left as-is.

    chunk_size and n_action_steps are related but not the same knob:
    chunk_size changes what the flow-matching loss actually supervises the
    model to predict (a training-time hyperparameter), while
    n_action_steps only controls how many of those predicted steps get
    executed before the policy replans against a fresh observation (a
    receding-horizon control choice pi05's own inference loop makes --
    confirmed live that n_action_steps=15 with chunk_size=50 parses and
    resolves fine, i.e. executing a short prefix of a longer prediction is
    a normal, valid combination, not a fallback). The one hard constraint
    (confirmed live via PI05Config.validate(), which pi05_base hits at its
    own default of 50/50): n_action_steps must not exceed chunk_size --
    "n_action_steps (50) cannot be greater than chunk_size (30)". So this
    only auto-caps n_action_steps down to chunk_size when the caller
    lowered chunk_size below 50 (pi05_base's confirmed pretrained default)
    without giving an explicit n_action_steps of their own -- an explicit
    n_action_steps always wins, letting a caller deliberately keep
    replanning more frequent than the prediction horizon.

    Returns (script, effective_n_action_steps) rather than just the script --
    get_recipe logs the latter into its MLflow params so an auto-capped run's
    provenance still records the value that actually got passed to
    lerobot-train, not just whatever the caller (or lack thereof) supplied.
    """
    optional_flags = ""
    if chunk_size is not None:
        optional_flags += f"    --policy.chunk_size={chunk_size} \\\n"
    effective_n_action_steps = n_action_steps
    if effective_n_action_steps is None and chunk_size is not None and chunk_size < PI05_BASE_DEFAULT_N_ACTION_STEPS:
        effective_n_action_steps = chunk_size
    if effective_n_action_steps is not None:
        optional_flags += f"    --policy.n_action_steps={effective_n_action_steps} \\\n"
    if empty_cameras is not None:
        optional_flags += f"    --policy.empty_cameras={empty_cameras} \\\n"

    script = f"""\
set -e
export HOME=/tmp
export HF_LEROBOT_HOME={DATASET_MOUNT_ROOT}
lerobot-train \\
    --dataset.repo_id={dataset_repo_id} \\
    --dataset.root={dataset_mount_path(dataset_repo_id)} \\
    --dataset.episodes="{train_episodes}" \\
    --policy.type=pi05 \\
    --policy.push_to_hub=false \\
    --policy.pretrained_path={PI05_PRETRAINED_PATH} \\
    --policy.train_expert_only=true \\
    --policy.gradient_checkpointing=true \\
    --policy.dtype=bfloat16 \\
    --policy.device=cuda \\
    --policy.normalization_mapping='{NORMALIZATION_MAPPING}' \\
{optional_flags}    --batch_size={batch_size} \\
    --steps={num_train_steps} \\
    --output_dir={CHECKPOINT_MOUNT_PATH}/{exp_name} \\
    --job_name={exp_name} \\
    --wandb.enable=false
"""
    return script, effective_n_action_steps


def _evaluate_script(dataset_repo_id: str, exp_name: str, eval_episodes: list[int]) -> str:
    """Offline, self-contained evaluation -- no dependency on
    robotics-playground/Isaac Lab or any external service. Loads the
    fine-tuned checkpoint, runs it against held-out episodes from the
    staged dataset (using the checkpoint's own saved pre/post-processors,
    since pi0.5 needs its language inputs tokenized the same way it was
    trained), and reports per-episode/mean action-prediction error as a
    smoke test rather than a task-success measure.

    Also logs those metrics to MLflow (via its REST API over stdlib
    urllib, bearer-token + workspace auth -- see finetune_pipeline.py's
    MLFLOW_TRACKING_URI) so they outlive this stage pod's short lifetime,
    and are queryable later by finetune.py's get_finetune_run_status.
    Best-effort: wrapped in its own try/except so an MLflow hiccup can't
    fail the eval stage itself.
    """
    checkpoint_dir = _checkpoint_dir(exp_name)
    eval_script = f"""\
set -e
export HOME=/tmp
export HF_LEROBOT_HOME={DATASET_MOUNT_ROOT}
cat > /tmp/run_eval.py << 'PYEOF'
import torch
import numpy as np
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CHECKPOINT_DIR = "{checkpoint_dir}"
EXP_NAME = "{exp_name}"
DATASET_REPO_ID = "{dataset_repo_id}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PI05Policy.from_pretrained(CHECKPOINT_DIR).to(device).eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=CHECKPOINT_DIR,
    preprocessor_overrides={{"device_processor": {{"device": str(device)}}}},
)

dataset = LeRobotDataset(DATASET_REPO_ID, root="{dataset_mount_path(dataset_repo_id)}")
held_out = {eval_episodes}
print(f"Evaluating against held-out episodes: {{held_out}}")

episode_frame_ranges = dataset.meta.episodes

errors = []
for ep_idx in held_out:
    policy.reset()
    from_idx = episode_frame_ranges["dataset_from_index"][ep_idx]
    to_idx = episode_frame_ranges["dataset_to_index"][ep_idx]
    frame_errors = []
    for frame_idx in range(from_idx, to_idx):
        ep = dataset[frame_idx]
        ground_truth = np.asarray(ep["action"])
        batch = {{k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in ep.items() if k != "action"}}
        batch = preprocessor(batch)
        with torch.no_grad():
            predicted = policy.select_action(batch)
        predicted = postprocessor(predicted).cpu().numpy().squeeze()
        frame_errors.append(float(np.mean((predicted - ground_truth) ** 2)))
    err = sum(frame_errors) / len(frame_errors)
    errors.append(err)
    print(f"episode {{ep_idx}}: mean action MSE over {{len(frame_errors)}} frames = {{err:.4f}}")

mean_mse = sum(errors) / len(errors)
print(f"EVAL_MEAN_ACTION_MSE={{mean_mse:.4f}}")
print("EVAL_SMOKE_TEST=PASS")
"""
    mlflow_logging = """
try:
    import json
    import os
    import ssl
    import time
    import urllib.error
    import urllib.request

    with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as _f:
        _sa_token = _f.read().strip()
    _mlflow_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_sa_token}",
        "X-MLFLOW-WORKSPACE": os.environ["MLFLOW_WORKSPACE"],
    }
    _no_verify_ctx = ssl.create_default_context()
    _no_verify_ctx.check_hostname = False
    _no_verify_ctx.verify_mode = ssl.CERT_NONE

    def _mlflow_request(method, path, payload=None):
        url = os.environ["MLFLOW_TRACKING_URI"] + path
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=_mlflow_headers)
        with urllib.request.urlopen(req, timeout=10, context=_no_verify_ctx) as resp:
            return json.loads(resp.read())

    try:
        experiment_id = _mlflow_request(
            "GET", "/api/2.0/mlflow/experiments/get-by-name?experiment_name=fine-tuning"
        )["experiment"]["experiment_id"]
    except urllib.error.HTTPError:
        experiment_id = _mlflow_request(
            "POST", "/api/2.0/mlflow/experiments/create", {"name": "fine-tuning"}
        )["experiment_id"]

    now_ms = int(time.time() * 1000)

    # submit_finetune_run's log_finetune_run_params already created this run
    # (run_name=EXP_NAME) at submission time to record the recipe params
    # before training even started -- find it and append metrics there
    # instead of creating a second run. Only create one here as a fallback,
    # e.g. if that submission-time logging failed.
    search_result = _mlflow_request(
        "POST",
        "/api/2.0/mlflow/runs/search",
        {
            "experiment_ids": [experiment_id],
            "filter": f"tags.\"mlflow.runName\" = '{EXP_NAME}'",
            "max_results": 1,
        },
    )
    existing_runs = search_result.get("runs", [])
    if existing_runs:
        run_id = existing_runs[0]["info"]["run_id"]
    else:
        run_id = _mlflow_request(
            "POST",
            "/api/2.0/mlflow/runs/create",
            {"experiment_id": experiment_id, "run_name": EXP_NAME, "start_time": now_ms},
        )["run"]["info"]["run_id"]

    metrics = [{"key": "mean_action_mse", "value": mean_mse, "timestamp": now_ms, "step": 0}]
    for ep_idx, err in zip(held_out, errors):
        metrics.append({"key": f"action_mse_ep{ep_idx}", "value": err, "timestamp": now_ms, "step": 0})

    _mlflow_request(
        "POST",
        "/api/2.0/mlflow/runs/log-batch",
        {"run_id": run_id, "metrics": metrics},
    )
    _mlflow_request(
        "POST",
        "/api/2.0/mlflow/runs/update",
        {"run_id": run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)},
    )
    print("Logged eval results to MLflow.")
except Exception as e:
    print(f"WARNING: failed to log eval results to MLflow: {e}")
PYEOF
cd /tmp && python3 run_eval.py
"""
    return eval_script + mlflow_logging


def get_recipe(
    model_name: str,
    dataset_repo_id: str,
    exp_name: str,
    dataset_subset: str | None = None,
    chunk_size: int | None = None,
    n_action_steps: int | None = None,
    empty_cameras: int | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Returns the ordered stage list for a model's fine-tuning recipe, plus
    the resolved recipe as a flat dict of MLflow-safe (string-valued) params
    -- submit_finetune_run logs this to MLflow at submission time so a run's
    exact provenance (dataset, step count, batch size, episode split, ...) is
    recoverable later even if these hardcoded values change in a future
    commit, or the run fails before the evaluate stage would otherwise be the
    only thing writing to MLflow at all.

    Each stage: name, image, command (list, passed to bash -c), gpu (int
    GPUs requested; 0 means no nodeSelector/GPU resource added).

    Called twice per run (submit_finetune_run for stage 0, then
    get_finetune_run_status again when advancing to stage 1) -- fetching
    total_episodes fresh each time rather than caching it is deliberate,
    since re-deriving the same split both times is what keeps
    train_episodes/eval_episodes identical across both calls without having
    to persist the split anywhere.

    dataset_subset: for a PVC pulled from a repo that bundles several
    independent LeRobot datasets as subfolders (see pull_dataset and
    split_dataset_repo_id in datasets.py) rather than one dataset per repo,
    which subfolder within that already-staged PVC to train on. Appended
    to dataset_repo_id (as "{dataset_repo_id}/{dataset_subset}") to build
    the effective identifier _fetch_lerobot_info/_train_script/
    _evaluate_script actually use for --dataset.root -- this is the one
    place that composition happens; the PVC mount path submit_finetune_run
    builds separately stays based on the plain dataset_repo_id, since the
    PVC holds the whole repo regardless of which subset a given run trains
    on.

    chunk_size/n_action_steps/empty_cameras: passed straight through to
    _train_script (see its docstring, including why rename_map was removed
    from here entirely rather than kept as an option) -- only needed for
    datasets whose fps or camera count differs from droid_100's. The eval
    stage doesn't need them separately: it loads the fine-tuned
    checkpoint's own saved config, which already has these baked in.
    """
    if model_name != "pi05":
        raise ValueError(f"No fine-tuning recipe for '{model_name}' -- only 'pi05' is defined so far.")

    effective_dataset_id = f"{dataset_repo_id}/{dataset_subset}" if dataset_subset else dataset_repo_id

    info = _fetch_lerobot_info(effective_dataset_id)
    if isinstance(info, str):
        raise ValueError(f"Could not resolve recipe for '{effective_dataset_id}': {info}")
    train_episodes, eval_episodes = split_episodes(info["total_episodes"])

    # Temporarily reduced from 3_000 -- at the measured ~5.4s/step pace on a
    # single L40S, 3_000 steps takes ~4.5 hours. 50 steps (~4.5 minutes) is
    # enough to validate the full pipeline (train -> checkpoint -> evaluate)
    # end to end without tying up a shared GPU for hours on every dry run.
    # Bump back up for a real training run meant to produce a usable policy.
    NUM_TRAIN_STEPS = 50
    BATCH_SIZE = 32

    train_script, effective_n_action_steps = _train_script(
        effective_dataset_id,
        exp_name,
        num_train_steps=NUM_TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        train_episodes=train_episodes,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        empty_cameras=empty_cameras,
    )

    stages = [
        {
            "name": "train",
            "image": LEROBOT_IMAGE,
            "gpu": 1,
            "command": [
                "/bin/bash",
                "-c",
                train_script,
            ],
        },
        {
            "name": "evaluate",
            "image": LEROBOT_IMAGE,
            "gpu": 1,
            "command": [
                "/bin/bash",
                "-c",
                _evaluate_script(effective_dataset_id, exp_name, eval_episodes=eval_episodes),
            ],
        },
    ]

    params = {
        "model_name": model_name,
        "dataset_repo_id": dataset_repo_id,
        "pretrained_path": PI05_PRETRAINED_PATH,
        "num_train_steps": str(NUM_TRAIN_STEPS),
        "batch_size": str(BATCH_SIZE),
        "normalization_mapping": NORMALIZATION_MAPPING,
        "train_expert_only": "true",
        "total_episodes": str(info["total_episodes"]),
        "num_train_episodes": str(len(train_episodes)),
        "num_eval_episodes": str(len(eval_episodes)),
        "eval_episodes": str(eval_episodes),
    }
    if dataset_subset:
        params["dataset_subset"] = dataset_subset
    if chunk_size is not None:
        params["chunk_size"] = str(chunk_size)
    if effective_n_action_steps is not None:
        params["n_action_steps"] = str(effective_n_action_steps)
    if empty_cameras is not None:
        params["empty_cameras"] = str(empty_cameras)

    return stages, params
