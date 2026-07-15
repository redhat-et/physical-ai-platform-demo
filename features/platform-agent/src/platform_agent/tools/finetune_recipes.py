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

# What a fine-tuning dataset must actually look like for this recipe --
# single source of truth for get_finetune_requirements, so this can't drift
# from what training actually does. NOT a strict feature-key allowlist:
# real-world DROID re-hosts on HF vary in exact naming (e.g.
# 'observation.image.X' vs 'observation.images.X', combined vs separate
# joint/gripper state) -- confirmed by checking multiple real datasets --
# so this is guidance for search/manual review, not an exact-match filter.
DATASET_REQUIREMENTS = {
    "pi05": {
        "dataset_format": (
            "LeRobot v3.0 (lerobot-train's current default -- NOT the older v2.x; "
            "a v2.x dataset needs `python -m lerobot.datasets.v30.convert_dataset_v21_to_v30` "
            "first, or an older lerobot package pin. This is the opposite of what our "
            "earlier openpi-based recipe needed -- the version requirement is tied to "
            "which training mechanism a recipe uses, not a fixed platform-wide fact.)"
        ),
        "robot_type": "franka",
        "expected_exterior_cameras": 2,
        "expected_wrist_cameras": 1,
        "expected_action_dim": 7,
        "state_action": "joint position + gripper position (as separate or combined fields), 7-dim action",
        "search_query_hint": "droid",
        "search_note": (
            "Search for 'droid', not 'pi05' -- 'pi05' as a keyword returns datasets for ANY "
            "embodiment someone used with a pi0.5 model (LIBERO sim, humanoids, custom rigs "
            "with different camera layouts), not specifically DROID-compatible data."
        ),
    }
}


def get_requirements(model_name: str) -> dict:
    """Returns this model's fine-tuning dataset requirements (see
    DATASET_REQUIREMENTS), for get_finetune_requirements to format.
    """
    if model_name not in DATASET_REQUIREMENTS:
        raise ValueError(f"No fine-tuning requirements defined for '{model_name}' -- only 'pi05' is defined so far.")
    return DATASET_REQUIREMENTS[model_name]


def _checkpoint_dir(exp_name: str) -> str:
    """lerobot-train's own convention: {output_dir}/checkpoints/last/pretrained_model
    always points at the most recent checkpoint (a directory containing
    config.json + model.safetensors + pre/post-processor json -- the same
    layout as lerobot/pi05_base itself)."""
    return f"{CHECKPOINT_MOUNT_PATH}/{exp_name}/checkpoints/last/pretrained_model"


def _train_script(dataset_repo_id: str, exp_name: str, num_train_steps: int, batch_size: int) -> str:
    """Training stage script: installs pi0.5 extras (per LeRobot's own docs,
    the base install doesn't include them) and runs lerobot-train directly
    -- a plain CLI, no custom Python config-construction shim needed unlike
    the old openpi-based recipe. Uses the MEAN_STD normalization override
    instead of the QUANTILES preprocessing script, for a simpler first pass
    (see plan's "Open risks" re: fine-tune quality tradeoff).
    NEEDS VERIFICATION during the first dry run: whether huggingface/lerobot-gpu
    already has lerobot[pi] installed, whether lerobot-train handles DROID's
    camera/state layout automatically, and whether train_expert_only fits a
    single 48GB L40S -- none of this is confirmed from docs alone.
    """
    return f"""\
set -e
export HOME=/tmp
export HF_LEROBOT_HOME={DATASET_MOUNT_ROOT}
pip install -q "lerobot[pi]"
lerobot-train \\
    --dataset.repo_id={dataset_repo_id} \\
    --policy.type=pi05 \\
    --policy.pretrained_path={PI05_PRETRAINED_PATH} \\
    --policy.train_expert_only=true \\
    --policy.gradient_checkpointing=true \\
    --policy.dtype=bfloat16 \\
    --policy.device=cuda \\
    --policy.normalization_mapping='{{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}}' \\
    --batch_size={batch_size} \\
    --steps={num_train_steps} \\
    --output_dir={CHECKPOINT_MOUNT_PATH}/{exp_name} \\
    --job_name={exp_name} \\
    --wandb.enable=false
"""


def _evaluate_script(dataset_repo_id: str, exp_name: str) -> str:
    """Offline, self-contained evaluation -- no dependency on
    robotics-playground/Isaac Lab or any external service. Loads the
    fine-tuned checkpoint via LeRobot's own PreTrainedPolicy.from_pretrained
    (no conversion needed -- see module docstring), runs it against a few
    held-out episodes from the staged dataset, and reports action-prediction
    error plus a load/shape smoke test. NEEDS VERIFICATION during the first
    dry run: policy.select_action's exact batch/tensor shape expectations
    are LeRobot's standard PreTrainedPolicy interface, but this exact script
    is our own, not copied from a tested upstream example.
    """
    checkpoint_dir = _checkpoint_dir(exp_name)
    return f"""\
set -e
export HOME=/tmp
export HF_LEROBOT_HOME={DATASET_MOUNT_ROOT}
pip install -q "lerobot[pi]"
cat > /tmp/run_eval.py << 'PYEOF'
import torch
import numpy as np
from lerobot.policies.pi05 import PI05Policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PI05Policy.from_pretrained("{checkpoint_dir}").to(device).eval()

dataset = LeRobotDataset("{dataset_repo_id}")
num_episodes = dataset.num_episodes
held_out = list(range(max(0, num_episodes - 5), num_episodes))
print(f"Evaluating against held-out episodes: {{held_out}}")

errors = []
for ep_idx in held_out:
    ep = dataset[ep_idx]
    ground_truth = np.asarray(ep["action"])
    batch = {{k: (v.unsqueeze(0).to(device) if hasattr(v, "unsqueeze") else v) for k, v in ep.items() if k != "action"}}
    with torch.no_grad():
        predicted = policy.select_action(batch).cpu().numpy().squeeze()
    err = float(np.mean((predicted - ground_truth) ** 2))
    errors.append(err)
    print(f"episode {{ep_idx}}: action MSE = {{err:.4f}}, shape={{predicted.shape}}")

print(f"EVAL_MEAN_ACTION_MSE={{sum(errors) / len(errors):.4f}}")
print("EVAL_SMOKE_TEST=PASS")
PYEOF
cd /tmp && python3 run_eval.py
"""


def get_recipe(model_name: str, dataset_repo_id: str, exp_name: str) -> list[dict]:
    """Returns the ordered stage list for a model's fine-tuning recipe.

    Each stage: name, image, command (list, passed to bash -c), gpu (int
    GPUs requested; 0 means no nodeSelector/GPU resource added).
    """
    if model_name != "pi05":
        raise ValueError(f"No fine-tuning recipe for '{model_name}' -- only 'pi05' is defined so far.")

    return [
        {
            "name": "train",
            "image": LEROBOT_IMAGE,
            "gpu": 1,
            "command": ["/bin/bash", "-c", _train_script(dataset_repo_id, exp_name, num_train_steps=3_000, batch_size=32)],
        },
        {
            "name": "evaluate",
            "image": LEROBOT_IMAGE,
            "gpu": 1,
            "command": ["/bin/bash", "-c", _evaluate_script(dataset_repo_id, exp_name)],
        },
    ]
