import httpx
from langchain_core.tools import tool
from kubernetes import client

from platform_agent.config import settings

DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"
DATASET_CACHE_LABEL = "physical-ai.io/dataset-cache"
DATASET_REPO_LABEL = "physical-ai.io/dataset-repo"


def _get_clients():
    try:
        from kubernetes import config as k8s_config
        k8s_config.load_incluster_config()
    except Exception:
        from kubernetes import config as k8s_config
        k8s_config.load_kube_config()
    return client.CoreV1Api(), client.BatchV1Api()


def dataset_repo_id_from_pvc(pvc) -> str | None:
    """Recovers the HF dataset repo id a staged PVC was pulled from, via the
    DATASET_REPO_LABEL pull_dataset sets (slashes get swapped for "--" since
    K8s label values can't contain "/"). Returns None if the PVC isn't
    labeled as a dataset cache -- e.g. it wasn't created by pull_dataset.
    """
    label = (pvc.metadata.labels or {}).get(DATASET_REPO_LABEL)
    return label.replace("--", "/") if label else None


def _resolve_config_split(
    dataset_repo_id: str, config: str | None, split: str, http: httpx.Client
) -> tuple[str, str, list[str]] | str:
    """Shared config/split resolution against the HF datasets-server API.
    Returns (resolved_config, resolved_split, available_configs), or an
    error string.
    """
    splits_resp = http.get(f"{DATASETS_SERVER_URL}/splits", params={"dataset": dataset_repo_id})
    if splits_resp.status_code != 200:
        return (
            f"Could not fetch split info for '{dataset_repo_id}' "
            f"(HTTP {splits_resp.status_code}) — it may be private, "
            f"gated, or not yet processed by the datasets-server."
        )
    available = splits_resp.json().get("splits", [])
    if not available:
        return f"'{dataset_repo_id}' has no known splits."

    resolved_config = config or available[0]["config"]
    matching = [s for s in available if s["config"] == resolved_config]
    if not matching:
        configs = sorted({s["config"] for s in available})
        return f"Config '{config}' not found for '{dataset_repo_id}'. Available configs: {configs}"

    resolved_split = split if any(s["split"] == split for s in matching) else matching[0]["split"]
    return resolved_config, resolved_split, sorted({s["config"] for s in available})


def _fetch_schema_preview(dataset_repo_id: str, config: str | None, split: str) -> dict | str:
    """Shared helper for get_dataset_info's summary view and validate_dataset's
    generic format. Returns a dict with resolved config/split/features/
    sample_row, or an error string. Uses the HF datasets-server REST API (no
    local `datasets` library dependency) to preview schema/rows without
    downloading anything.
    """
    try:
        with httpx.Client(timeout=15.0) as http:
            resolved = _resolve_config_split(dataset_repo_id, config, split, http)
            if isinstance(resolved, str):
                return resolved
            resolved_config, resolved_split, available_configs = resolved

            rows_resp = http.get(
                f"{DATASETS_SERVER_URL}/first-rows",
                params={"dataset": dataset_repo_id, "config": resolved_config, "split": resolved_split},
            )
            if rows_resp.status_code != 200:
                return (
                    f"Could not fetch a row preview for '{dataset_repo_id}' "
                    f"config='{resolved_config}' split='{resolved_split}' "
                    f"(HTTP {rows_resp.status_code})."
                )
            payload = rows_resp.json()
    except httpx.HTTPError as e:
        return f"Network error reaching the HF datasets-server: {e}"

    features = [
        {"name": f["name"], "dtype": f.get("type", {}).get("dtype", f.get("type", {}).get("_type", "unknown"))}
        for f in payload.get("features", [])
    ]
    sample_row = payload.get("rows", [{}])[0].get("row") if payload.get("rows") else None

    return {
        "config": resolved_config,
        "split": resolved_split,
        "available_configs": available_configs,
        "features": features,
        "sample_row": sample_row,
    }


MAX_DATASET_ROWS_PER_CALL = 20
MAX_DATASET_ROWS_OUTPUT_CHARS = 20_000


def _dataset_info_rows(
    dataset_repo_id: str,
    offset: int,
    length: int,
    config: str | None,
    split: str,
) -> str:
    length = max(1, min(length, MAX_DATASET_ROWS_PER_CALL))
    try:
        with httpx.Client(timeout=15.0) as http:
            resolved = _resolve_config_split(dataset_repo_id, config, split, http)
            if isinstance(resolved, str):
                return resolved
            resolved_config, resolved_split, _ = resolved

            rows_resp = http.get(
                f"{DATASETS_SERVER_URL}/rows",
                params={
                    "dataset": dataset_repo_id,
                    "config": resolved_config,
                    "split": resolved_split,
                    "offset": offset,
                    "length": length,
                },
            )
            if rows_resp.status_code != 200:
                return (
                    f"Could not fetch rows for '{dataset_repo_id}' "
                    f"config='{resolved_config}' split='{resolved_split}' "
                    f"offset={offset} length={length} (HTTP {rows_resp.status_code})."
                )
            payload = rows_resp.json()
    except httpx.HTTPError as e:
        return f"Network error reaching the HF datasets-server: {e}"

    rows = payload.get("rows", [])
    if not rows:
        return (
            f"No rows returned for '{dataset_repo_id}' at offset={offset} "
            f"(config='{resolved_config}', split='{resolved_split}')."
        )

    lines = [
        f"'{dataset_repo_id}' config='{resolved_config}' split='{resolved_split}', "
        f"rows {offset}-{offset + len(rows) - 1}:"
    ]
    for r in rows:
        lines.append(f"  row_idx={r.get('row_idx')}: {r.get('row')}")
    out = "\n".join(lines)
    if len(out) > MAX_DATASET_ROWS_OUTPUT_CHARS:
        out = out[:MAX_DATASET_ROWS_OUTPUT_CHARS] + "\n... (truncated -- request fewer rows)"
    return out


def _dataset_info_summary(dataset_repo_id: str, config: str | None, split: str) -> str:
    from huggingface_hub import HfApi

    try:
        info = HfApi().dataset_info(dataset_repo_id, files_metadata=True)
    except Exception as e:
        return f"Could not fetch dataset info for '{dataset_repo_id}' from Hugging Face: {e}"

    size_bytes = info.used_storage
    if size_bytes is None and info.siblings:
        size_bytes = sum((s.size or (s.lfs.size if s.lfs else 0) or 0) for s in info.siblings)
    size_str = f"{size_bytes / 1e9:.2f} GB" if size_bytes else "unknown"

    license_str = getattr(info.card_data, "license", None) or "unknown"
    task_categories = getattr(info.card_data, "task_categories", None) or []

    preview = _fetch_schema_preview(dataset_repo_id, config, split)
    if isinstance(preview, str):
        schema_section = f"Schema preview unavailable: {preview}"
    else:
        feature_lines = "\n".join(f"  - {f['name']}: {f['dtype']}" for f in preview["features"])
        schema_section = (
            f"Configs available: {preview['available_configs']}\n"
            f"Previewing config='{preview['config']}' split='{preview['split']}'\n"
            f"Columns:\n{feature_lines}\n"
            f"Sample row: {preview['sample_row']}"
        )

    return (
        f"'{dataset_repo_id}':\n"
        f"Size: {size_str}\n"
        f"License: {license_str}\n"
        f"Task categories: {task_categories}\n"
        f"Downloads: {info.downloads or 0}, Likes: {info.likes or 0}\n"
        f"Gated: {info.gated}\n"
        f"Created: {info.created_at}, Last modified: {info.last_modified}\n"
        f"Tags: {info.tags or []}\n"
        f"{schema_section}"
    )


MAX_DATASET_FILE_BYTES = 2_000_000
MAX_DATASET_FILE_CHARS = 20_000


def _dataset_info_file(dataset_repo_id: str, filename: str) -> str:
    from huggingface_hub import HfApi, hf_hub_download

    try:
        info = HfApi().dataset_info(dataset_repo_id, files_metadata=True)
    except Exception as e:
        return f"Could not look up '{dataset_repo_id}': {e}"

    sibling = next((s for s in (info.siblings or []) if s.rfilename == filename), None)
    if sibling is None:
        available = sorted(s.rfilename for s in (info.siblings or []))[:30]
        return f"'{filename}' not found in '{dataset_repo_id}'. Some files present: {available}"

    size = sibling.size or (sibling.lfs.size if sibling.lfs else 0)
    if size and size > MAX_DATASET_FILE_BYTES:
        return (
            f"'{filename}' is {size / 1e6:.1f}MB — too large to fetch as text "
            f"(limit {MAX_DATASET_FILE_BYTES / 1e6:.0f}MB). This tool is for "
            f"docs/metadata, not data files."
        )

    try:
        path = hf_hub_download(repo_id=dataset_repo_id, repo_type="dataset", filename=filename)
    except Exception as e:
        return f"Could not download '{filename}' from '{dataset_repo_id}': {e}"

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return f"'{filename}' isn't text — this tool only reads text/metadata files."

    if len(content) > MAX_DATASET_FILE_CHARS:
        content = content[:MAX_DATASET_FILE_CHARS] + f"\n... (truncated, {len(content)} total characters)"
    return f"'{dataset_repo_id}/{filename}':\n{content}"


@tool
def get_dataset_info(
    dataset_repo_id: str,
    view: str = "summary",
    filename: str | None = None,
    offset: int = 0,
    length: int = 5,
    config: str | None = None,
    split: str = "train",
) -> str:
    """Inspect a Hugging Face dataset without downloading it. Three views:
    'summary' (default) for size, license, gated status, tags, dates,
    schema, and one sample row; 'rows' for a range of raw rows via
    offset/length, to compare fields across several rows instead of just
    one; 'file' for one file's raw text (e.g. 'README.md' or
    'meta/stats.json'). See the datasets skill for which view answers
    which compatibility question.

    Always call view='summary' and relay its size/license/Gated status to
    the user before ever calling pull_dataset -- pulling consumes real
    shared-cluster storage, and a gated dataset needs manual approval
    first.

    Args:
        dataset_repo_id: Hugging Face dataset repo id, e.g. 'GEAR-Dreams/DreamZero-DROID'.
        view: 'summary' (default), 'rows', or 'file'.
        filename: Required when view='file' -- exact path within the repo,
            e.g. 'README.md' or 'meta/stats.json'.
        offset: Row index to start from, only used when view='rows' (default 0).
        length: Number of rows to fetch, only used when view='rows' (default 5, capped at 20).
        config: Dataset config/subset name, used by 'summary'/'rows'. Defaults to the first available one.
        split: Dataset split, used by 'summary'/'rows' (default 'train').
    """
    if view == "summary":
        return _dataset_info_summary(dataset_repo_id, config, split)
    if view == "rows":
        return _dataset_info_rows(dataset_repo_id, offset, length, config, split)
    if view == "file":
        if not filename:
            return "view='file' requires a filename, e.g. 'README.md' or 'meta/stats.json'."
        return _dataset_info_file(dataset_repo_id, filename)
    return f"Unknown view '{view}'. Valid views: 'summary', 'rows', 'file'."


def _validate_generic_schema(
    dataset_repo_id: str,
    expected_feature_keys: list[str] | None,
    config: str | None,
    split: str,
) -> str:
    preview = _fetch_schema_preview(dataset_repo_id, config, split)
    if isinstance(preview, str):
        return preview

    actual_columns = {f["name"] for f in preview["features"]}
    feature_lines = "\n".join(f"  - {f['name']}: {f['dtype']}" for f in preview["features"])
    header = (
        f"'{dataset_repo_id}' config='{preview['config']}' split='{preview['split']}'\n"
        f"Columns:\n{feature_lines}\n"
        f"Sample row: {preview['sample_row']}"
    )

    if not expected_feature_keys:
        return header

    missing = [c for c in expected_feature_keys if c not in actual_columns]
    extra = sorted(actual_columns - set(expected_feature_keys))
    if missing:
        return (
            f"{header}\n\n"
            f"INCOMPATIBLE: missing expected column(s) {missing}. "
            f"Present but unexpected: {extra or 'none'}."
        )
    return f"{header}\n\nCOMPATIBLE: all expected columns {expected_feature_keys} are present."


# Tied to the pi0.5 recipe's specific training mechanism (tools/finetune_recipes.py),
# NOT a fixed platform-wide fact: that recipe trains via LeRobot's own native
# `lerobot-train` CLI, whose current releases default to the newer LeRobotDataset
# v3.0 format (v2.x needs an explicit conversion script or an older lerobot pin).
# An earlier version of this recipe trained via openpi's JAX scripts instead, which
# are pinned to the OLDER v2.x format -- the opposite requirement. If a future
# recipe for a different model uses a v2.x-only training mechanism again, it needs
# its own version check rather than sharing this one.
LEROBOT_COMPATIBLE_VERSION_PREFIX = "v3"


def split_dataset_repo_id(dataset_repo_id: str) -> tuple[str, str | None]:
    """A real Hugging Face repo id is always exactly two slash-separated
    segments (org/name) -- anything past that in dataset_repo_id is a
    subfolder within the repo, not part of the id, and needs splitting back
    off before any call that actually hits the Hub API (e.g.
    hf_hub_download). Exists because some repos (e.g. nvidia's
    PhysicalAI-Robotics-Manipulation-SingleArm) bundle several independent
    LeRobot datasets as subfolders of one repo instead of one dataset per
    repo -- confirmed live via the Hub API's file listing: each subfolder
    has its own meta/info.json, not the repo root.
    """
    parts = dataset_repo_id.split("/", 2)
    if len(parts) <= 2:
        return dataset_repo_id, None
    return "/".join(parts[:2]), parts[2]


def _fetch_lerobot_info(dataset_repo_id: str) -> dict | str:
    """Shared helper for validate_dataset's lerobot format.
    Returns the parsed meta/info.json, or an error string.
    """
    from huggingface_hub import hf_hub_download
    import json

    real_repo_id, subset = split_dataset_repo_id(dataset_repo_id)
    filename = f"{subset}/meta/info.json" if subset else "meta/info.json"

    try:
        info_path = hf_hub_download(repo_id=real_repo_id, repo_type="dataset", filename=filename)
    except Exception as e:
        return f"Could not fetch {filename} for '{dataset_repo_id}': {e}. Is this actually a LeRobot-format dataset?"

    with open(info_path) as f:
        return json.load(f)


def _count_camera_features(features: dict, substring: str) -> int:
    """Counts image/video-typed features whose key contains `substring`
    (case-insensitive) -- used for expected_exterior_cameras/
    expected_wrist_cameras, which check camera COUNT rather than exact key
    spelling. Exact key names vary too much between real-world DROID
    re-hosts (confirmed: 'observation.image.X' vs 'observation.images.X',
    'wrist_image_left' vs 'wrist_left' vs 'wrist') for a string-match check
    to be reliable -- and asking a model to supply an exact key it doesn't
    actually know tends to produce fabricated-but-plausible-looking keys
    (confirmed repeatedly in practice) rather than an honest "I don't know".
    """
    substring = substring.lower()
    return sum(
        1
        for name, spec in features.items()
        if substring in name.lower() and spec.get("dtype") in ("image", "video")
    )


def _validate_lerobot_dataset(
    dataset_repo_id: str,
    expected_action_dim: int | None,
    expected_exterior_cameras: int | None,
    expected_wrist_cameras: int | None,
    expected_feature_keys: list[str] | None,
) -> str:
    info = _fetch_lerobot_info(dataset_repo_id)
    if isinstance(info, str):
        return info

    codebase_version = info.get("codebase_version", "unknown")
    version_note = ""
    if not codebase_version.startswith(LEROBOT_COMPATIBLE_VERSION_PREFIX):
        version_note = (
            f"INCOMPATIBLE: codebase_version is '{codebase_version}', but this platform's current "
            f"fine-tuning recipe requires LeRobot {LEROBOT_COMPATIBLE_VERSION_PREFIX}.x -- LeRobot "
            f"dataset format versions are NOT backward/forward compatible with each other. Look for "
            f"a {LEROBOT_COMPATIBLE_VERSION_PREFIX}.x version of this dataset (or convert it), or a different one."
        )

    features = info.get("features", {})
    feature_lines = "\n".join(
        f"  - {name}: dtype={spec.get('dtype')}, shape={spec.get('shape')}" for name, spec in features.items()
    )

    result = (
        f"'{dataset_repo_id}' LeRobot metadata:\n"
        f"codebase_version: {codebase_version}\n"
        f"robot_type: {info.get('robot_type', 'unknown')}\n"
        f"fps: {info.get('fps', 'unknown')}, total_episodes: {info.get('total_episodes', 'unknown')}\n"
        f"Features:\n{feature_lines}"
    )
    if version_note:
        result += f"\n\n{version_note}"

    checks = []
    if expected_action_dim is not None:
        action_shape = features.get("action", {}).get("shape")
        actual_dim = action_shape[0] if isinstance(action_shape, list) and action_shape else None
        if actual_dim == expected_action_dim:
            checks.append(f"COMPATIBLE: raw action dim {actual_dim} matches expected {expected_action_dim}.")
        else:
            checks.append(f"INCOMPATIBLE: raw action dim is {actual_dim}, expected {expected_action_dim}.")

    if expected_exterior_cameras is not None:
        actual = _count_camera_features(features, "exterior")
        if actual == expected_exterior_cameras:
            checks.append(f"COMPATIBLE: {actual} exterior camera(s) found, matches expected {expected_exterior_cameras}.")
        else:
            checks.append(f"INCOMPATIBLE: {actual} exterior camera(s) found, expected {expected_exterior_cameras}.")

    if expected_wrist_cameras is not None:
        actual = _count_camera_features(features, "wrist")
        if actual == expected_wrist_cameras:
            checks.append(f"COMPATIBLE: {actual} wrist camera(s) found, matches expected {expected_wrist_cameras}.")
        else:
            checks.append(f"INCOMPATIBLE: {actual} wrist camera(s) found, expected {expected_wrist_cameras}.")

    if expected_feature_keys:
        missing = [k for k in expected_feature_keys if k not in features]
        if missing:
            checks.append(f"INCOMPATIBLE: missing expected feature key(s) {missing}.")
        else:
            checks.append(f"COMPATIBLE: all expected feature keys {expected_feature_keys} are present.")

    if checks:
        result += "\n\n" + "\n".join(checks)

    return result


@tool
def validate_dataset(
    dataset_repo_id: str,
    dataset_format: str = "lerobot",
    expected_feature_keys: list[str] | None = None,
    expected_action_dim: int | None = None,
    expected_exterior_cameras: int | None = None,
    expected_wrist_cameras: int | None = None,
    config: str | None = None,
    split: str = "train",
) -> str:
    """Check a Hugging Face dataset's compatibility with a target model's
    expected input format, without downloading it. Two formats:
    'lerobot' (default) reads the dataset's own meta/info.json for
    robot-policy compatibility -- action dimensionality, camera counts,
    feature keys, fps, episode count, and LeRobot codebase_version; use
    this for robot-policy models. 'generic' checks flat column names via
    the HF datasets-server instead -- use this for non-LeRobot data (e.g.
    a video world model expecting 'video'/'caption' columns). See the
    datasets skill for detailed guidance on which fields to pass for a
    given target model, and why a dimension-count match alone isn't
    sufficient proof of compatibility.

    NEVER invent expected_action_dim/expected_exterior_cameras/
    expected_wrist_cameras/expected_feature_keys from memory or general
    knowledge of the model -- these are caller-supplied inputs this tool
    trusts verbatim and echoes back in its verdict, so a wrong value you
    supplied produces a confident-looking but false result. Omit a field
    entirely and read the schema/Features list yourself if you don't have
    a real value for it.

    Args:
        dataset_repo_id: Hugging Face dataset repo id.
        dataset_format: 'lerobot' (default) or 'generic'.
        expected_feature_keys: For 'lerobot', exact LeRobot feature keys.
            For 'generic', the expected column names. Only pass values
            documented for the target model or given directly by the user.
        expected_action_dim: 'lerobot' only -- raw action feature
            dimensionality to check for.
        expected_exterior_cameras: 'lerobot' only -- number of
            exterior-camera features expected (counts image/video features
            with 'exterior' in the key name, regardless of exact spelling).
        expected_wrist_cameras: 'lerobot' only -- number of wrist-camera
            features expected (same counting approach, for 'wrist').
        config: 'generic' only -- dataset config/subset name. Defaults to
            the first available one.
        split: 'generic' only -- dataset split to check (default 'train').
    """
    if dataset_format == "lerobot":
        return _validate_lerobot_dataset(
            dataset_repo_id,
            expected_action_dim,
            expected_exterior_cameras,
            expected_wrist_cameras,
            expected_feature_keys,
        )
    if dataset_format == "generic":
        return _validate_generic_schema(dataset_repo_id, expected_feature_keys, config, split)
    return f"Unknown dataset_format '{dataset_format}'. Valid formats: 'lerobot', 'generic'."


@tool
def pull_dataset(dataset_repo_id: str, dataset_name: str, pvc_size_gb: int = 50) -> str:
    """Download a Hugging Face dataset onto the cluster so a fine-tuning job
    can read it. This consumes real shared-cluster storage — only call this
    after calling get_dataset_info and showing the user its size and license,
    and after the user has explicitly said to proceed. Never call this
    speculatively or as the first response to "find me a dataset for X".

    Downloads the entire dataset repo (config/subset selection happens at
    fine-tuning time via submit_finetune_run's dataset_subset, not download
    time) -- including for a repo that bundles several independent LeRobot
    datasets as subfolders rather than one dataset per repo (e.g. nvidia's
    PhysicalAI-Robotics-Manipulation-SingleArm: panda-stack-platforms,
    panda-open-drawer, ... each in their own subfolder). Pulling once here
    and picking a subset per fine-tuning run means one PVC serves every
    subset in the repo, instead of needing a separate PVC (and a separate,
    redundant download) per subset. Creates a PVC and a Kubernetes Job in
    the datasets namespace that runs huggingface_hub.snapshot_download.
    Check progress with get_dataset_job_status afterward — this tool returns
    as soon as the Job is created, not once the download finishes.

    Falls back to a plain `git clone` + `git lfs pull` of the same repo if
    snapshot_download keeps hitting HTTP 429 after a few retries (confirmed
    live: a repo subset with ~53k files exhausted the 1000 req/5min
    authenticated rate limit after only ~6.6k files, since snapshot_download
    makes a HEAD+GET pair per file; git-lfs instead fetches object URLs via
    the LFS batch API in ~100-object batches, cutting total requests by
    ~2 orders of magnitude -- the same fallback finished in 18 minutes with
    zero 429s where snapshot_download's steady-state throttled rate would
    have taken ~8 hours). The fallback clones into a scratch dir and copies
    the checked-out tree into local_dir, so the end result is
    indistinguishable from a snapshot_download-produced PVC either way.

    Args:
        dataset_repo_id: Hugging Face dataset repo id to download, e.g.
            'GEAR-Dreams/DreamZero-DROID'. Always the real two-segment
            org/name id -- never include a subfolder here, even for a
            multi-subset repo; pick the subset later via
            submit_finetune_run's dataset_subset instead.
        dataset_name: Short name for this staged dataset, lowercase
            alphanumeric and hyphens (used as the K8s resource name prefix).
        pvc_size_gb: Size of the PersistentVolumeClaim in GB. Should
            comfortably exceed the size reported by get_dataset_info for
            the *whole* repo (all subsets combined, for a multi-subset one).
    """
    # Deferred import: finetune_recipes imports _fetch_lerobot_info from this
    # module, so importing it back at module load time would be circular.
    from platform_agent.tools.finetune_recipes import LEROBOT_IMAGE

    core_api, batch_api = _get_clients()
    pvc_name = f"dataset-{dataset_name}-pvc"
    job_name = f"download-{dataset_name}-dataset"
    labels = {DATASET_CACHE_LABEL: "true", DATASET_REPO_LABEL: dataset_repo_id.replace("/", "--")}

    try:
        core_api.create_namespaced_persistent_volume_claim(
            namespace=settings.datasets_namespace,
            body={
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {"name": pvc_name, "labels": labels},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {"requests": {"storage": f"{pvc_size_gb}Gi"}},
                    "storageClassName": "gp3-csi",
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status != 409:
            return f"Failed to create PVC '{pvc_name}': {e.reason}"

    # Not an f-string / .format() call: the git fallback below is full of
    # literal ${VAR} bash syntax that would otherwise collide with brace
    # interpolation. dataset_repo_id is spliced in via .replace() instead.
    download_script = """\
set -uo pipefail

python3 - <<'PYEOF'
import os, re, sys, time
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

token = os.getenv("HF_TOKEN")
repo_id = "__DATASET_REPO_ID__"

# snapshot_download can swallow a rate-limit error itself: if it can't reach
# the repo but local_dir already has *something* in it (e.g. a partial file
# from a prior attempt sharing this PVC), it logs a warning and returns the
# existing directory as-is instead of raising -- so a bare try/except around
# it can't tell "downloaded everything" from "downloaded nothing, gave up
# quietly". Compare against the repo's real file count instead of trusting
# a clean return.
try:
    expected_files = len(HfApi().list_repo_files(repo_id, repo_type="dataset", token=token))
except Exception as e:
    print(f"could not list repo files up front ({e}) -- skipping completeness check", flush=True)
    expected_files = None


def local_file_count():
    total = 0
    for root, _dirs, files in os.walk("/mnt/dataset"):
        if root == "/mnt/dataset/.cache" or root.startswith("/mnt/dataset/.cache/"):
            continue
        total += len(files)
    return total


max_attempts = 3
for attempt in range(1, max_attempts + 1):
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir="/mnt/dataset",
            token=token,
            max_workers=4,
        )
        actual_files = local_file_count()
        if expected_files is not None and actual_files < expected_files * 0.95:
            print(
                f"snapshot_download returned without error but only "
                f"{actual_files}/{expected_files} files are present -- treating as "
                f"incomplete (likely its own silent existing-local-dir fallback, not "
                f"a real success)",
                flush=True,
            )
        else:
            print(f"SNAPSHOT_DOWNLOAD_OK ({actual_files} files)", flush=True)
            sys.exit(0)
    except HfHubHTTPError as e:
        wait = 90
        m = re.search(r"Retry after (\\d+) seconds", str(e))
        if m:
            wait = int(m.group(1)) + 10
        print(f"snapshot_download attempt {attempt}/{max_attempts} failed: {e}", flush=True)
        if attempt < max_attempts:
            print(f"sleeping {wait}s before retry", flush=True)
            time.sleep(wait)
            continue
    if attempt < max_attempts:
        print(f"sleeping 90s before retry", flush=True)
        time.sleep(90)
print("SNAPSHOT_DOWNLOAD_EXHAUSTED -- falling back to git+lfs", flush=True)
sys.exit(1)
PYEOF
snapshot_rc=$?

if [ "$snapshot_rc" -ne 0 ]; then
    # snapshot_download does a HEAD+GET per file, so it burns through HF's
    # per-token rate limit fast on repos with tens of thousands of files.
    # git-lfs instead resolves object URLs via the LFS batch API in ~100-
    # object batches -- ~2 orders of magnitude fewer requests for the same
    # content (confirmed live: 18min/0 429s vs an ~8h throttled crawl for a
    # ~53k-file subset). git itself is preinstalled on this image; git-lfs
    # isn't, so fetch its static binary into a user-writable dir (this
    # image's restricted-SCC UID can't apt-get install into /usr).
    set -e
    mkdir -p /tmp/bin
    export PATH="/tmp/bin:$PATH"
    curl -sL -m 60 -o /tmp/git-lfs.tar.gz \\
        https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
    tar -xzf /tmp/git-lfs.tar.gz -C /tmp
    cp /tmp/git-lfs-*/git-lfs /tmp/bin/git-lfs
    chmod +x /tmp/bin/git-lfs
    git lfs install --skip-smudge
    git config --global --add safe.directory '*'
    rm -rf /tmp/repo
    # Auth via an explicit header (not a token-in-URL) so it never shows up
    # in `git remote -v` or error output. -c only applies to this one clone;
    # persist the same header into the new repo's own config afterward so
    # `git lfs pull` (a separate process) picks it up too.
    GIT_LFS_SKIP_SMUDGE=1 git -c http.extraHeader="Authorization: Bearer ${HF_TOKEN}" \\
        clone --depth 1 "https://huggingface.co/datasets/__DATASET_REPO_ID__" /tmp/repo
    git -C /tmp/repo config http.extraHeader "Authorization: Bearer ${HF_TOKEN}"
    git -C /tmp/repo lfs pull
    find /tmp/repo -mindepth 1 -maxdepth 1 ! -name .git -exec cp -a {} /mnt/dataset/ \\;
    rm -rf /tmp/repo /mnt/dataset/.cache
    echo "DOWNLOAD_COMPLETE (git+lfs fallback)"
else
    echo "DOWNLOAD_COMPLETE (snapshot_download)"
fi
""".replace("__DATASET_REPO_ID__", dataset_repo_id)

    try:
        batch_api.create_namespaced_job(
            namespace=settings.datasets_namespace,
            body={
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": job_name, "labels": labels},
                "spec": {
                    "backoffLimit": 3,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "downloader",
                                    # Also used by convert_dataset_to_v3 -- already has git +
                                    # python3 + huggingface_hub preinstalled, so the fallback
                                    # below only needs to fetch git-lfs itself, and the primary
                                    # snapshot_download path needs no pip install step at all.
                                    "image": LEROBOT_IMAGE,
                                    "command": ["/bin/bash", "-c", download_script],
                                    "env": [
                                        {
                                            "name": "HF_TOKEN",
                                            "valueFrom": {
                                                "secretKeyRef": {"name": "huggingface-token", "key": "HF_TOKEN"}
                                            },
                                        },
                                        {"name": "HF_HOME", "value": "/tmp/hf_home"},
                                        # OpenShift's restricted SCC runs this container as an
                                        # arbitrary non-root UID with no /etc/passwd entry, so
                                        # $HOME resolves to something unwritable (e.g. "/") --
                                        # both huggingface_hub and git need a writable HOME for
                                        # their caches/config.
                                        {"name": "HOME", "value": "/tmp"},
                                        # Confirmed live: a repo with 100k+ small files (one per
                                        # episode/camera) hit HTTP 429 from HF's xet-read-token
                                        # endpoint within ~2 minutes at snapshot_download's default
                                        # concurrency, twice in a row (both attempts died at
                                        # nearly the same file count) -- disabling xet falls back
                                        # to plain HTTP/LFS downloads, which don't hit that
                                        # specific rate limit.
                                        {"name": "HF_HUB_DISABLE_XET", "value": "1"},
                                    ],
                                    "volumeMounts": [{"name": "dataset-storage", "mountPath": "/mnt/dataset"}],
                                    "resources": {
                                        "requests": {"cpu": "2", "memory": "4Gi"},
                                        "limits": {"cpu": "4", "memory": "8Gi"},
                                    },
                                }
                            ],
                            "volumes": [{"name": "dataset-storage", "persistentVolumeClaim": {"claimName": pvc_name}}],
                        }
                    },
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status == 409:
            return (
                f"Job '{job_name}' already exists — a pull for '{dataset_name}' "
                f"is already in progress or complete. Check get_dataset_job_status."
            )
        return f"Failed to create download Job '{job_name}': {e.reason}"

    return (
        f"Started downloading '{dataset_repo_id}' into PVC '{pvc_name}' via "
        f"Job '{job_name}' in namespace '{settings.datasets_namespace}'. "
        f"Call get_dataset_job_status('{dataset_name}') to check progress."
    )


@tool
def get_dataset_job_status(dataset_name: str) -> str:
    """Check the status of a dataset download started by pull_dataset:
    whether the Job succeeded/failed/is still running, and the backing PVC's
    bound state.

    Args:
        dataset_name: The dataset_name passed to pull_dataset.
    """
    core_api, batch_api = _get_clients()
    pvc_name = f"dataset-{dataset_name}-pvc"
    job_name = f"download-{dataset_name}-dataset"

    try:
        # read_namespaced_job (not _status) -- the /status subresource needs
        # separate RBAC from the base "jobs" resource we're actually granted;
        # the plain read already returns the full object including .status.
        job = batch_api.read_namespaced_job(name=job_name, namespace=settings.datasets_namespace)
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"No download Job '{job_name}' found — has pull_dataset been called for '{dataset_name}'?"
        return f"Could not read Job '{job_name}': {e.reason}"

    status = job.status
    if status.succeeded:
        state = "succeeded"
    elif status.failed:
        state = "failed"
    elif status.active:
        state = "running"
    else:
        state = "pending"

    try:
        pvc = core_api.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=settings.datasets_namespace)
        pvc_phase = pvc.status.phase
    except client.exceptions.ApiException:
        pvc_phase = "unknown"

    result = f"Dataset '{dataset_name}': download Job is {state}, PVC '{pvc_name}' is {pvc_phase}."

    if state == "failed":
        pods = core_api.list_namespaced_pod(
            namespace=settings.datasets_namespace,
            label_selector=f"job-name={job_name}",
        )
        if pods.items:
            try:
                logs = core_api.read_namespaced_pod_log(
                    name=pods.items[0].metadata.name,
                    namespace=settings.datasets_namespace,
                    tail_lines=30,
                )
                result += f"\nLast 30 log lines:\n{logs}"
            except client.exceptions.ApiException:
                pass
    elif state == "succeeded":
        result += " Ready to reference by PVC name in a fine-tuning pipeline."

    return result


def _conversion_job_name(dataset_pvc_name: str) -> str:
    return f"convert-{dataset_pvc_name}-v3"


@tool
def convert_dataset_to_v3(dataset_pvc_name: str) -> str:
    """Convert an already-staged LeRobot dataset from v2.1 to v3.0 format, in
    place on its existing PVC. Only call this after a fine-tuning run's train
    stage has actually failed with a dataset-format error (e.g.
    BackwardCompatibilityError, or log lines mentioning "v2.1"/"v3.0") --
    check get_finetune_run_status's stage logs first. lerobot-train and this
    platform's fine-tuning recipes only support v3.0 datasets.

    Runs `python -m lerobot.scripts.convert_dataset_v21_to_v30` (confirmed
    live against the huggingface/lerobot-gpu:latest training image -- the
    older `lerobot.datasets.v30.convert_dataset_v21_to_v30` path no longer
    exists there) as a Kubernetes Job against the dataset's existing PVC, no
    GPU needed. Check progress with get_dataset_conversion_status afterward --
    this tool returns as soon as the Job is created.

    Args:
        dataset_pvc_name: The PVC name of an already-pull_dataset-staged
            dataset (e.g. 'dataset-my-droid-set-pvc').
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

    dataset_repo_id = dataset_repo_id_from_pvc(pvc)
    if not dataset_repo_id:
        return f"PVC '{dataset_pvc_name}' isn't labeled as a dataset cache — was it created by pull_dataset?"

    # Deferred import: finetune_recipes imports _fetch_lerobot_info from this
    # module, so importing it back at module load time would be circular.
    from platform_agent.tools.finetune_recipes import LEROBOT_IMAGE

    job_name = _conversion_job_name(dataset_pvc_name)
    convert_script = """\
set -e
export HOME=/tmp
# convert_dataset_v21_to_v30 converts --root in place, but uses a `<root>_v30`
# sibling directory as scratch space while doing so (confirmed live: pointing
# --root directly at the PVC mount gets a PermissionError trying to create
# that sibling under /mnt itself, which OpenShift's restricted SCC --
# arbitrary non-root UID -- can't write to). So nest the original content one
# level inside the PVC mount first, giving the scratch dir room to exist
# inside the still-writable PVC too. The script also leaves its own internal
# v2.1 backup as a SIBLING of --root, at <root>_old (confirmed live -- not
# nested inside --root, despite it looking that way on an earlier inspection;
# that turned out to be a retry re-processing an already-converted directory,
# whose own first move-into-place step swept the previous attempt's stray
# sibling backup inside too).
#
# Restore EVERYTHING from the scratch dir back to the PVC root on ANY exit --
# not just an allowlist of known LeRobot dirs (data/meta/videos), which used
# to silently rm -rf any other top-level file (README.md, .gitattributes,
# ...) along with the scratch dir on every successful conversion. This same
# trap also fires when the conversion itself FAILS (set -e triggers the EXIT
# trap on any nonzero exit): it falls back to lerobot's own <root>_old
# pre-conversion backup if the scratch dir ends up empty, so the PVC is left
# with a working dataset at the root path every other tool expects, instead
# of stuck nested and unusable with no repair tool available.
trap '
    set +e
    restore_from=/mnt/dataset/_v21_orig
    if [ ! -d "$restore_from" ] || [ -z "$(ls -A "$restore_from" 2>/dev/null)" ]; then
        restore_from=/mnt/dataset/_v21_orig_old
    fi
    if [ -d "$restore_from" ]; then
        find "$restore_from" -mindepth 1 -maxdepth 1 -exec mv {{}} /mnt/dataset/ \\;
    fi
    rm -rf /mnt/dataset/_v21_orig /mnt/dataset/_v21_orig_old
' EXIT
mkdir -p /mnt/dataset/_v21_orig
find /mnt/dataset -mindepth 1 -maxdepth 1 ! -name _v21_orig -exec mv {{}} /mnt/dataset/_v21_orig/ \\;
python -m lerobot.scripts.convert_dataset_v21_to_v30 \\
    --repo-id={dataset_repo_id} \\
    --root=/mnt/dataset/_v21_orig \\
    --push-to-hub=false
""".format(dataset_repo_id=dataset_repo_id)

    try:
        batch_api.create_namespaced_job(
            namespace=settings.datasets_namespace,
            body={
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": job_name, "labels": {DATASET_REPO_LABEL: dataset_repo_id.replace("/", "--")}},
                "spec": {
                    "backoffLimit": 1,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "convert",
                                    "image": LEROBOT_IMAGE,
                                    "command": ["/bin/bash", "-c", convert_script],
                                    "env": [
                                        {
                                            "name": "HF_TOKEN",
                                            "valueFrom": {
                                                "secretKeyRef": {"name": "huggingface-token", "key": "HF_TOKEN"}
                                            },
                                        },
                                        # The image bakes in a non-writable default HF_HOME
                                        # (confirmed live: the conversion script's episodes-metadata
                                        # step failed with PermissionError writing to
                                        # /home/user_lerobot/.cache -- an inline `export HOME=/tmp`
                                        # in the script doesn't override it, since the `datasets`
                                        # library resolves its cache dir from HF_HOME directly, not
                                        # from $HOME). Same env var finetune_pipeline.py's
                                        # submit_pipeline_run already sets for this same image.
                                        {"name": "HF_HOME", "value": "/tmp/hf_home"},
                                        {"name": "HOME", "value": "/tmp"},
                                    ],
                                    "volumeMounts": [{"name": "dataset-storage", "mountPath": "/mnt/dataset"}],
                                    "resources": {
                                        "requests": {"cpu": "2", "memory": "4Gi"},
                                        "limits": {"cpu": "4", "memory": "8Gi"},
                                    },
                                }
                            ],
                            "volumes": [
                                {"name": "dataset-storage", "persistentVolumeClaim": {"claimName": dataset_pvc_name}}
                            ],
                        }
                    },
                },
            },
        )
    except client.exceptions.ApiException as e:
        if e.status == 409:
            return (
                f"Job '{job_name}' already exists — a conversion for '{dataset_pvc_name}' "
                f"is already in progress or complete. Check get_dataset_conversion_status."
            )
        return f"Failed to create conversion Job '{job_name}': {e.reason}"

    return (
        f"Started converting '{dataset_repo_id}' (PVC '{dataset_pvc_name}') to v3.0 via "
        f"Job '{job_name}'. Call get_dataset_conversion_status('{dataset_pvc_name}') to check progress."
    )


@tool
def get_dataset_conversion_status(dataset_pvc_name: str) -> str:
    """Check the status of a v2.1-to-v3.0 dataset conversion started by
    convert_dataset_to_v3: whether the Job succeeded/failed/is still running.

    Args:
        dataset_pvc_name: The dataset_pvc_name passed to convert_dataset_to_v3.
    """
    core_api, batch_api = _get_clients()
    job_name = _conversion_job_name(dataset_pvc_name)

    try:
        job = batch_api.read_namespaced_job(name=job_name, namespace=settings.datasets_namespace)
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return f"No conversion Job '{job_name}' found — has convert_dataset_to_v3 been called for '{dataset_pvc_name}'?"
        return f"Could not read Job '{job_name}': {e.reason}"

    status = job.status
    if status.succeeded:
        state = "succeeded"
    elif status.failed:
        state = "failed"
    elif status.active:
        state = "running"
    else:
        state = "pending"

    result = f"Dataset conversion for '{dataset_pvc_name}': Job is {state}."

    if state == "failed":
        pods = core_api.list_namespaced_pod(
            namespace=settings.datasets_namespace,
            label_selector=f"job-name={job_name}",
        )
        if pods.items:
            try:
                logs = core_api.read_namespaced_pod_log(
                    name=pods.items[0].metadata.name,
                    namespace=settings.datasets_namespace,
                    tail_lines=30,
                )
                result += f"\nLast 30 log lines:\n{logs}"
            except client.exceptions.ApiException:
                pass
    elif state == "succeeded":
        result += " Dataset is now v3.0 -- retry submit_finetune_run with the same dataset_pvc_name."

    return result


@tool
def list_staged_datasets() -> str:
    """List datasets already staged on the cluster (downloaded via
    pull_dataset), so you don't redundantly re-pull one that's already
    available. Shows each staged dataset's PVC name, source HF repo, size,
    and bound status.
    """
    core_api, _ = _get_clients()

    pvcs = core_api.list_namespaced_persistent_volume_claim(
        namespace=settings.datasets_namespace,
        label_selector=f"{DATASET_CACHE_LABEL}=true",
    )

    if not pvcs.items:
        return "No datasets currently staged."

    lines = []
    for pvc in pvcs.items:
        repo = pvc.metadata.labels.get(DATASET_REPO_LABEL, "unknown").replace("--", "/")
        size = pvc.spec.resources.requests.get("storage", "?")
        phase = pvc.status.phase
        lines.append(f"- {pvc.metadata.name}: source={repo}, size={size}, status={phase}")

    return "Staged datasets:\n" + "\n".join(lines)
