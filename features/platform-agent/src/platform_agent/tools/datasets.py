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


def _fetch_schema_preview(dataset_repo_id: str, config: str | None, split: str) -> dict | str:
    """Shared helper for get_dataset_info/validate_dataset_schema. Returns a
    dict with resolved config/split/features/sample_row, or an error string.
    Uses the HF datasets-server REST API (no local `datasets` library
    dependency) to preview schema/rows without downloading anything.
    """
    try:
        with httpx.Client(timeout=15.0) as http:
            splits_resp = http.get(
                f"{DATASETS_SERVER_URL}/splits", params={"dataset": dataset_repo_id}
            )
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
        "available_configs": sorted({s["config"] for s in available}),
        "features": features,
        "sample_row": sample_row,
    }


@tool
def search_datasets(query: str, task: str | None = None, limit: int = 10) -> str:
    """Search Hugging Face Hub for datasets by keyword, optionally filtered
    by task category. No download happens — this is a pure metadata search.

    Args:
        query: Free-text search query, e.g. 'robot manipulation trajectories'.
        task: Optional HF task category to filter by, e.g. 'robotics',
            'video-generation', 'image-to-text'.
        limit: Max number of results to return (default 10).
    """
    from huggingface_hub import HfApi

    try:
        results = list(
            HfApi().list_datasets(
                search=query,
                task_categories=[task] if task else None,
                sort="downloads",
                limit=limit,
            )
        )
    except Exception as e:
        return f"Dataset search failed: {e}"

    if not results:
        return f"No datasets found matching '{query}'" + (f" (task={task})" if task else "") + "."

    lines = []
    for d in results:
        tags = ", ".join((d.tags or [])[:5])
        lines.append(
            f"- {d.id}: downloads={d.downloads or 0}, likes={d.likes or 0}, "
            f"tags=[{tags}]"
        )
    return f"Datasets matching '{query}':\n" + "\n".join(lines)


def _dataset_size_bytes(dataset_repo_id: str) -> int | None:
    """Shared helper for get_dataset_info/search_compatible_lerobot_datasets.
    `used_storage` is a repo-level stat available without the heavier
    files_metadata=True call; only falls back to summing sibling file sizes
    (which does need files_metadata=True) when used_storage is missing.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        info = api.dataset_info(dataset_repo_id)
    except Exception:
        return None
    if info.used_storage:
        return info.used_storage

    try:
        info = api.dataset_info(dataset_repo_id, files_metadata=True)
    except Exception:
        return None
    if info.siblings:
        return sum((s.size or (s.lfs.size if s.lfs else 0) or 0) for s in info.siblings)
    return None


@tool
def get_dataset_info(dataset_repo_id: str, config: str | None = None, split: str = "train") -> str:
    """Get size, license, configs/splits, column schema, and a sample row for
    a Hugging Face dataset — without downloading it. Always call this and
    relay its size/license to the user before ever calling pull_dataset:
    pulling consumes real shared-cluster storage, so the user must explicitly
    confirm after seeing this info.

    Args:
        dataset_repo_id: Hugging Face dataset repo id, e.g. 'GEAR-Dreams/DreamZero-DROID'.
        config: Dataset config/subset name. Defaults to the first available one.
        split: Dataset split to preview a sample row from (default 'train').
    """
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
        f"{schema_section}"
    )


@tool
def validate_dataset_schema(
    dataset_repo_id: str,
    expected_columns: list[str] | None = None,
    config: str | None = None,
    split: str = "train",
) -> str:
    """Check a Hugging Face dataset's column schema against a target model's
    expected input format, without downloading it. Use this to check
    compatibility before recommending a dataset for fine-tuning a specific
    model — get the expected columns from that model's catalog README
    'Architecture'/'Intended Use' section (e.g. a robot policy model
    typically expects 'observation'/'action' keys; a video world model
    typically expects 'video'/'caption' or 'text'/'image' pairs).

    Args:
        dataset_repo_id: Hugging Face dataset repo id.
        expected_columns: Column names the target model requires. If
            omitted, this just reports the schema for manual comparison.
        config: Dataset config/subset name. Defaults to the first available one.
        split: Dataset split to check (default 'train').
    """
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

    if not expected_columns:
        return header

    missing = [c for c in expected_columns if c not in actual_columns]
    extra = sorted(actual_columns - set(expected_columns))
    if missing:
        return (
            f"{header}\n\n"
            f"INCOMPATIBLE: missing expected column(s) {missing}. "
            f"Present but unexpected: {extra or 'none'}."
        )
    return f"{header}\n\nCOMPATIBLE: all expected columns {expected_columns} are present."


# Tied to the pi0.5 recipe's specific training mechanism (tools/finetune_recipes.py),
# NOT a fixed platform-wide fact: that recipe trains via LeRobot's own native
# `lerobot-train` CLI, whose current releases default to the newer LeRobotDataset
# v3.0 format (v2.x needs an explicit conversion script or an older lerobot pin).
# An earlier version of this recipe trained via openpi's JAX scripts instead, which
# are pinned to the OLDER v2.x format -- the opposite requirement. If a future
# recipe for a different model uses a v2.x-only training mechanism again, it needs
# its own version check rather than sharing this one.
LEROBOT_COMPATIBLE_VERSION_PREFIX = "v3"


def _fetch_lerobot_info(dataset_repo_id: str) -> dict | str:
    """Shared helper for validate_lerobot_dataset/search_compatible_lerobot_datasets.
    Returns the parsed meta/info.json, or an error string.
    """
    from huggingface_hub import hf_hub_download
    import json

    try:
        info_path = hf_hub_download(repo_id=dataset_repo_id, repo_type="dataset", filename="meta/info.json")
    except Exception as e:
        return f"Could not fetch meta/info.json for '{dataset_repo_id}': {e}. Is this actually a LeRobot-format dataset?"

    with open(info_path) as f:
        return json.load(f)


@tool
def search_compatible_lerobot_datasets(
    query: str,
    expected_robot_type: str | None = None,
    expected_feature_keys: list[str] | None = None,
    max_size_gb: float | None = None,
    limit: int = 5,
) -> str:
    """Search Hugging Face Hub for LeRobot-format robot-policy datasets and
    filter out incompatible ones automatically. Prefer this over
    search_datasets when looking for a dataset to fine-tune a specific
    robot-policy model (e.g. pi0.5): a plain keyword search returns many
    plausible-sounding results that turn out to be the wrong LeRobot
    codebase_version, the wrong robot embodiment, or a different
    camera/state layout than the target recipe expects -- this tool checks
    each candidate's actual meta/info.json before returning it, instead of
    leaving that vetting for a human to do one dataset at a time afterward.

    Every returned result includes its real size in GB. If asked for "a
    smaller dataset", use max_size_gb to actually filter by size -- do NOT
    substitute episode count or download count as a size proxy, they don't
    correlate (confirmed: two DROID re-hosts with ~95,000 episodes each,
    same order of magnitude as the "larger" ones, have very different
    download counts).

    Args:
        query: Free-text search query, e.g. 'droid franka manipulation'.
        expected_robot_type: Substring to match against each candidate's
            robot_type field (case-insensitive), e.g. 'franka'. Omit to
            accept any robot type.
        expected_feature_keys: Exact LeRobot feature keys the target
            model's data config expects (dot-notation, e.g.
            'observation.images.wrist_image_left'). Only pass values that
            came from get_finetune_requirements or the user -- never a
            guess (real DROID re-hosts vary in exact naming). Omit to skip
            this check.
        max_size_gb: Skip candidates larger than this size in GB. Omit to
            accept any size.
        limit: Max number of COMPATIBLE datasets to return (more candidates
            than this may be checked internally to find them).
    """
    from huggingface_hub import HfApi

    try:
        # filter="LeRobot" narrows at the API level to datasets carrying that
        # tag, before we spend a per-candidate network call on each one --
        # confirmed empirically that every genuine LeRobot-format dataset
        # checked this session (including lerobot/droid_100) carries this
        # exact tag, so this shouldn't cause false negatives in practice.
        candidates = list(
            HfApi().list_datasets(search=query, filter="LeRobot", sort="downloads", limit=limit * 4)
        )
    except Exception as e:
        return f"Dataset search failed: {e}"

    if not candidates:
        return f"No datasets found matching '{query}' tagged as LeRobot-format."

    compatible = []
    skipped = []
    for d in candidates:
        if len(compatible) >= limit:
            break

        info = _fetch_lerobot_info(d.id)
        if isinstance(info, str):
            skipped.append(f"{d.id} (not LeRobot-format or meta/info.json unavailable)")
            continue

        codebase_version = info.get("codebase_version", "unknown")
        if not codebase_version.startswith(LEROBOT_COMPATIBLE_VERSION_PREFIX):
            skipped.append(f"{d.id} (codebase_version={codebase_version}, incompatible with openpi)")
            continue

        # robot_type is frequently left unpopulated ("unknown"/empty) even on
        # genuinely correct-embodiment datasets -- confirmed on lerobot/droid_100
        # itself. Only exclude on a confirmed DIFFERENT robot_type, not a missing
        # one; an unpopulated field can't confirm OR deny a match.
        robot_type = info.get("robot_type") or ""
        robot_type_known = bool(robot_type) and robot_type.lower() != "unknown"
        if expected_robot_type and robot_type_known and expected_robot_type.lower() not in robot_type.lower():
            skipped.append(f"{d.id} (robot_type='{robot_type}', expected '{expected_robot_type}')")
            continue

        features = info.get("features", {})
        if expected_feature_keys:
            missing = [k for k in expected_feature_keys if k not in features]
            if missing:
                skipped.append(f"{d.id} (missing feature key(s) {missing})")
                continue

        size_bytes = _dataset_size_bytes(d.id)
        size_gb = size_bytes / 1e9 if size_bytes else None
        if max_size_gb is not None and (size_gb is None or size_gb > max_size_gb):
            skipped.append(f"{d.id} (size={f'{size_gb:.2f}GB' if size_gb else 'unknown'}, over max_size_gb={max_size_gb})")
            continue

        size_str = f"{size_gb:.2f} GB" if size_gb is not None else "unknown"
        compatible.append(
            f"- {d.id}: size={size_str}, codebase_version={codebase_version}, robot_type={robot_type or 'unknown'}, "
            f"episodes={info.get('total_episodes', '?')}, downloads={d.downloads or 0}"
        )

    # Cap how many skip reasons get spelled out -- with up to limit*4
    # candidates checked, this list can otherwise run to 15-20+ entries and
    # consume a disproportionate share of a single tool call's token cost
    # within a turn (confirmed contributing to a real context-overflow
    # incident). Never silently drop the COUNT, just the verbose detail past
    # a point.
    MAX_SKIP_REASONS_SHOWN = 5
    skip_summary = "; ".join(skipped[:MAX_SKIP_REASONS_SHOWN])
    if len(skipped) > MAX_SKIP_REASONS_SHOWN:
        skip_summary += f"; and {len(skipped) - MAX_SKIP_REASONS_SHOWN} more (reasons omitted for brevity)"

    if not compatible:
        return (
            f"No compatible LeRobot-format datasets found matching '{query}' out of "
            f"{len(candidates)} candidate(s) checked.\n"
            f"Skipped: {skip_summary if skipped else 'none'}"
        )

    result = f"Compatible LeRobot-format datasets matching '{query}':\n" + "\n".join(compatible)
    result += f"\n\n({len(skipped)} incompatible candidate(s) filtered out of {len(candidates)} checked"
    if skipped:
        result += f": {skip_summary}"
    result += ".)"
    return result


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


@tool
def validate_lerobot_dataset(
    dataset_repo_id: str,
    model_name: str | None = None,
    expected_action_dim: int | None = None,
    expected_exterior_cameras: int | None = None,
    expected_wrist_cameras: int | None = None,
    expected_feature_keys: list[str] | None = None,
) -> str:
    """Check a robot-policy dataset's LeRobot-format metadata for
    fine-tuning compatibility, without downloading it. Unlike
    validate_dataset_schema (a generic flat-column check via the HF
    datasets-server, which doesn't understand LeRobot's structure), this
    reads the dataset's own meta/info.json directly -- action
    dimensionality, camera/observation feature keys, fps, episode count,
    and robot type all live there, not in flat "columns". Also checks the
    LeRobot codebase_version for compatibility with this platform's current
    fine-tuning recipe -- see LEROBOT_COMPATIBLE_VERSION_PREFIX (LeRobot's
    dataset format versions are not compatible with each other, and which
    one is required depends on the training mechanism a recipe uses, not a
    fixed platform-wide fact).

    STRONGLY PREFER passing model_name (e.g. 'pi05') over manually passing
    expected_exterior_cameras/expected_wrist_cameras/expected_action_dim
    yourself: model_name looks the real numbers up directly from the same
    recipe get_finetune_requirements reads, in code, with no re-typing step
    for you to get wrong. Manually copying numbers from a prior
    get_finetune_requirements call into this call's arguments has
    repeatedly gone wrong in practice (confirmed: swapped counts, both
    counts set to the same wrong value) even when the correct numbers were
    sitting right there in context -- model_name eliminates that transcription
    step entirely by not requiring it.

    Note: expected_action_dim compares the RAW dataset's action feature
    shape, not a model's internal (possibly padded/unified) action space --
    e.g. pi0.5 internally uses a 32-dim action space but real DROID data's
    raw action feature is 7-dim; the training pipeline's own input
    transforms handle that conversion.

    NEVER invent an expected_feature_keys value -- if you don't have one
    from get_finetune_requirements or the user, omit it and read the
    returned Features list yourself instead of asserting a match/mismatch.

    Args:
        dataset_repo_id: Hugging Face dataset repo id, e.g. 'lerobot/droid_100'.
        model_name: Fine-tuning recipe to check compatibility against (e.g.
            'pi05') -- auto-fills the expected_* args below from the same
            source get_finetune_requirements uses. Any expected_* arg passed
            explicitly overrides the auto-filled value for that field only.
        expected_action_dim: Raw action feature dimensionality to check for.
            Only pass this manually if you're not passing model_name.
        expected_exterior_cameras: Number of exterior-camera features
            expected (counts image/video features with 'exterior' in the
            key name, regardless of exact spelling). Only pass this
            manually if you're not passing model_name.
        expected_wrist_cameras: Number of wrist-camera features expected
            (same counting approach, for 'wrist'). Only pass this manually
            if you're not passing model_name.
        expected_feature_keys: Exact LeRobot feature keys, only if you
            already know them are correct for this exact dataset (e.g. the
            user gave them, or you already fetched this dataset's own
            Features list earlier this conversation) -- see warning above.
    """
    if model_name:
        try:
            from platform_agent.tools.finetune_recipes import get_requirements

            req = get_requirements(model_name)
            if expected_action_dim is None:
                expected_action_dim = req.get("expected_action_dim")
            if expected_exterior_cameras is None:
                expected_exterior_cameras = req.get("expected_exterior_cameras")
            if expected_wrist_cameras is None:
                expected_wrist_cameras = req.get("expected_wrist_cameras")
        except ValueError as e:
            return str(e)

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
def pull_dataset(dataset_repo_id: str, dataset_name: str, pvc_size_gb: int = 50) -> str:
    """Download a Hugging Face dataset onto the cluster so a fine-tuning job
    can read it. This consumes real shared-cluster storage — only call this
    after calling get_dataset_info and showing the user its size and license,
    and after the user has explicitly said to proceed. Never call this
    speculatively or as the first response to "find me a dataset for X".

    Downloads the entire dataset repo (config/subset selection happens at
    training time, not download time). Creates a PVC and a Kubernetes Job in
    the datasets namespace that runs huggingface_hub.snapshot_download.
    Check progress with get_dataset_job_status afterward — this tool returns
    as soon as the Job is created, not once the download finishes.

    Args:
        dataset_repo_id: Hugging Face dataset repo id to download, e.g.
            'GEAR-Dreams/DreamZero-DROID'.
        dataset_name: Short name for this staged dataset, lowercase
            alphanumeric and hyphens (used as the K8s resource name prefix).
        pvc_size_gb: Size of the PersistentVolumeClaim in GB. Should
            comfortably exceed the size reported by get_dataset_info.
    """
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

    download_script = f"""\
set -e
pip install -q huggingface_hub
python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import os
token = os.getenv('HF_TOKEN')
snapshot_download(
    repo_id='{dataset_repo_id}',
    repo_type='dataset',
    local_dir='/mnt/dataset',
    token=token,
)
PYEOF
"""

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
                                    "image": "python:3.11-slim",
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
                                        # `pip install` fails trying to write its cache/user-site
                                        # under that. Same root cause class as HF_HOME above, for
                                        # a different tool.
                                        {"name": "HOME", "value": "/tmp"},
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
