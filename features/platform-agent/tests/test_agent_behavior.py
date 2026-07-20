"""
Regression tests for specific tool-use bugs found in manual testing. Each test
maps to a real incident -- see the docstring for what actually happened.
"""

# Known GPU products that don't exist on this cluster (only NVIDIA-L40S does).
# If a response mentions one of these, the model fabricated it.
FABRICATED_GPUS = ["A100", "H100", "H200", "V100", "T4", "A10"]


def test_gpu_question_calls_tool_not_fabricated(chat):
    """Regression: asked 'What GPU compute is available?' and the agent
    invented a full JSON blob for a nonexistent NVIDIA-A100 GPU instead of
    calling list_cluster_gpus."""
    result = chat("What GPU compute is available?")
    assert "list_cluster_gpus" in result["tools_called"], (
        f"Expected list_cluster_gpus to be called, got: {result}"
    )
    for gpu in FABRICATED_GPUS:
        assert gpu not in result["response"], (
            f"Response mentions '{gpu}', which isn't on this cluster: "
            f"{result['response']!r}"
        )


def _scale_model_calls(result: dict) -> list[dict]:
    return [c for c in result["tool_calls"] if c["name"] == "scale_model"]


def test_scale_up_then_down_then_retry_all_call_tool_correctly(chat, real_min_replicas):
    """Regression: after scaling qwen25-cpu up, three separate 'scale it
    down' / 'try again' requests all got a confident-sounding reply, but
    scale_model was never called again -- the model was still running the
    whole time. Later found a second failure mode: the tool WAS called on
    a later turn, but with a corrupted model_name ('gc-qwen25-cpu' instead
    of 'qwen25-cpu').

    Checks three things per turn, which is stricter than tool-name-only
    trajectory checking (see 'tool correctness' / 'argument correctness' in
    agent-eval literature -- selecting the right tool with wrong arguments
    is a distinct failure mode from not calling it at all):
      1. scale_model was actually called this turn (not fabricated).
      2. It was called with the correct model_name (not a corrupted one).
      3. After the sequence, the real InferenceService is actually at the
         expected minReplicas -- independently verified via the K8s API,
         not just trusted because the tool claims it happened.

    Runs all three turns and collects every failure rather than aborting at
    the first, since each turn is an expensive real LLM call.
    """
    turns = [
        ("Can you scale up qwen25-cpu?", 1),
        ("Actually, can you scale it down?", 0),
        ("Can you try again, it looks like it's still up", 0),
    ]
    history = []
    failures = []

    for i, (message, expected_min_replicas) in enumerate(turns, start=1):
        result = chat(message, history)
        calls = _scale_model_calls(result)
        if not calls:
            failures.append(
                f"Turn {i} {message!r}: no scale_model call — "
                f"tools_called={result['tools_called']}, "
                f"response={result['response']!r}"
            )
        else:
            args = calls[-1]["args"] or {}
            if args.get("model_name") != "qwen25-cpu":
                failures.append(
                    f"Turn {i} {message!r}: scale_model called with wrong "
                    f"model_name {args.get('model_name')!r}, expected "
                    f"'qwen25-cpu' — full args={args}"
                )
            if args.get("min_replicas") != expected_min_replicas:
                failures.append(
                    f"Turn {i} {message!r}: scale_model called with "
                    f"min_replicas={args.get('min_replicas')!r}, expected "
                    f"{expected_min_replicas} — full args={args}"
                )
        history += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["response"]},
        ]

    actual = real_min_replicas("qwen25-cpu")
    if actual != 0:
        failures.append(
            f"After the full sequence, qwen25-cpu's real minReplicas is "
            f"{actual}, expected 0 — the last scale-down didn't actually "
            f"take effect even if scale_model reported success"
        )

    assert not failures, "\n".join(failures)


def test_call_model_checks_status_before_calling(chat):
    """get_model_status must be called before call_model, to know the
    model's real output_kind rather than guessing it."""
    result = chat("Ask mocklm-echo to say hello")
    assert "get_model_status" in result["tools_called"], result
    if "call_model" in result["tools_called"]:
        assert (
            result["tools_called"].index("get_model_status")
            < result["tools_called"].index("call_model")
        ), f"call_model happened before get_model_status: {result['tools_called']}"


def test_unsupported_model_is_not_called(chat):
    """dreamzero is tagged output-kind 'unsupported' -- call_model has no
    compatible API for it and should never be invoked."""
    result = chat("Can you ask dreamzero to say hello?")
    assert "call_model" not in result["tools_called"], (
        f"dreamzero has no compatible API and should not be called: {result}"
    )


def test_generate_manifests_for_chat_model_uses_stock_vllm(chat):
    """Regression: generate_model_manifests always used the vLLM-Omni image
    and --omni flag, even for a plain text chat model that doesn't need
    Omni's multimodal/diffusion orchestration -- it should use the stock
    vllm/vllm-openai image + hermes tool-call-parser instead, matching the
    hand-built qwen25-gpu model."""
    result = chat(
        "Draft manifests to deploy Qwen/Qwen2.5-7B-Instruct as a new chat "
        "model called qwen25-eval-test, using 1 GPU."
    )
    assert "generate_model_manifests" in result["tools_called"], result
    assert "automation-vllm-omni" not in result["response"], (
        "Plain chat model should use the stock vllm/vllm-openai image, not "
        f"vLLM-Omni: {result['response']!r}"
    )
    assert "--omni" not in result["response"]


def test_list_models_called_for_informal_reference(chat):
    """RULE 1: an informal/nickname reference to a model ('the echo model')
    should trigger list_models to resolve the real name, not a guess or a
    request for clarification."""
    result = chat("Can you use the echo model?")
    assert "list_models" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "list_models"]
    assert calls, result
    # list_models's real output format is "- {name}: status=... minReplicas=...
    # url=..." -- check the raw tool result, not just the final paraphrase,
    # to confirm it actually hit the real API rather than fabricating a list.
    assert "status=" in (calls[0]["result"] or ""), calls[0]
    assert "mocklm-echo" in (result["response"] or ""), (
        f"Expected the resolved name 'mocklm-echo' in the response: {result}"
    )


def test_get_pod_logs_called_for_log_request(chat):
    """A direct request for a model's logs should call get_pod_logs against
    the real pod, not fabricate log content."""
    result = chat("Can you show me the recent logs for mocklm-echo?")
    assert "get_pod_logs" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_pod_logs"]
    assert calls, result
    args = calls[0]["args"] or {}
    assert args.get("model_name") == "mocklm-echo", args
    # Real tool output starts with "Logs from {pod_name} (last N lines):" or
    # "No pods found for model ...". Either is a real result; empty/missing
    # is not.
    assert calls[0]["result"], calls[0]


def test_search_datasets_called_for_find_dataset_query(chat):
    """A request to find a dataset for a non-robot-policy task should call
    search_datasets, not fabricate a plausible-sounding dataset name/id.
    Deliberately NOT robot-policy-flavored (e.g. not "robot manipulation")
    -- that phrasing now correctly triggers the smarter
    get_finetune_requirements/search_compatible_lerobot_datasets path
    instead, which is the improved, intended behavior, not a failure."""
    result = chat("Find me a dataset for training a text sentiment classification model")
    assert "search_datasets" in result["tools_called"], result


def test_search_compatible_lerobot_datasets_used_for_robot_policy(chat):
    """Regression: asked to find a dataset for fine-tuning pi05, the agent
    called plain search_datasets and dumped an unvetted list (including
    wrong-robot-embodiment and wrong-LeRobot-version results) for the user
    to sort through one at a time. For a named robot-policy model,
    search_compatible_lerobot_datasets (which checks each candidate's real
    metadata before returning it) should be used instead."""
    result = chat("Find me a dataset to fine-tune the pi05 model")
    assert "search_compatible_lerobot_datasets" in result["tools_called"], result


def test_get_finetune_requirements_called_before_search(chat):
    """Regression: asked to find a dataset for pi05, the agent searched for
    the model's own name ('pi05') and got back datasets for unrelated
    embodiments (LIBERO sim, humanoids, custom rigs) instead of DROID data
    this recipe actually needs. get_finetune_requirements must be called
    before search_compatible_lerobot_datasets so the search is grounded in
    the recipe's real robot_type/query hint, not a guess."""
    result = chat("Find me a dataset to fine-tune the pi05 model")
    assert "get_finetune_requirements" in result["tools_called"], result
    if "search_compatible_lerobot_datasets" in result["tools_called"]:
        assert (
            result["tools_called"].index("get_finetune_requirements")
            < result["tools_called"].index("search_compatible_lerobot_datasets")
        ), f"search happened before checking requirements: {result['tools_called']}"


def test_validate_lerobot_dataset_grounded_before_asserting_incompatible(chat):
    """Regression: asked to validate 'aractingi/droid_100_test' (a real,
    confirmed-compatible dataset) against pi05, the agent called
    validate_lerobot_dataset with fabricated expected_feature_keys
    ('observation.images.wrist_image_left'/'wrist_image_right' -- wrong
    prefix, and DROID doesn't even have a right wrist camera) and reported
    a confident but false INCOMPATIBLE verdict. get_finetune_requirements
    must be called first so any expected_* criteria passed are grounded in
    the real recipe, not invented."""
    result = chat("Can you validate aractingi/droid_100_test for fine-tuning pi05?")
    assert "validate_lerobot_dataset" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "validate_lerobot_dataset"]
    assert calls, result
    args = calls[0]["args"] or {}
    if args.get("expected_feature_keys") or args.get("expected_action_dim"):
        assert "get_finetune_requirements" in result["tools_called"], (
            f"validate_lerobot_dataset was called with expected_* criteria "
            f"but get_finetune_requirements (the only grounded source for "
            f"them) was never called -- criteria are likely fabricated: {result}"
        )


def test_validate_lerobot_dataset_uses_camera_counts_not_invented_keys(chat):
    """Regression: asked to check camera compatibility for pi05, the agent
    invented an exact feature key ('observation.images.wrist_image_right')
    that has never existed in any real DROID dataset checked on this
    platform -- DROID has exactly one wrist camera, never two. Camera
    compatibility should be grounded, not a guessed expected_feature_keys
    string -- either via model_name (preferred: looks the counts up
    directly, no transcription step to get wrong) or, failing that, the
    manually-passed count-based expected_exterior_cameras/
    expected_wrist_cameras args."""
    result = chat(
        "Is lerobot/droid_100 compatible with pi05's expected camera setup?"
    )
    assert "validate_lerobot_dataset" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "validate_lerobot_dataset"]
    assert calls, result
    args = calls[-1]["args"] or {}
    invented_keys = [
        k for k in (args.get("expected_feature_keys") or [])
        if "wrist_image_right" in k or "wrist_right" in k
    ]
    assert not invented_keys, (
        f"validate_lerobot_dataset was called with an invented camera key "
        f"that has never existed in any real DROID dataset: {invented_keys}. "
        f"Full args: {args}"
    )
    grounded = (
        args.get("model_name")
        or args.get("expected_exterior_cameras") is not None
        or args.get("expected_wrist_cameras") is not None
    )
    assert grounded, (
        f"Expected the agent to check camera compatibility via model_name "
        f"(preferred) or the count-based expected_exterior_cameras/"
        f"expected_wrist_cameras args, rather than no check at all or a "
        f"guessed expected_feature_keys: {args}"
    )


def test_search_smaller_dataset_uses_real_size_filter(chat):
    """Regression: after being shown several large DROID datasets and asked
    for 'a smaller dataset', the agent claimed two ~95,000-episode datasets
    were 'relatively small compared to the others' based on download count
    -- a fabricated size claim (episode/download count don't correlate with
    actual size; the real sizes were 400+ GB each). Asking for a smaller
    dataset should trigger a new search_compatible_lerobot_datasets call
    using max_size_gb, which reports real size, not a re-narration of the
    same results using download count as a size proxy."""
    turn1 = chat("Find me a dataset to fine-tune pi05")
    history = [
        {"role": "user", "content": "Find me a dataset to fine-tune pi05"},
        {"role": "assistant", "content": turn1["response"]},
    ]
    turn2 = chat("Can you find a smaller one?", history)
    calls = [c for c in turn2["tool_calls"] if c["name"] == "search_compatible_lerobot_datasets"]
    assert calls, (
        f"Expected a new search_compatible_lerobot_datasets call to actually "
        f"filter by size, not just a re-narrated answer from memory: {turn2}"
    )
    assert calls[-1]["args"].get("max_size_gb") is not None, (
        f"search_compatible_lerobot_datasets was called again but without "
        f"max_size_gb -- 'smaller' has no real size data behind it "
        f"otherwise: {calls[-1]}"
    )


def test_get_dataset_info_called_with_exact_repo_id(chat):
    """A question about a specific dataset should call get_dataset_info with
    the exact repo id, not guess its size/license/schema."""
    result = chat("What's the size and license of the squad dataset (repo id 'squad')?")
    assert "get_dataset_info" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_dataset_info"]
    assert calls, result
    assert calls[0]["args"].get("dataset_repo_id") == "squad", calls[0]


def test_pull_dataset_requires_confirmation_after_info(chat):
    """DATASETS rule: pull_dataset must never be called in the same turn as
    get_dataset_info -- the agent must show size/license and wait for
    explicit user go-ahead before consuming shared-cluster storage on a
    download. Turn 1 asks to pull directly (info not yet shown this
    conversation); turn 2 confirms after seeing it."""
    turn1 = chat("Pull the 'squad' dataset (repo id 'squad') so I can fine-tune with it")
    assert "get_dataset_info" in turn1["tools_called"], turn1
    assert "pull_dataset" not in turn1["tools_called"], (
        f"pull_dataset must not be called before the user has seen size/license "
        f"and confirmed: {turn1}"
    )

    history = [
        {"role": "user", "content": "Pull the 'squad' dataset (repo id 'squad') so I can fine-tune with it"},
        {"role": "assistant", "content": turn1["response"]},
    ]
    turn2 = chat("Yes, go ahead and pull it", history)
    assert "pull_dataset" in turn2["tools_called"], turn2
    calls = [c for c in turn2["tool_calls"] if c["name"] == "pull_dataset"]
    assert calls, turn2
    assert calls[0]["args"].get("dataset_repo_id") == "squad", calls[0]


def test_submit_finetune_run_requires_confirmation(chat):
    """FINE-TUNING rule: submit_finetune_run must never be called as the
    first response to a fine-tuning request -- it consumes real GPU-hours
    on the shared cluster, so the agent must discuss the recipe and wait
    for explicit user go-ahead first, same carve-out as pull_dataset."""
    message = (
        "Fine-tune pi05 using the dataset staged at PVC 'dataset-test-droid-pvc', "
        "call the experiment 'test-exp-1'"
    )
    turn1 = chat(message)
    assert "submit_finetune_run" not in turn1["tools_called"], (
        f"submit_finetune_run must not be called before the user has confirmed: {turn1}"
    )

    history = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": turn1["response"]},
    ]
    turn2 = chat("Yes, go ahead and start it", history)
    assert "submit_finetune_run" in turn2["tools_called"], turn2
    calls = [c for c in turn2["tool_calls"] if c["name"] == "submit_finetune_run"]
    assert calls, turn2
    args = calls[0]["args"] or {}
    assert args.get("dataset_pvc_name") == "dataset-test-droid-pvc", calls[0]
    assert args.get("exp_name") == "test-exp-1", calls[0]


def test_get_finetune_run_status_called_for_status_query(chat):
    """A question about a fine-tuning run's progress should call
    get_finetune_run_status with the exact exp_name, not guess or reuse a
    stale answer from an earlier turn (state changes between messages)."""
    result = chat("What's the status of the 'test-exp-1' fine-tuning run?")
    assert "get_finetune_run_status" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_finetune_run_status"]
    assert calls, result
    assert calls[0]["args"].get("exp_name") == "test-exp-1", calls[0]


def test_list_finetune_runs_called_for_general_status_query(chat):
    """Regression: asked broadly about fine-tuning runs with no exp_name
    given (e.g. because it was forgotten), the agent had no way to look
    one up -- get_finetune_run_status requires the exact name and there
    was no listing tool, so a forgotten name meant the run was
    unrecoverable. list_finetune_runs should be called instead of
    guessing a name or claiming there's no way to check."""
    result = chat("What fine-tuning runs are currently going on?")
    assert "list_finetune_runs" in result["tools_called"], result


def test_list_staged_datasets_called_for_staged_query(chat):
    """Asking what's already staged should call list_staged_datasets, not
    fabricate an answer from memory (state can change between turns)."""
    result = chat("What datasets are already staged on the cluster?")
    assert "list_staged_datasets" in result["tools_called"], result


def test_estimate_model_footprint_called_with_exact_repo_id(chat):
    """A sizing/hardware question about a specific HF model should call
    estimate_model_footprint with the exact repo id, not guess VRAM numbers
    itself."""
    result = chat("How much VRAM would Qwen/Qwen2.5-7B-Instruct need on an L40S?")
    assert "estimate_model_footprint" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "estimate_model_footprint"]
    assert calls, result
    args = calls[0]["args"] or {}
    assert args.get("hf_repo_id") == "Qwen/Qwen2.5-7B-Instruct", args
    # Real tool output mentions GB and a tensor_parallel_size recommendation;
    # a fabricated answer wouldn't have called the tool at all (checked
    # above), but confirm the real result actually landed too.
    assert "GB" in (calls[0]["result"] or ""), calls[0]
