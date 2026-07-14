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
    # list_models's real output format is "- {name}: ready=... minReplicas=...
    # url=..." -- check the raw tool result, not just the final paraphrase,
    # to confirm it actually hit the real API rather than fabricating a list.
    assert "ready=" in (calls[0]["result"] or ""), calls[0]
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
