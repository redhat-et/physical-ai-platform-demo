"""
Canary tests for the skills migration: confirm the agent actually decides to
call get_skill(name) for the workflows whose detailed ordering/confirmation
rules now live only in skills/*.md rather than the always-on system prompt.

These are deliberately narrow -- if one of these passes but a related test in
test_agent_behavior.py fails (e.g. wrong ordering, missing confirmation),
that isolates the bug to skill *content*, not the triggering mechanism.
"""


def test_get_skill_called_for_dataset_search(chat):
    """Also takes over the ordering check the now-deleted
    test_get_finetune_requirements_called_before_search used to own: since
    get_finetune_requirements no longer exists, get_skill('datasets') is the
    only grounded source for pi05's real embodiment/camera/action facts --
    it must be called before search_compatible_lerobot_datasets, not after
    a guess."""
    result = chat("Find me a dataset to fine-tune the pi05 model")
    assert "get_skill" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_skill"]
    assert any((c["args"] or {}).get("name") == "datasets" for c in calls), calls
    if "search_compatible_lerobot_datasets" in result["tools_called"]:
        assert (
            result["tools_called"].index("get_skill")
            < result["tools_called"].index("search_compatible_lerobot_datasets")
        ), f"search happened before loading the datasets skill: {result['tools_called']}"


def test_get_skill_called_for_finetune_request(chat):
    result = chat(
        "Fine-tune pi05 using the dataset staged at PVC 'dataset-test-droid-pvc', "
        "call the experiment 'test-exp-1'"
    )
    assert "get_skill" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_skill"]
    assert any((c["args"] or {}).get("name") == "fine-tuning" for c in calls), calls


def test_get_skill_called_for_deploy_manifests(chat):
    result = chat(
        "Draft manifests to deploy Qwen/Qwen2.5-7B-Instruct as a new chat "
        "model called qwen25-eval-test, using 1 GPU."
    )
    assert "get_skill" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_skill"]
    assert any((c["args"] or {}).get("name") == "deploy-model" for c in calls), calls


def test_get_skill_called_for_unfamiliar_runtime(chat):
    """A model whose serving mechanism isn't vLLM/vLLM-Omni should route to
    new-model-runtime, not deploy-model (which would force it through
    generate_model_manifests' two hardcoded templates)."""
    result = chat(
        "I want to add a new robot-policy model to the catalog. It's served "
        "by its own custom Python inference server, not vLLM at all -- "
        "how do I set that up?"
    )
    assert "get_skill" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_skill"]
    assert any((c["args"] or {}).get("name") == "new-model-runtime" for c in calls), calls


def test_get_skill_called_for_deploy_checkpoint(chat):
    result = chat(
        "My fine-tuning run 'test-exp-1' finished -- deploy that checkpoint "
        "as a live model so I can try it out."
    )
    assert "get_skill" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_skill"]
    assert any((c["args"] or {}).get("name") == "deploy-checkpoint" for c in calls), calls


def test_get_skill_called_for_manage_models(chat):
    result = chat("Scale mocklm down to zero, I'm not using it right now")
    assert "get_skill" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_skill"]
    assert any((c["args"] or {}).get("name") == "manage-models" for c in calls), calls


def test_dreamzero_training_data_grounded_in_droid(chat):
    """A pure factual question about a model's training data (no search/
    pull/validate action involved) should still trigger the datasets skill
    -- its description was broadened specifically to cover this case -- and
    the answer should be grounded in the skill's real DROID lineage content,
    not fabricated or a generic 'I don't know'."""
    result = chat("What dataset was dreamzero trained on?")
    assert "get_skill" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_skill"]
    assert any((c["args"] or {}).get("name") == "datasets" for c in calls), calls
    assert "DROID" in (result["response"] or "").upper(), (
        f"Expected the grounded answer (DROID) in the response: {result}"
    )
