"""
Canary tests for the skills migration: confirm the agent actually decides to
call get_skill(name) for the workflows whose detailed ordering/confirmation
rules now live only in skills/*.md rather than the always-on system prompt.

These are deliberately narrow -- if one of these passes but a related test in
test_agent_behavior.py fails (e.g. wrong ordering, missing confirmation),
that isolates the bug to skill *content*, not the triggering mechanism.
"""


def test_get_skill_called_for_dataset_search(chat):
    result = chat("Find me a dataset to fine-tune the pi05 model")
    assert "get_skill" in result["tools_called"], result
    calls = [c for c in result["tool_calls"] if c["name"] == "get_skill"]
    assert any((c["args"] or {}).get("name") == "datasets" for c in calls), calls


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
