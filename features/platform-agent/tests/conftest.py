"""
Integration tests against the live platform_agent, exercising real tool-calling
behavior end to end (real LLM, real K8s API, real cluster state) -- not mocks.
The bugs this suite targets (fabricated tool results, wrong tool choice) are
LLM behavior, not Python logic, so a mocked-LLM test wouldn't catch them.

Run with the agent's port forwarded from the cluster:

    oc port-forward svc/platform-agent 8000:8000 -n physical-ai &
    pytest

Some tests (e.g. scaling) mutate real cluster state on the shared cluster.
"""
import json
import os

import httpx
import pytest

AGENT_URL = os.environ.get("PLATFORM_AGENT_URL", "http://localhost:8000")


def _parse_sse(text: str) -> dict:
    result = {"response": None, "tools_called": [], "tool_calls": []}
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        data = json.loads(payload)
        if "response" in data:
            result["response"] = data["response"]
            result["tools_called"] = data.get("tools_called", [])
            # Full detail per call: {"name": ..., "args": {...}, "result": "..."}
            # -- use this to assert on *what* a tool was called with, not just
            # that it was called; see tool_correctness in test_agent_behavior.py.
            result["tool_calls"] = data.get("tool_calls", [])
    return result


@pytest.fixture(scope="session", autouse=True)
def _agent_reachable():
    try:
        r = httpx.get(f"{AGENT_URL}/api/health", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        pytest.skip(f"platform_agent not reachable at {AGENT_URL}: {e}")


@pytest.fixture
def real_min_replicas():
    """Independently read an InferenceService's real minReplicas straight from
    the K8s API -- for verifying a scale_model call's actual effect, not just
    that the tool was invoked. Reuses the same client setup as the tool
    itself (platform_agent.tools.models._get_k8s_client), so this needs the
    same kubeconfig/in-cluster access the live agent has (already required
    for the port-forward this suite depends on)."""
    from platform_agent.tools.models import _get_k8s_client
    from platform_agent.config import settings

    def _get(model_name: str) -> int:
        custom_api, _ = _get_k8s_client()
        isvc = custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=settings.models_namespace,
            plural="inferenceservices",
            name=model_name,
        )
        return isvc["spec"]["predictor"]["minReplicas"]
    return _get


@pytest.fixture
def chat():
    """Send a message (with optional prior history) to the live agent.

    Returns {"response": str, "tools_called": [str, ...]} -- tools_called is
    in call order, so sequencing (e.g. "get_model_status before call_model")
    can be asserted on.
    """
    def _chat(message: str, history: list[dict] | None = None) -> dict:
        r = httpx.post(
            f"{AGENT_URL}/api/chat",
            json={"message": message, "history": history or []},
            timeout=180.0,
        )
        r.raise_for_status()
        return _parse_sse(r.text)
    return _chat
