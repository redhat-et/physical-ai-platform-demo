import httpx
from langchain_core.tools import tool

from platform_agent.config import settings


@tool
def call_model(model_name: str, prompt: str, max_tokens: int = 512) -> str:
    """Send an inference request to a model through the MaaS proxy. If the
    model is scaled to zero, this will trigger it to scale up automatically
    and may take several minutes on the first request.

    Args:
        model_name: The model to call (e.g. 'mocklm-echo', 'cosmos3-nano').
        prompt: The user message to send to the model.
        max_tokens: Maximum tokens in the response (default 512).
    """
    url = (
        f"{settings.maas_proxy_url}/physical-ai-models/"
        f"{model_name}/v1/chat/completions"
    )

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    try:
        with httpx.Client(verify=False, timeout=300.0) as http_client:
            resp = http_client.post(
                url,
                json=payload,
                headers={"Authorization": "Bearer unused"},
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return f"Response from {model_name}:\n{content}"
    except httpx.TimeoutException:
        return (
            f"Request to '{model_name}' timed out. The model may still be "
            f"scaling up from zero — try again in a minute."
        )
    except httpx.HTTPStatusError as e:
        return f"Inference call to '{model_name}' failed: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Inference call to '{model_name}' failed: {e}"
