import base64
import logging

import httpx
from langchain_core.tools import tool

from platform_agent import media_store
from platform_agent.config import settings

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
def call_model(
    model_name: str,
    prompt: str,
    output_kind: str = "chat",
    max_tokens: int = 512,
    num_frames: int = 41,
    num_inference_steps: int = 12,
):
    """Send an inference request to a model through the MaaS proxy. If the
    model is scaled to zero, this will trigger it to scale up automatically
    and may take several minutes on the first request.

    Before calling this on a model you haven't used before, check its
    "Output kind" (chat, image, video, or unsupported) via the models
    skill's CHECKING A SPECIFIC MODEL'S STATUS steps and pass that value as
    output_kind here — models don't all speak the same API. A model tagged
    "unsupported" has no compatible endpoint and should not be called.

    Args:
        model_name: The model to call (e.g. 'mocklm-echo', 'cosmos3-nano').
        prompt: The user's request/prompt to send to the model.
        output_kind: One of 'chat' (default), 'image', or 'video' — must
            match the model's advertised output kind (see the models skill).
        max_tokens: Maximum tokens in the response, for output_kind='chat' only.
        num_frames: Number of video frames to generate, for output_kind='video' only.
        num_inference_steps: Denoising steps, for output_kind='video' only
            (more steps = higher quality but slower; kept low by default to
            fit within the request timeout).
    """
    base_url = f"{settings.maas_proxy_url}/physical-ai-models/{model_name}"

    if output_kind == "chat":
        return _call_chat(base_url, model_name, prompt, max_tokens)
    if output_kind == "image":
        return _call_image(base_url, model_name, prompt)
    if output_kind == "video":
        return _call_video(base_url, model_name, prompt, num_frames, num_inference_steps)
    logger.warning(
        "call_model: unknown output_kind=%r requested for model=%r", output_kind, model_name
    )
    return (
        f"Unknown output_kind '{output_kind}' for '{model_name}'. Check the "
        f"models skill's CHECKING A SPECIFIC MODEL'S STATUS steps to find "
        f"the correct output kind — if it says 'unsupported', this model "
        f"has no compatible inference API.",
        None,
    )


def _call_chat(base_url: str, model_name: str, prompt: str, max_tokens: int):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    try:
        with httpx.Client(verify=False, timeout=300.0) as http_client:
            resp = http_client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer unused"},
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return f"Response from {model_name}:\n{content}", None
    except httpx.TimeoutException:
        logger.warning("call_model chat timeout: model=%r url=%s", model_name, base_url)
        return (
            f"Request to '{model_name}' timed out. The model may still be "
            f"scaling up from zero — try again in a minute.",
            None,
        )
    except httpx.HTTPStatusError as e:
        logger.warning(
            "call_model chat HTTP error: model=%r status=%s body=%r",
            model_name, e.response.status_code, e.response.text[:500],
        )
        return f"Inference call to '{model_name}' failed: HTTP {e.response.status_code}", None
    except Exception:
        logger.exception("call_model chat failed unexpectedly: model=%r", model_name)
        return f"Inference call to '{model_name}' failed unexpectedly — check agent logs.", None


def _call_image(base_url: str, model_name: str, prompt: str):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "response_format": "b64_json",
        "size": "1024x1024",
        "n": 1,
    }
    try:
        with httpx.Client(verify=False, timeout=300.0) as http_client:
            resp = http_client.post(
                f"{base_url}/v1/images/generations",
                json=payload,
                headers={"Authorization": "Bearer unused"},
            )
            resp.raise_for_status()
            data = resp.json()
            b64 = data["data"][0]["b64_json"]
            image_bytes = base64.b64decode(b64)
            media_id = media_store.put(image_bytes, "image/png")
            return (
                f"Generated an image with '{model_name}' for prompt: {prompt!r}",
                {"kind": "image", "media_id": media_id},
            )
    except httpx.TimeoutException:
        logger.warning("call_model image timeout: model=%r url=%s", model_name, base_url)
        return f"Image generation with '{model_name}' timed out.", None
    except httpx.HTTPStatusError as e:
        logger.warning(
            "call_model image HTTP error: model=%r status=%s body=%r",
            model_name, e.response.status_code, e.response.text[:500],
        )
        return f"Image generation with '{model_name}' failed: HTTP {e.response.status_code}", None
    except Exception:
        logger.exception("call_model image failed unexpectedly: model=%r", model_name)
        return f"Image generation with '{model_name}' failed unexpectedly — check agent logs.", None


def _call_video(
    base_url: str,
    model_name: str,
    prompt: str,
    num_frames: int,
    num_inference_steps: int,
):
    form_data = {
        "model": model_name,
        "prompt": prompt,
        "negative_prompt": "blurry, distorted, low quality",
        "size": "1280x720",
        "num_frames": str(num_frames),
        "fps": "24",
        "num_inference_steps": str(num_inference_steps),
        "guidance_scale": "6.0",
        "flow_shift": "10.0",
    }
    try:
        with httpx.Client(verify=False, timeout=260.0) as http_client:
            resp = http_client.post(
                f"{base_url}/v1/videos/sync",
                data=form_data,
                headers={"Authorization": "Bearer unused", "Accept": "video/mp4"},
            )
            resp.raise_for_status()
            media_id = media_store.put(resp.content, "video/mp4")
            return (
                f"Generated a video with '{model_name}' for prompt: {prompt!r}",
                {"kind": "video", "media_id": media_id},
            )
    except httpx.TimeoutException:
        logger.warning("call_model video timeout: model=%r url=%s", model_name, base_url)
        return (
            f"Video generation with '{model_name}' timed out. Try fewer "
            f"num_frames or num_inference_steps for a quicker result.",
            None,
        )
    except httpx.HTTPStatusError as e:
        logger.warning(
            "call_model video HTTP error: model=%r status=%s body=%r",
            model_name, e.response.status_code, e.response.text[:500],
        )
        return f"Video generation with '{model_name}' failed: HTTP {e.response.status_code}", None
    except Exception:
        logger.exception("call_model video failed unexpectedly: model=%r", model_name)
        return f"Video generation with '{model_name}' failed unexpectedly — check agent logs.", None
