import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from platform_agent import media_store
from platform_agent.agent import build_agent
from platform_agent.config import settings
from platform_agent.tools.models import get_model_readiness, resume_scaling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent_mode = None
agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_mode, agent
    agent_mode, agent = build_agent(use_tools=True)
    print(f"Agent started in '{agent_mode}' mode")
    yield


app = FastAPI(title="Physical AI Platform Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": agent_mode}


@app.get("/api/media/{media_id}")
def get_media(media_id: str):
    entry = media_store.get(media_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Media not found or expired.")
    data, mime = entry
    return Response(content=data, media_type=mime)


@app.get("/api/model/status")
def model_status():
    """Readiness check for the agent's own backing LLM — safe to call before
    the model is up, since it doesn't go through the LLM itself."""
    return get_model_readiness(settings.llm_model)


@app.post("/api/model/start")
async def start_model():
    """Fire a lightweight MaaS request to trigger KEDA's scale-from-zero via
    the HTTP interceptor. Doesn't wait for readiness — the UI polls
    /api/model/status separately for that."""
    resume_scaling(settings.llm_model)
    url = f"{settings.maas_proxy_url}/physical-ai-models/{settings.llm_model}/v1/models"
    try:
        async with httpx.AsyncClient(verify=False, timeout=5.0) as http_client:
            await http_client.get(url, headers={"Authorization": "Bearer unused"})
    except Exception:
        logger.info("start_model: trigger request to %s did not complete (expected while cold)", url)
    return {"status": "triggered"}


MAX_HISTORY_CHARS = 20000


def _trim_history(messages: list[dict]) -> list[dict]:
    total = 0
    trimmed = []
    for msg in reversed(messages):
        total += len(msg.get("content", ""))
        if total > MAX_HISTORY_CHARS:
            break
        trimmed.append(msg)
    return list(reversed(trimmed))


async def _stream_chat(messages: list[dict]):
    response_text = ""
    tools_called = []
    # Full detail (name + args the LLM actually decided on, plus the raw tool
    # result) for callers that need to verify more than just "a tool with
    # this name ran somewhere" -- e.g. tests asserting on tool arguments.
    tool_calls_detail = []
    try:
        async for chunk in agent.astream(
            {"messages": messages}, stream_mode="updates"
        ):
            for node, updates in chunk.items():
                if node == "tools":
                    for tm in updates.get("messages", []):
                        name = getattr(tm, "name", "tool")
                        tools_called.append(name)
                        yield f"data: {json.dumps({'status': f'Calling {name}...'})}\n\n"
                        content = getattr(tm, "content", "")
                        logger.info(
                            "tool result: name=%s status=%s",
                            name,
                            getattr(tm, "status", "unknown"),
                        )
                        # Raw tool output can be pod logs, model responses, or
                        # generated manifests -- arbitrary content that may be
                        # sensitive, so it's DEBUG-only, not INFO.
                        logger.debug(
                            "tool result content: name=%s content=%r",
                            name,
                            str(content)[:300],
                        )
                        tool_call_id = getattr(tm, "tool_call_id", None)
                        for call in tool_calls_detail:
                            if call.get("tool_call_id") == tool_call_id:
                                call["result"] = content if isinstance(content, str) else str(content)
                                break
                        artifact = getattr(tm, "artifact", None)
                        if isinstance(artifact, dict) and artifact.get("media_id"):
                            media = {
                                "kind": artifact["kind"],
                                "url": f"/api/media/{artifact['media_id']}",
                            }
                            yield f"data: {json.dumps({'media': media})}\n\n"
                elif node == "agent":
                    for msg in updates.get("messages", []):
                        tool_calls = getattr(msg, "tool_calls", None)
                        if tool_calls:
                            logger.info(
                                "agent requested tool calls: %s",
                                [tc.get("name") for tc in tool_calls],
                            )
                            # Args can contain arbitrary user-submitted text
                            # (e.g. call_model's prompt) -- DEBUG-only.
                            logger.debug(
                                "agent tool call args: %s",
                                [(tc.get("name"), tc.get("args")) for tc in tool_calls],
                            )
                            for tc in tool_calls:
                                tool_calls_detail.append(
                                    {
                                        "name": tc.get("name"),
                                        "args": tc.get("args"),
                                        "result": None,
                                        "tool_call_id": tc.get("id"),
                                    }
                                )
                        content = getattr(msg, "content", "")
                        if content and isinstance(content, str):
                            if not tool_calls:
                                response_text = content
    except Exception as e:
        logger.exception("agent stream failed")
        response_text = f"Agent error: {e}"

    yield f"data: {json.dumps({'response': response_text or 'No response.', 'tools_called': tools_called, 'tool_calls': tool_calls_detail})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]
    history = _trim_history(history)
    messages = history + [{"role": "user", "content": req.message}]

    return StreamingResponse(
        _stream_chat(messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
