import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from platform_agent import media_store
from platform_agent.agent import build_agent

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
                        logger.info(
                            "tool result: name=%s status=%s content=%r",
                            name,
                            getattr(tm, "status", "unknown"),
                            str(getattr(tm, "content", ""))[:300],
                        )
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
                                [(tc.get("name"), tc.get("args")) for tc in tool_calls],
                            )
                        content = getattr(msg, "content", "")
                        if content and isinstance(content, str):
                            if not tool_calls:
                                response_text = content
    except Exception as e:
        response_text = f"Agent error: {e}"

    yield f"data: {json.dumps({'response': response_text or 'No response.', 'tools_called': tools_called})}\n\n"
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
