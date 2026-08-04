import asyncio
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel

from platform_agent import media_store
from platform_agent.agent import build_agent
from platform_agent.config import settings
from platform_agent.model_readiness import get_model_readiness, resume_scaling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent_mode = None
agent = None


MCP_SERVER_URL = "http://localhost:8080/mcp"
MCP_CONNECT_RETRIES = 5
MCP_CONNECT_BACKOFF_SECONDS = 2  # doubles each attempt: 2, 4, 8, 16 (~30s worst case)

_SCRIPT_PATH_RE = re.compile(r"[\w\-./]+\.(?:py|sh)\b")


def _classify_tool_call(name: str, args: dict | None) -> tuple[str, str]:
    """(category, label) for a tool call. category is 'skill' for a get_skill
    lookup, or 'used' for anything that actually ran (a skill's shell script,
    an MCP cluster tool, or any other tool) -- lets the UI say 'Loaded
    Skill: X' vs 'Used: Y' instead of the meaningless raw tool name for
    skills (every skill's every script runs through the same generic 'shell'
    tool). Every other tool name (list_skills, resources_list, pods_log, ...)
    is kept as-is -- unlike 'shell'/'get_skill', the name itself already says
    what ran.
    """
    args = args or {}
    if name == "get_skill":
        return "skill", args.get("name", "?")
    if name == "shell":
        command = (args.get("command") or "").strip()
        if not command:
            return "used", "Shell restart"
        scripts = _SCRIPT_PATH_RE.findall(command)
        if scripts:
            return "used", scripts[-1].rsplit("/", 1)[-1]
        return "used", command.splitlines()[0][:60]
    return "used", name


def _summarize_tool_calls(tool_calls_detail: list[dict]) -> list[str]:
    """Collapse a turn's tool calls into at most two entries -- 'Loaded
    Skill: ...' and/or 'Used: ...' -- deduped and order-preserving, present
    only when that category actually occurred this turn.
    """
    skills: list[str] = []
    used: list[str] = []
    for call in tool_calls_detail:
        category, label = _classify_tool_call(call.get("name"), call.get("args"))
        bucket = skills if category == "skill" else used
        if label not in bucket:
            bucket.append(label)
    summary = []
    if skills:
        summary.append(f"Loaded Skill: {', '.join(skills)}")
    if used:
        summary.append(f"Used: {', '.join(used)}")
    return summary


async def _load_mcp_tools() -> list:
    """Cluster-access tools (resource get/list/scale, pod logs, ...) served by
    the openshift-mcp-server sidecar. Falls back to no MCP tools, rather than
    blocking startup forever, once retries are exhausted.
    """
    client = MultiServerMCPClient(
        {"openshift": {"url": MCP_SERVER_URL, "transport": "streamable_http"}}
    )
    for attempt in range(1, MCP_CONNECT_RETRIES + 1):
        try:
            return await client.get_tools()
        except Exception:
            if attempt == MCP_CONNECT_RETRIES:
                logger.exception(
                    "could not load tools from openshift-mcp-server at %s after %d attempt(s)",
                    MCP_SERVER_URL, attempt,
                )
                return []
            delay = MCP_CONNECT_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "openshift-mcp-server not reachable at %s (attempt %d/%d) -- retrying in %ds",
                MCP_SERVER_URL, attempt, MCP_CONNECT_RETRIES, delay,
            )
            await asyncio.sleep(delay)
    return []  # unreachable -- every branch above returns


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_mode, agent
    mcp_tools = await _load_mcp_tools()
    agent_mode, agent = build_agent(use_tools=True, extra_tools=mcp_tools)
    print(f"Agent started in '{agent_mode}' mode with {len(mcp_tools)} MCP tool(s)")
    yield


app = FastAPI(title="Physical AI Platform Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


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


async def _stream_chat(messages: list[dict], thread_id: str, disposable_thread: bool = False):
    response_text = ""
    tool_calls_detail = []
    try:
        async for chunk in agent.astream(
            {"messages": messages},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            for node, updates in chunk.items():
                if node == "tools":
                    for tm in updates.get("messages", []):
                        name = getattr(tm, "name", "tool")
                        tool_call_id = getattr(tm, "tool_call_id", None)
                        call = next(
                            (c for c in tool_calls_detail if c.get("tool_call_id") == tool_call_id),
                            None,
                        )
                        category, label = _classify_tool_call(name, call.get("args") if call else None)
                        status_text = f"Loaded Skill: {label}..." if category == "skill" else f"Used: {label}..."
                        yield f"data: {json.dumps({'status': status_text})}\n\n"
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
                        if call is not None:
                            call["result"] = content if isinstance(content, str) else str(content)
                        artifact = getattr(tm, "artifact", None)
                        if isinstance(artifact, dict) and artifact.get("media_id"):
                            media = {
                                "kind": artifact["kind"],
                                "url": f"/api/media/{artifact['media_id']}",
                            }
                            yield f"data: {json.dumps({'media': media})}\n\n"
                elif node == "model":
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
    finally:
        if disposable_thread:
            agent.checkpointer.delete_thread(thread_id)

    tools_called = _summarize_tool_calls(tool_calls_detail)
    yield f"data: {json.dumps({'response': response_text or 'No response.', 'tools_called': tools_called, 'tool_calls': tool_calls_detail})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    disposable_thread = req.thread_id is None
    thread_id = req.thread_id or str(uuid.uuid4())
    messages = [{"role": "user", "content": req.message}]

    return StreamingResponse(
        _stream_chat(messages, thread_id, disposable_thread=disposable_thread),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
