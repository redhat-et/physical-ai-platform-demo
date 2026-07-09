import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from platform_agent.agent import build_agent

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


class ChatResponse(BaseModel):
    response: str


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": agent_mode}


async def _stream_agent(messages: list[dict]):
    try:
        final_text = ""
        async for chunk in agent.astream(
            {"messages": messages}, stream_mode="updates"
        ):
            for node, updates in chunk.items():
                if node == "tools":
                    tool_msgs = updates.get("messages", [])
                    for tm in tool_msgs:
                        name = getattr(tm, "name", "tool")
                        yield f"data: {json.dumps({'status': f'Called {name}'})}\n\n"
                elif node == "agent":
                    ai_msgs = updates.get("messages", [])
                    for msg in ai_msgs:
                        content = getattr(msg, "content", "")
                        if content and isinstance(content, str):
                            if not getattr(msg, "tool_calls", None):
                                final_text = content
        if final_text:
            yield f"data: {json.dumps({'token': final_text})}\n\n"
        else:
            yield f"data: {json.dumps({'token': 'No response.'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    if req.message and "stream" != "disabled":
        return StreamingResponse(
            _stream_agent(messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        if agent_mode == "agent":
            result = await agent.ainvoke({"messages": messages})
            ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content]
            response_text = ai_messages[-1].content if ai_messages else "No response."
        else:
            result = await agent.ainvoke({"input": req.message})
            response_text = result.content
    except Exception as e:
        response_text = f"Agent error: {e}"
    return ChatResponse(response=response_text)
