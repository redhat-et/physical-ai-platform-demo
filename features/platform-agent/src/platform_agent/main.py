from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": agent_mode}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        if agent_mode == "agent":
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": req.message}]}
            )
            ai_messages = [m for m in result["messages"] if m.type == "ai" and m.content]
            response_text = ai_messages[-1].content if ai_messages else "No response."
        else:
            result = await agent.ainvoke({"input": req.message})
            response_text = result.content
    except Exception as e:
        response_text = f"Agent error: {e}"
    return ChatResponse(response=response_text)
