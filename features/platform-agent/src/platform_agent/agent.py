import httpx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from openai import DefaultHttpxClient

from platform_agent.config import settings

_NAMESPACES = ", ".join(dict.fromkeys([
    settings.models_namespace,
    settings.infra_namespace,
    settings.maas_namespace,
    settings.datasets_namespace,
]))

SYSTEM_PROMPT = (
    settings.system_prompt
    .replace("{model}", settings.llm_model)
    .replace("{namespaces}", _NAMESPACES)
)


def build_agent(extra_tools: list = ()):
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0,
        http_client=DefaultHttpxClient(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
    )

    agent = create_agent(
        llm,
        tools=list(extra_tools),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return ("agent", agent)
