import httpx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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


def build_agent(use_tools: bool = True, extra_tools: list = ()):
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0,
        http_client=DefaultHttpxClient(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
    )

    # Every tool this agent has -- list_skills/load_skill/get_script/
    # run_script and the k8s_* passthrough -- comes from pai-mcp-server via
    # extra_tools (see main.py's _load_mcp_tools). No local tools, no shell:
    # this process holds no cluster credentials and executes nothing itself.
    if use_tools:
        try:
            agent = create_agent(
                llm,
                tools=list(extra_tools),
                system_prompt=SYSTEM_PROMPT,
                checkpointer=InMemorySaver(),
            )
            return ("agent", agent)
        except Exception:
            pass

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    return ("chain", chain)
