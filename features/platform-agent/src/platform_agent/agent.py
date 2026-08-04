import os
import sys
from pathlib import Path

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware.shell_tool import HostExecutionPolicy, ShellToolMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from openai import DefaultHttpxClient

from platform_agent.config import settings
from platform_agent.tools.skills import get_skill, list_skills

_SHELL_ENV = {**os.environ, "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])}

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

TOOLS = [get_skill, list_skills]


def build_agent(use_tools: bool = True, extra_tools: list = ()):
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0,
        http_client=DefaultHttpxClient(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
    )

    if use_tools:
        try:
            agent = create_agent(
                llm,
                tools=[*TOOLS, *extra_tools],
                system_prompt=SYSTEM_PROMPT,
                middleware=[
                    ShellToolMiddleware(
                        execution_policy=HostExecutionPolicy(),
                        env={**_SHELL_ENV, "SKILLS_ROOT": settings.skills_root},
                    ),
                ],
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
