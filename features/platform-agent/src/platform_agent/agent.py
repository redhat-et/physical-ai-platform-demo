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

# ShellToolMiddleware's env replaces the spawned shell's whole environment
# rather than extending it, so PATH must be rebuilt explicitly here or every
# skill script fails with ModuleNotFoundError for httpx/kubernetes/etc.
# (confirmed live). Putting sys.executable's own directory first on PATH --
# rather than just copying os.environ's PATH verbatim -- guarantees the
# shell's `python3` resolves to this exact interpreter (with all its
# installed deps) regardless of whether this process was launched from an
# activated venv, a direct venv/bin/python invocation, or the container's
# single system python3.
_SHELL_ENV = {**os.environ, "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])}

# Deduped, order-preserving -- infra_namespace and datasets_namespace are
# both "physical-ai" today but aren't guaranteed to stay that way, and this
# drives RULE 5's namespace list, so it must reflect the real settings
# rather than being hardcoded separately in the prompt text.
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

# get_skill/list_skills are how the agent finds and loads a skill's
# instructions in the first place. Every skill's actual capabilities now
# live as plain CLI scripts (see skills/<name>/scripts/ in the skills repo)
# run through ShellToolMiddleware's generic shell tool below, not as
# bespoke per-skill @tool functions -- so there's nothing left to
# auto-discover or scope per active skill.
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
                    # Generic shell-exec tool for every skill's CLI scripts
                    # (see e.g. skills/datasets/scripts/) -- HostExecutionPolicy
                    # runs commands directly in this container, which already
                    # carries the platform-agent ServiceAccount's RBAC.
                    # $SKILLS_ROOT lets SKILL.md docs reference script paths
                    # without hardcoding the container's install path. See
                    # _SHELL_ENV above for why PATH is rebuilt explicitly.
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
