import importlib

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from openai import DefaultHttpxClient

from platform_agent.config import settings
from platform_agent.tools.skills import all_skills, get_skill, list_skills

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

# get_skill/list_skills are always bound -- they're how the agent finds and
# loads a skill in the first place, so SkillScopedToolsMiddleware can't wait
# for one of them to be called before making them available.
ALWAYS_AVAILABLE_TOOLS = [get_skill, list_skills]


def _discover_skill_tools() -> dict[str, list[BaseTool]]:
    """Auto-discovers every skill's tools.py in the skills repo and collects
    its @tool-decorated functions, keyed by the skill's logical name (the
    SKILL.md frontmatter name get_skill()/list_skills() already expose to the
    agent). Adding a new skill, or a new tool to an existing skill's
    tools.py, never requires touching this file -- as long as the skill
    follows the tools.py convention, it shows up here automatically. A skill
    with no tools.py (e.g. new-model-runtime, by design) is simply skipped.
    """
    discovered: dict[str, list[BaseTool]] = {}
    for skill in all_skills().values():
        try:
            module = importlib.import_module(f"platform_agent.skills.{skill.dir_name}.tools")
        except ModuleNotFoundError:
            continue
        discovered[skill.name] = [obj for obj in vars(module).values() if isinstance(obj, BaseTool)]
    return discovered


# Keyed by each skill's SKILL.md frontmatter `name:` field.
SKILL_TOOLS: dict[str, list[BaseTool]] = _discover_skill_tools()

# Flat view of every skill tool -- still what gets bound at graph-build time
# (extra_tools, e.g. MCP tools, get concatenated onto this); the middleware
# below is what actually narrows what the model sees on any given call.
TOOLS = ALWAYS_AVAILABLE_TOOLS + [t for skill_tools in SKILL_TOOLS.values() for t in skill_tools]


def _active_skill(messages) -> str | None:
    """The most recent get_skill(name=...) call in the conversation, if any
    -- this is what SkillScopedToolsMiddleware treats as "which skill is the
    agent currently working in." Scans from the end so a later get_skill
    call (switching skills mid-conversation) always wins over an earlier one.
    """
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            if tool_call.get("name") == "get_skill":
                return tool_call.get("args", {}).get("name")
    return None


class SkillScopedToolsMiddleware(AgentMiddleware):
    """Narrows the tool schema sent to the model on every call to just the
    always-available meta-tools (get_skill/list_skills) plus whichever
    skill's tools match the most recent get_skill() call -- instead of
    binding all tools across every skill unconditionally. This is the
    LangChain-native substitute for Anthropic's server-side Tool Search Tool,
    which isn't reachable here since this agent talks to a self-hosted vLLM
    endpoint through an OpenAI-compatible client, not Anthropic's API.
    """

    def wrap_model_call(self, request, handler):
        skill = _active_skill(request.messages)
        request.tools = ALWAYS_AVAILABLE_TOOLS + SKILL_TOOLS.get(skill, [])
        return handler(request)


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
                middleware=[SkillScopedToolsMiddleware()],
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
