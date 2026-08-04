from __future__ import annotations

import logging

import anyio
import mcp.types as types
from mcp.server.fastmcp import FastMCP

from pai_mcp_server import executor, skills
from pai_mcp_server.config import settings
from pai_mcp_server.executor import ScriptError
from pai_mcp_server.proxy import DownstreamProxy

logger = logging.getLogger(__name__)

app = FastMCP("pai-mcp-server", host=settings.host, port=settings.port)

# FastMCP's public tool API (`@app.tool()`) derives each tool's JSON schema
# from a Python function signature, which doesn't fit a server that also
# needs to re-expose arbitrary downstream MCP tool schemas discovered at
# runtime (the k8s_* passthrough). Reaching into the low-level Server FastMCP
# wraps internally is the documented escape hatch for exactly this case --
# `list_tools`/`call_tool` are its public, stable decorator API, so this is
# safe even though `_mcp_server` itself is a private attribute.
_server = app._mcp_server
_proxy = DownstreamProxy()

_LIST_SKILLS_TOOL = types.Tool(
    name="list_skills",
    description=(
        "List the narrow workflow skills available in the platform's skill "
        "catalog, with a one-line description of when to use each. Call "
        "this first for any request that might touch a specific workflow -- "
        "the one-line description alone is never enough to act on, so "
        "follow up with load_skill(name) on whichever one matches."
    ),
    inputSchema={"type": "object", "properties": {}},
)

_LOAD_SKILL_TOOL = types.Tool(
    name="load_skill",
    description=(
        "Load the full step-by-step procedure for one named skill, "
        "including which scripts it exposes and when to use each. Call "
        "this before your first action in a matching workflow -- the "
        "one-line description from list_skills is not enough to act on."
    ),
    inputSchema={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Exact skill name from list_skills."}},
        "required": ["name"],
    },
)

_GET_SCRIPT_TOOL = types.Tool(
    name="get_script",
    description=(
        "Return one script's calling contract -- its description and "
        "typed parameters -- without its implementation. Call this before "
        "run_script to learn exactly what arguments a script expects."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Exact skill name from list_skills."},
            "script": {"type": "string", "description": "Script filename, as named in the skill's SKILL.md."},
        },
        "required": ["skill", "script"],
    },
)

_RUN_SCRIPT_TOOL = types.Tool(
    name="run_script",
    description=(
        "Execute one script belonging to a skill, server-side, with the "
        "given named arguments. Arguments are validated against the "
        "script's declared parameters (see get_script) before anything runs."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "skill": {"type": "string"},
            "script": {"type": "string"},
            "args": {
                "type": "object",
                "description": "Named arguments matching get_script's parameter list for this script.",
            },
        },
        "required": ["skill", "script"],
    },
)

_NATIVE_TOOLS = [_LIST_SKILLS_TOOL, _LOAD_SKILL_TOOL, _GET_SCRIPT_TOOL, _RUN_SCRIPT_TOOL]


@_server.list_tools()
async def list_tools() -> list[types.Tool]:
    return _NATIVE_TOOLS + _proxy.list_proxied_tools()


@_server.call_tool()
async def call_tool(name: str, arguments: dict) -> dict | list[types.ContentBlock]:
    if _proxy.handles(name):
        return await _proxy.call(name, arguments)

    if name == "list_skills":
        return {"skills": skills.skills_index()}

    if name == "load_skill":
        skill = skills.get_skill(arguments["name"])
        if skill is None:
            raise ValueError(f"No skill named '{arguments['name']}'. Available: {sorted(skills.load_skills())}.")
        return {"name": skill.name, "body": skill.body}

    if name == "get_script":
        try:
            meta = executor.get_script_meta(arguments["skill"], arguments["script"])
        except (KeyError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        return meta.as_dict()

    if name == "run_script":
        try:
            return executor.run_script(arguments["skill"], arguments["script"], arguments.get("args", {}))
        except (KeyError, ValueError, ScriptError) as exc:
            raise ValueError(str(exc)) from exc

    raise ValueError(f"Unknown tool '{name}'")


async def _run() -> None:
    async with anyio.create_task_group() as tg:
        tg.start_soon(_proxy.run)
        await app.run_streamable_http_async()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    anyio.run(_run)


if __name__ == "__main__":
    main()
