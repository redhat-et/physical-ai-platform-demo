from __future__ import annotations

import logging

import anyio
import mcp.types as types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from pai_mcp_server.config import settings

logger = logging.getLogger(__name__)


class DownstreamProxy:
    """MCP client to openshift-mcp-server (its sidecar, reachable only from
    this process -- see Phase 3 manifests), re-exposing its tools under a
    `k8s_` prefix so agents reach it exclusively through this server's one
    connection instead of dialing it directly.

    run() must be started exactly once, as a background task in the same
    top-level anyio task group the HTTP server itself runs under (see
    server.py's main()) -- NOT connected lazily on first use from inside a
    per-request handler. anyio task groups/cancel scopes are tied to the
    task that created them: opening this connection inside one request's
    handler task and then reusing it from a later, unrelated request's
    handler task corrupts the whole MCP session (observed live against a
    real cluster: "Attempted to exit a cancel scope that isn't the current
    task's current cancel scope", right after an otherwise-successful
    call). Every other method here only reads/uses state `run()` maintains;
    none of them open the connection themselves.
    """

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._tools_by_prefixed_name: dict[str, types.Tool] = {}

    async def run(self) -> None:
        delay = 1
        while True:
            try:
                async with streamablehttp_client(settings.openshift_mcp_url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._session = session
                        await self._refresh_tools()
                        logger.info(
                            "connected to openshift-mcp-server at %s (%d tool(s) proxied)",
                            settings.openshift_mcp_url, len(self._tools_by_prefixed_name),
                        )
                        delay = 1
                        await anyio.sleep_forever()
            except Exception:
                logger.exception(
                    "openshift-mcp-server connection lost/unavailable at %s -- retrying in %ds",
                    settings.openshift_mcp_url, delay,
                )
            finally:
                self._session = None
                self._tools_by_prefixed_name = {}
            await anyio.sleep(delay)
            delay = min(delay * 2, 30)

    async def _refresh_tools(self) -> None:
        result = await self._session.list_tools()
        self._tools_by_prefixed_name = {
            f"{settings.k8s_tool_prefix}{tool.name}": tool.model_copy(
                update={"name": f"{settings.k8s_tool_prefix}{tool.name}"}
            )
            for tool in result.tools
        }

    def list_proxied_tools(self) -> list[types.Tool]:
        """Whatever's currently available -- empty if openshift-mcp-server
        is down or hasn't connected yet. Synchronous: just reads state
        run() maintains, no I/O of its own.
        """
        return list(self._tools_by_prefixed_name.values())

    def handles(self, name: str) -> bool:
        return name in self._tools_by_prefixed_name

    async def call(self, name: str, arguments: dict) -> list[types.ContentBlock]:
        if self._session is None:
            raise RuntimeError(f"openshift-mcp-server is currently unavailable -- cannot call '{name}'")
        downstream_name = name[len(settings.k8s_tool_prefix):]
        result = await self._session.call_tool(downstream_name, arguments)
        return result.content
