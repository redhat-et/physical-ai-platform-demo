from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

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

    Connects lazily on first use rather than via a server lifespan hook:
    the low-level mcp.server.Server's lifespan context is scoped per
    client *session*, not per process, so tying the proxy's connection to
    it would reconnect (or misbehave) on every new agent session. A
    lazily-created, lock-guarded singleton tied to this process's single
    event loop (see server.py's anyio.run) is simpler and correct
    regardless of how many sessions the transport creates.
    """

    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tools_by_prefixed_name: dict[str, types.Tool] = {}
        self._lock = asyncio.Lock()

    async def _ensure_connected(self) -> None:
        if self._session is not None:
            return
        async with self._lock:
            if self._session is not None:
                return
            # Deliberately no try/except-and-aclose around this: closing a
            # streamablehttp_client generator after it's already failed
            # internally (e.g. connection refused mid-task-group) raises its
            # own "asynchronous generator is already running" error instead
            # of the original one. On failure we just drop `stack`
            # unentered-on-self and let the next call build a fresh one.
            stack = AsyncExitStack()
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(settings.openshift_mcp_url)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._stack = stack
            self._session = session
            await self._refresh_tools()

    async def _refresh_tools(self) -> None:
        result = await self._session.list_tools()
        self._tools_by_prefixed_name = {
            f"{settings.k8s_tool_prefix}{tool.name}": tool.model_copy(
                update={"name": f"{settings.k8s_tool_prefix}{tool.name}"}
            )
            for tool in result.tools
        }

    async def list_proxied_tools(self) -> list[types.Tool]:
        try:
            await self._ensure_connected()
        # A refused connection surfaces here as asyncio.CancelledError, not a
        # plain Exception: streamablehttp_client's internal anyio task group
        # cancels its sibling receive-task when the POST task fails, and
        # that cancellation is *of our own nested task group*, not an
        # external shutdown signal to this coroutine -- safe to swallow.
        except (Exception, asyncio.CancelledError):
            logger.exception(
                "openshift-mcp-server unreachable at %s -- k8s_* tools unavailable this call",
                settings.openshift_mcp_url,
            )
            return []
        return list(self._tools_by_prefixed_name.values())

    async def handles(self, name: str) -> bool:
        if not self._tools_by_prefixed_name:
            await self.list_proxied_tools()
        return name in self._tools_by_prefixed_name

    async def call(self, name: str, arguments: dict) -> list[types.ContentBlock]:
        await self._ensure_connected()
        downstream_name = name[len(settings.k8s_tool_prefix):]
        result = await self._session.call_tool(downstream_name, arguments)
        return result.content
