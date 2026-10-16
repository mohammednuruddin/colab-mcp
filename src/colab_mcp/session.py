# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
from collections.abc import AsyncIterator
import contextlib
from contextlib import AsyncExitStack
from fastmcp import FastMCP, Client
from fastmcp.client.transports import ClientTransport
from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.tool_injection import ToolInjectionMiddleware
from fastmcp.server.proxy import FastMCPProxy
from fastmcp.tools.tool import Tool, ToolResult
from mcp.client.session import ClientSession
from mcp.types import TextContent
import logging
import os
import webbrowser

from colab_mcp.notebook_ops import NOTEBOOK_TOOLS, PASSIVE_TOOL_NAMES
from colab_mcp.notebooks import (
    UnsupportedNotebookUrl,
    connection_url,
    normalize_notebook_url,
    scratch_notebook_url,
)
from colab_mcp.state import (
    ConnectionState,
    default_state_dir,
    load_state,
    new_token,
    save_state,
)
from colab_mcp.websocket_server import ColabWebSocketServer
from fastmcp.exceptions import ToolError
from pathlib import Path

# How long open_colab_browser_connection waits for the user to accept the
# connection in the Colab tab. Override with COLAB_MCP_UI_TIMEOUT (seconds).
UI_CONNECTION_TIMEOUT = float(os.environ.get("COLAB_MCP_UI_TIMEOUT", "60"))  # secs

FE_CONNECTED_KEY = "fe_connected"
PROXY_TOKEN_KEY = "proxy_token"
PROXY_PORT_KEY = "proxy_port"
PROXY_CLIENT_KEY = "proxy_client"
INJECTED_TOOL_NAME = "open_colab_browser_connection"
STATUS_TOOL_NAME = "colab_connection_status"
INJECTED_TOOL_NAMES = (
    frozenset({INJECTED_TOOL_NAME, STATUS_TOOL_NAME}) | PASSIVE_TOOL_NAMES
)

# When a notebook tool is called and the tab has dropped, re-open the last
# notebook and wait for it instead of failing. COLAB_MCP_AUTO_REOPEN=0 disables.
AUTO_REOPEN = os.environ.get("COLAB_MCP_AUTO_REOPEN", "1") not in ("0", "false", "no")

# Filled in by main() so the status tool can point people at the logs.
SERVER_INFO: dict = {"log_dir": None, "state_dir": None, "ephemeral": True}


class ColabTransport(ClientTransport):
    def __init__(self, wss: ColabWebSocketServer):
        self.wss = wss

    @contextlib.asynccontextmanager
    async def connect_session(self, **session_kwargs) -> AsyncIterator[ClientSession]:
        async with ClientSession(
            self.wss.read_stream, self.wss.write_stream, **session_kwargs
        ) as session:
            yield session

    def __repr__(self) -> str:
        return "<ColabSessionProxyTransport>"


class ColabProxyClient:
    def __init__(self, wss: ColabWebSocketServer):
        self.wss = wss
        self.stubbed_mcp_client = Client(FastMCP())
        self.proxy_mcp_client: Client | None = None
        self._exit_stack = AsyncExitStack()
        self._start_task = None
        # The notebook we last asked the browser to open, and the one the live
        # connection was established for. Best effort: the tab does not tell
        # us which notebook it is, so this is the URL we sent it to.
        self.requested_notebook_url: str | None = None
        self.notebook_url: str | None = None
        self.on_connected = None
        self._recycle_task: asyncio.Task | None = None
        self._recycle_lock = asyncio.Lock()

    def __deepcopy__(self, memo):
        # This object is stored in the request context state so the injected
        # tools can reach it. FastMCP deep-copies that state into nested
        # contexts (e.g. when listing the proxied notebook tools), and the live
        # websocket server and asyncio tasks held here cannot be copied: the
        # copy would fail and FastMCP would silently drop the Colab tools.
        # A connection handle has no meaningful copy, so share the instance.
        return self

    def is_connected(self):
        return self.wss.connection_live.is_set() and self.proxy_mcp_client is not None

    def mark_connected(self):
        """Record that the pending browser connection is now live."""
        self.notebook_url = self.requested_notebook_url
        if self.on_connected is not None:
            self.on_connected(self.notebook_url)

    def session_is_dead(self) -> bool:
        """True when the MCP client task ended without a live connection.

        That happens after a cancelled or failed startup: the ClientSession
        closed the websocket server's streams on the way out, so any tab that
        connects now is dropped immediately. The fix is a recycle.
        """
        task = self._start_task
        return task is not None and task.done() and not self.is_connected()

    async def ensure_ready(self):
        """Make sure a fresh MCP session is waiting before a tab is opened."""
        if self.session_is_dead():
            logging.warning("MCP client session had ended; recycling before reconnect")
            await self._recycle()

    def handle_disconnect(self):
        """Websocket callback: the tab dropped, so recycle the MCP session.

        The ClientSession bound to the old tab closes its streams on the way
        out; a fresh session (and fresh streams) must be waiting before the
        next tab, or an auto re-open, can connect.
        """
        if self._recycle_task is None or self._recycle_task.done():
            self._recycle_task = asyncio.create_task(self._recycle())

    async def _recycle(self):
        async with self._recycle_lock:
            if self._start_task:
                self._start_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._start_task
            with contextlib.suppress(Exception):
                await self._exit_stack.aclose()
            self._exit_stack = AsyncExitStack()
            self.proxy_mcp_client = None
            self.notebook_url = None
            self.wss.reset_streams()
            self._start_task = asyncio.create_task(self._start_proxy_client())

    async def restart(self):
        """Drop the current Colab tab and get ready for a new one.

        Used to switch notebooks: the old tab's socket is closed (which
        triggers handle_disconnect through the websocket server when wired),
        then we make sure a fresh session is waiting for the next tab.
        """
        await self.wss.disconnect()
        if self._recycle_task is not None:
            with contextlib.suppress(Exception):
                await self._recycle_task
        if (
            self.proxy_mcp_client is not None
            or self._start_task is None
            or self._start_task.done()
        ):
            await self._recycle()

    async def await_proxy_connection(self):
        """Wait for a tab to connect and the MCP client to finish initialising.

        Never wraps the long-lived client task in gather/wait_for: those
        cancel their children on timeout or on cancellation of the tool call,
        which killed the session and left every later tab dropping instantly.
        """
        deadline = asyncio.get_running_loop().time() + UI_CONNECTION_TIMEOUT

        def remaining() -> float:
            return max(0.0, deadline - asyncio.get_running_loop().time())

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.wss.connection_live.wait(), timeout=remaining())
        task = self._start_task
        if task is None or task.done():
            return
        # asyncio.wait() observes the task without cancelling it.
        await asyncio.wait({task}, timeout=remaining())

    def client_factory(self):
        if self.is_connected():
            return self.proxy_mcp_client
        # return a client mapped to a stubbed mcp server if there is no session proxy
        return self.stubbed_mcp_client

    async def _start_proxy_client(self):
        # blocks until a websocket connection is made successfully
        self.proxy_mcp_client = await self._exit_stack.enter_async_context(
            Client(ColabTransport(self.wss))
        )

    async def __aenter__(self):
        self._start_task = asyncio.create_task(self._start_proxy_client())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._start_task:
            self._start_task.cancel()
        await self._exit_stack.aclose()


class ColabProxyMiddleware(Middleware):
    def __init__(self, proxy_client: ColabProxyClient):
        self.proxy_client = proxy_client
        self.last_message_connected = self.proxy_client.is_connected()

    async def on_message(self, context: MiddlewareContext, call_next):
        """
        Check for a change to Colab session connectivity on any communication with this MCP server and
        notify the client when the connectivity status has changed.
        """
        context.fastmcp_context.set_state(
            FE_CONNECTED_KEY, self.proxy_client.is_connected()
        )
        context.fastmcp_context.set_state(PROXY_TOKEN_KEY, self.proxy_client.wss.token)
        context.fastmcp_context.set_state(PROXY_PORT_KEY, self.proxy_client.wss.port)
        context.fastmcp_context.set_state(PROXY_CLIENT_KEY, self.proxy_client)

        result = await call_next(context)

        connected = self.proxy_client.is_connected()
        connection_state_changed = connected != self.last_message_connected
        self.last_message_connected = connected
        if connection_state_changed:
            await context.fastmcp_context.send_tool_list_changed()

        return result

    async def _reopen(self, context, tool_name: str):
        """The tab dropped mid-session: re-open the last notebook and wait for it."""
        notebook = self.proxy_client.requested_notebook_url
        url = connection_url(
            notebook, self.proxy_client.wss.token, self.proxy_client.wss.port
        )
        logging.info("Tab disconnected before %s; re-opening %s", tool_name, notebook)
        await context.fastmcp_context.report_progress(
            progress=1,
            total=2,
            message=f"Colab tab is not connected; re-opening {notebook} and waiting up to {UI_CONNECTION_TIMEOUT:.0f}s",
        )
        await self.proxy_client.ensure_ready()
        webbrowser.open_new(url)
        await self.proxy_client.await_proxy_connection()
        if not self.proxy_client.is_connected():
            raise ToolError(
                f"The Colab tab for {notebook} is not connected and did not reconnect within "
                f"{UI_CONNECTION_TIMEOUT:.0f}s. Ask the user to accept the MCP connection in the tab "
                f"(or call {INJECTED_TOOL_NAME}), then retry {tool_name}."
            )
        self.proxy_client.mark_connected()
        await context.fastmcp_context.report_progress(
            progress=2, total=2, message="Colab tab reconnected"
        )

    async def on_call_tool(self, context, call_next):
        name = context.message.name
        if (
            name not in INJECTED_TOOL_NAMES
            and AUTO_REOPEN
            and not self.proxy_client.is_connected()
            and self.proxy_client.requested_notebook_url
        ):
            await self._reopen(context, name)
        if name in INJECTED_TOOL_NAMES:
            result = await call_next(context)
        else:
            # A proxied call whose tab drops mid-flight would otherwise wait
            # forever for a response that can never arrive.
            result = await until_disconnect(
                self.proxy_client.wss,
                call_next(context),
                f"Colab tab disconnected during {name}",
            )
        if name != INJECTED_TOOL_NAME:
            return result
        if self.proxy_client.is_connected():
            return result
        # if the tool call was for open_colab_browser_connection and there is no existing connection, try to await full connection
        notebook = self.proxy_client.requested_notebook_url or scratch_notebook_url()
        await context.fastmcp_context.report_progress(
            progress=1, total=3, message="The user is not connected to the Colab UI"
        )
        await context.fastmcp_context.report_progress(
            progress=2,
            total=3,
            message=f"Waiting for user to connect in Colab - will wait for {UI_CONNECTION_TIMEOUT:.0f}s",
        )
        await self.proxy_client.await_proxy_connection()
        if self.proxy_client.is_connected():
            self.proxy_client.mark_connected()
            await context.fastmcp_context.report_progress(
                progress=3, total=3, message="The Colab UI is successfully connected!"
            )
            return _connection_result(
                True,
                notebook,
                "Connected. Call tools/list again: the notebook tools are now available.",
            )
        else:
            await context.fastmcp_context.report_progress(
                progress=3,
                total=3,
                message="Timeout while waiting for the user to connect.",
            )
            return _connection_result(
                False,
                notebook,
                f"The Colab tab did not connect within {UI_CONNECTION_TIMEOUT:.0f}s. "
                "Ask the user to open the notebook tab and accept the MCP connection, then call this tool again.",
            )


async def until_disconnect(wss, awaitable, message: str):
    """Await ``awaitable`` but fail with ToolError as soon as the tab drops."""
    if wss.disconnected.is_set():
        raise ToolError(message.replace("during", "before"))
    call = asyncio.ensure_future(awaitable)
    drop = asyncio.ensure_future(wss.disconnected.wait())
    try:
        done, _ = await asyncio.wait({call, drop}, return_when=asyncio.FIRST_COMPLETED)
        if call in done:
            return call.result()
        call.cancel()
        with contextlib.suppress(BaseException):
            await call
        raise ToolError(message)
    finally:
        if not drop.done():
            drop.cancel()
        if not call.done():
            call.cancel()


def _connection_result(connected: bool, notebook_url: str, message: str) -> ToolResult:
    payload = {
        "result": connected,
        "connected": connected,
        "notebook_url": notebook_url,
        "message": message,
    }
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
    )


async def check_session_proxy_tool_fn(
    notebook_url: str | None = None, ctx: Context = CurrentContext()
) -> dict:
    """Open (or switch) the Colab browser connection.

    Without ``notebook_url`` this opens Colab's scratch notebook. With one, the
    link is normalised to a Colab URL and that notebook is opened instead. If a
    different notebook is already connected, the old tab is disconnected first.
    """
    fe_connected = ctx.get_state(FE_CONNECTED_KEY)
    token = ctx.get_state(PROXY_TOKEN_KEY)
    port = ctx.get_state(PROXY_PORT_KEY)
    client: ColabProxyClient | None = ctx.get_state(PROXY_CLIENT_KEY)

    if notebook_url:
        try:
            target = normalize_notebook_url(notebook_url)
        except UnsupportedNotebookUrl as exc:
            raise ValueError(str(exc)) from exc
    else:
        target = None

    current = client.notebook_url if client else None
    if fe_connected and (target is None or target == current):
        return {
            "result": True,
            "connected": True,
            "notebook_url": current or scratch_notebook_url(),
            "message": "Already connected. The notebook tools are available via tools/list.",
        }

    if fe_connected and client is not None:
        logging.info("Switching Colab connection from %s to %s", current, target)
        await client.restart()

    target = target or scratch_notebook_url()
    if client is not None:
        await client.ensure_ready()
        client.requested_notebook_url = target
    url = connection_url(target, token, port)
    logging.info("Opening Colab notebook %s", target)
    webbrowser.open_new(url)
    return {
        "result": False,
        "connected": False,
        "notebook_url": target,
        "open_url": url,
        "message": "Opened the notebook in the browser; waiting for the user to accept the connection.",
    }


async def connection_status_tool_fn(ctx: Context = CurrentContext()) -> dict:
    client: ColabProxyClient | None = ctx.get_state(PROXY_CLIENT_KEY)
    connected = bool(ctx.get_state(FE_CONNECTED_KEY))
    return {
        "connected": connected,
        "notebook_url": (client.notebook_url if client and connected else None),
        "requested_notebook_url": client.requested_notebook_url if client else None,
        "session_dead": client.session_is_dead() if client else None,
        "proxy_port": ctx.get_state(PROXY_PORT_KEY),
        "auto_reopen": AUTO_REOPEN,
        "connection_wait_seconds": UI_CONNECTION_TIMEOUT,
        "persistent_identity": not SERVER_INFO.get("ephemeral", True),
        "state_dir": SERVER_INFO.get("state_dir"),
        "log_dir": SERVER_INFO.get("log_dir"),
        "server_pid": os.getpid(),
    }


check_session_proxy_tool = Tool.from_function(
    fn=check_session_proxy_tool_fn,
    name=INJECTED_TOOL_NAME,
    description=(
        "Opens a connection to a Google Colab browser session and unlocks notebook editing tools. "
        "Pass notebook_url to open a specific notebook instead of the scratch notebook: Colab links, "
        "Google Drive share links or file ids, GitHub .ipynb blob/raw links, and gist links are accepted. "
        "Calling it with a different notebook_url while connected switches to that notebook. "
        "The call waits for the user to accept the connection in the browser tab; on success, "
        "list tools again to see the notebook tools."
    ),
)

connection_status_tool = Tool.from_function(
    fn=connection_status_tool_fn,
    name=STATUS_TOOL_NAME,
    description="Reports whether a Colab browser session is connected and which notebook it was opened for. Does not open a browser.",
)


class ColabSessionProxy:
    def __init__(
        self,
        state_dir: Path | None = None,
        ephemeral: bool = False,
        port: int | None = None,
    ):
        self._exit_stack = AsyncExitStack()
        self.proxy_server: FastMCPProxy | None = None
        # list order matters, see: https://gofastmcp.com/servers/middleware#multiple-middleware
        self.middleware: list[Middleware] = []
        self.wss: ColabWebSocketServer | None = None
        self.ephemeral = ephemeral
        self.state_dir = None if ephemeral else (state_dir or default_state_dir())
        self.port = port
        self.state: ConnectionState | None = None

    def _persist(self, notebook_url: str | None = None):
        if self.ephemeral or self.wss is None or self.state_dir is None:
            return
        previous = self.state.last_notebook_url if self.state else None
        self.state = ConnectionState(
            token=self.wss.token,
            port=self.wss.port,
            last_notebook_url=notebook_url or previous,
        )
        try:
            save_state(self.state_dir, self.state)
        except OSError as exc:
            logging.warning(
                "Could not persist connection state to %s: %s", self.state_dir, exc
            )

    async def start_proxy_server(self):
        saved = None if self.ephemeral else load_state(self.state_dir)
        token = saved.token if saved else new_token()
        port = self.port if self.port is not None else (saved.port if saved else 0)
        self.wss = await self._exit_stack.enter_async_context(
            ColabWebSocketServer(port=port, token=token)
        )
        if saved and port and self.wss.port != port:
            logging.info("Persisted port %s was taken; now on %s", port, self.wss.port)
        self._persist()
        proxy_client = await self._exit_stack.enter_async_context(
            ColabProxyClient(self.wss)
        )
        self.wss.on_disconnect = proxy_client.handle_disconnect
        proxy_client.on_connected = self._persist
        if saved and saved.last_notebook_url:
            # Lets auto re-open bring back the last notebook after a restart.
            proxy_client.requested_notebook_url = saved.last_notebook_url
        SERVER_INFO["state_dir"] = str(self.state_dir) if self.state_dir else None
        SERVER_INFO["ephemeral"] = self.ephemeral
        self.proxy_server = FastMCPProxy(
            client_factory=proxy_client.client_factory,
            instructions="Connects to a user's Google Colab session in a browser and allows for interactions with their Google Colab notebook",
        )
        # ColabProxyMiddleware must be first because it sets the fe_connected state
        self.middleware.append(ColabProxyMiddleware(proxy_client))
        self.middleware.append(
            ToolInjectionMiddleware(
                tools=[
                    check_session_proxy_tool,
                    connection_status_tool,
                    *NOTEBOOK_TOOLS,
                ]
            )
        )

    async def cleanup(self):
        await self._exit_stack.aclose()
