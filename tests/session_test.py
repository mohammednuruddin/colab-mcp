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
import copy
import time
from colab_mcp import session
from colab_mcp.state import ConnectionState, load_state, save_state
from fastmcp.server.middleware import MiddlewareContext
import pytest
from unittest.mock import patch, AsyncMock, Mock


@pytest.fixture(autouse=True)
def mock_webbrowser(monkeypatch):
    import webbrowser

    mock_open = Mock()
    monkeypatch.setattr(webbrowser, "open_new", mock_open)
    return mock_open


@pytest.fixture
def mock_wss():
    """Provides a mock ColabWebSocketServer instance."""
    return MockColabWebSocketServer()


class MockColabWebSocketServer:
    def __init__(self):
        self.connection_live = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.read_stream = AsyncMock()
        self.write_stream = AsyncMock()
        self.token = "test-token"
        self.port = 1234

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_proxy_client(mock_wss):
    client = Mock(spec=session.ColabProxyClient)
    client.wss = mock_wss
    client.is_connected.return_value = False
    client.requested_notebook_url = None
    client.notebook_url = None
    client.previous_notebook_url = None
    return client


class TestColabProxyMiddleware:
    @pytest.mark.asyncio
    async def test_connection_live(self, mock_proxy_client):
        """Tests connection state change from disconnected to connected."""
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        mock_proxy_client.is_connected.return_value = True
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.set_state = Mock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        context.fastmcp_context.set_state.assert_any_call("fe_connected", True)
        context.fastmcp_context.set_state.assert_any_call("proxy_token", "test-token")
        context.fastmcp_context.set_state.assert_any_call("proxy_port", 1234)
        assert middleware.last_message_connected is True
        context.fastmcp_context.send_tool_list_changed.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_not_live(self, mock_proxy_client):
        """Tests connection state change from connected to disconnected."""
        mock_proxy_client.is_connected.return_value = True
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        mock_proxy_client.is_connected.return_value = False
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.set_state = Mock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        context.fastmcp_context.set_state.assert_any_call("fe_connected", False)
        assert middleware.last_message_connected is False
        context.fastmcp_context.send_tool_list_changed.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_connection_change(self, mock_proxy_client):
        """Tests no connection state change."""
        mock_proxy_client.is_connected.return_value = True
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.set_state = Mock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        context.fastmcp_context.set_state.assert_any_call("fe_connected", True)
        assert middleware.last_message_connected is True
        context.fastmcp_context.send_tool_list_changed.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_call_tool_await_connection(self, mock_proxy_client):
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        mock_proxy_client.is_connected.side_effect = [False, True]
        mock_proxy_client.await_proxy_connection = AsyncMock()
        call_next = AsyncMock()

        result = await middleware.on_call_tool(context, call_next)

        mock_proxy_client.await_proxy_connection.assert_called_once()
        mock_proxy_client.mark_connected.assert_called_once()
        context.fastmcp_context.report_progress.assert_called()
        assert result.structured_content["result"] is True
        assert result.structured_content["connected"] is True
        assert (
            result.structured_content["notebook_url"] == session.scratch_notebook_url()
        )

    @pytest.mark.asyncio
    async def test_on_call_tool_timeout(self, mock_proxy_client):
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        mock_proxy_client.is_connected.return_value = False
        mock_proxy_client.await_proxy_connection = AsyncMock()
        call_next = AsyncMock()

        result = await middleware.on_call_tool(context, call_next)

        mock_proxy_client.await_proxy_connection.assert_called_once()
        assert result.structured_content["result"] is False
        assert result.structured_content["connected"] is False


def _ctx(connected, client=None):
    ctx = Mock()

    def get_state(k):
        if k == session.FE_CONNECTED_KEY:
            return connected
        if k == session.PROXY_TOKEN_KEY:
            return "test-token"
        if k == session.PROXY_PORT_KEY:
            return 1234
        if k == session.PROXY_CLIENT_KEY:
            return client
        return None

    ctx.get_state.side_effect = get_state
    return ctx


class TestCheckSessionProxyToolFn:
    @pytest.mark.asyncio
    async def test_connected(self):
        ctx = _ctx(True)
        result = await session.check_session_proxy_tool_fn(ctx=ctx)
        assert result["result"] is True
        assert result["connected"] is True

    @pytest.mark.asyncio
    async def test_disconnected(self, mock_webbrowser, mock_proxy_client):
        ctx = _ctx(False, mock_proxy_client)
        result = await session.check_session_proxy_tool_fn(ctx=ctx)
        assert result["result"] is False
        assert result["notebook_url"] == session.scratch_notebook_url()
        mock_webbrowser.assert_called_once()
        args, _ = mock_webbrowser.call_args
        assert args[0].startswith(session.scratch_notebook_url() + "#")
        assert "mcpProxyToken=test-token" in args[0]
        assert "mcpProxyPort=1234" in args[0]
        assert result["open_url"] == args[0]
        assert (
            mock_proxy_client.requested_notebook_url == session.scratch_notebook_url()
        )

    @pytest.mark.asyncio
    async def test_opens_requested_notebook(self, mock_webbrowser, mock_proxy_client):
        ctx = _ctx(False, mock_proxy_client)
        result = await session.check_session_proxy_tool_fn(
            "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing",
            ctx=ctx,
        )
        expected = "https://colab.research.google.com/drive/1AbCdEfGhIjKlMnOp"
        assert result["notebook_url"] == expected
        args, _ = mock_webbrowser.call_args
        assert args[0] == f"{expected}#mcpProxyToken=test-token&mcpProxyPort=1234"
        mock_proxy_client.restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_same_notebook_already_connected_is_noop(
        self, mock_webbrowser, mock_proxy_client
    ):
        mock_proxy_client.notebook_url = (
            "https://colab.research.google.com/drive/abc123def456"
        )
        ctx = _ctx(True, mock_proxy_client)
        result = await session.check_session_proxy_tool_fn(
            "https://colab.research.google.com/drive/abc123def456#scrollTo=x", ctx=ctx
        )
        assert result["connected"] is True
        mock_webbrowser.assert_not_called()
        mock_proxy_client.restart.assert_not_called()

    @pytest.mark.asyncio
    async def test_switches_notebook_when_connected_elsewhere(
        self, mock_webbrowser, mock_proxy_client
    ):
        mock_proxy_client.notebook_url = session.scratch_notebook_url()
        mock_proxy_client.restart = AsyncMock()
        ctx = _ctx(True, mock_proxy_client)
        result = await session.check_session_proxy_tool_fn(
            "https://github.com/org/repo/blob/main/nb/train.ipynb", ctx=ctx
        )
        expected = (
            "https://colab.research.google.com/github/org/repo/blob/main/nb/train.ipynb"
        )
        mock_proxy_client.restart.assert_awaited_once()
        assert result["connected"] is False
        assert result["notebook_url"] == expected
        assert mock_proxy_client.requested_notebook_url == expected
        mock_webbrowser.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejects_unsupported_link(self, mock_webbrowser, mock_proxy_client):
        ctx = _ctx(False, mock_proxy_client)
        with pytest.raises(ValueError, match="Cannot open"):
            await session.check_session_proxy_tool_fn(
                "https://example.com/nb.ipynb", ctx=ctx
            )
        mock_webbrowser.assert_not_called()


class TestConnectionStatusTool:
    @pytest.mark.asyncio
    async def test_status(self, mock_proxy_client):
        mock_proxy_client.notebook_url = "https://colab.research.google.com/drive/abc"
        mock_proxy_client.requested_notebook_url = (
            "https://colab.research.google.com/drive/abc"
        )
        mock_proxy_client.session_is_dead.return_value = False
        status = await session.connection_status_tool_fn(
            ctx=_ctx(True, mock_proxy_client)
        )
        assert status["connected"] is True
        assert status["notebook_url"] == "https://colab.research.google.com/drive/abc"
        assert (
            status["requested_notebook_url"]
            == "https://colab.research.google.com/drive/abc"
        )
        assert status["session_dead"] is False
        assert status["previous_notebook_url"] is None
        assert status["proxy_port"] == 1234
        assert status["server_pid"] > 0
        status = await session.connection_status_tool_fn(
            ctx=_ctx(False, mock_proxy_client)
        )
        assert status["connected"] is False
        assert status["notebook_url"] is None


class TestColabProxyClient:
    def test_is_connected(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        assert client.is_connected() is False
        mock_wss.connection_live.set()
        assert client.is_connected() is False
        client.proxy_mcp_client = Mock()
        assert client.is_connected() is True

    def test_client_factory_connection_live(self, mock_wss):
        mock_wss.connection_live.set()
        client = session.ColabProxyClient(mock_wss)
        client.proxy_mcp_client = Mock()

        assert client.client_factory() is client.proxy_mcp_client

    def test_client_factory_connection_not_live(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        assert client.client_factory() is client.stubbed_mcp_client

    @pytest.mark.asyncio
    async def test_await_proxy_connection(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        client._start_task = asyncio.create_task(asyncio.sleep(0.01))
        mock_wss.connection_live.set()
        with patch("colab_mcp.session.UI_CONNECTION_TIMEOUT", 0.1):
            await client.await_proxy_connection()
        await client._start_task

    @pytest.mark.asyncio
    @patch("colab_mcp.session.Client")
    @patch("colab_mcp.session.ColabTransport", spec=session.ColabTransport)
    async def test_start_proxy_client(
        self, mock_colab_transport, mock_client, mock_wss
    ):
        mock_client.return_value.__aenter__ = AsyncMock()
        client = session.ColabProxyClient(mock_wss)
        mock_wss.connection_live.set()
        async with client:
            await client._start_task

        mock_colab_transport.assert_called_once_with(mock_wss)
        mock_client.assert_called_with(mock_colab_transport.return_value)


class TestColabTransport:
    @pytest.mark.asyncio
    @patch("colab_mcp.session.ClientSession")
    async def test_connect_session(self, mock_client_session, mock_wss):
        transport = session.ColabTransport(mock_wss)
        mock_client_session.return_value.__aenter__ = AsyncMock()
        async with transport.connect_session(foo="bar") as client_session:
            assert (
                client_session
                == mock_client_session.return_value.__aenter__.return_value
            )

        mock_client_session.assert_called_once_with(
            mock_wss.read_stream, mock_wss.write_stream, foo="bar"
        )


class TestColabSessionProxy:
    @pytest.mark.asyncio
    @patch("colab_mcp.session.ToolInjectionMiddleware")
    @patch("colab_mcp.session.ColabWebSocketServer")
    @patch("colab_mcp.session.ColabProxyClient")
    @patch("colab_mcp.session.ColabProxyMiddleware")
    async def test_start_proxy_server(
        self,
        mock_colab_proxy_middleware,
        mock_colab_proxy_client,
        mock_colab_web_socket_server,
        mock_tool_injection_middleware,
    ):
        mock_colab_web_socket_server.return_value.__aenter__ = AsyncMock()
        mock_colab_proxy_client.return_value.__aenter__ = AsyncMock()
        proxy = session.ColabSessionProxy(ephemeral=True)
        await proxy.start_proxy_server()
        mock_colab_proxy_client.assert_called_once()
        assert proxy.proxy_server is not None
        mock_colab_proxy_middleware.assert_called_once()
        mock_tool_injection_middleware.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        proxy = session.ColabSessionProxy()
        proxy._exit_stack = AsyncMock()
        await proxy.cleanup()
        proxy._exit_stack.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_proxy_client_survives_context_state_deepcopy(mock_wss):
    """FastMCP deep-copies context state into nested contexts; the live client must survive it."""
    client = session.ColabProxyClient(mock_wss)
    client._start_task = asyncio.create_task(asyncio.sleep(10))
    try:
        state = {session.PROXY_CLIENT_KEY: client, session.FE_CONNECTED_KEY: True}
        copied = copy.deepcopy(state)
        assert copied[session.PROXY_CLIENT_KEY] is client
        assert copied[session.FE_CONNECTED_KEY] is True
    finally:
        client._start_task.cancel()


@pytest.mark.asyncio
async def test_mounted_proxy_lists_notebook_tools_when_connected(mock_wss):
    """Mirror main(): the parent server mounts the proxy and runs the middleware.

    Listing tools through the parent deep-copies the request context state,
    which holds the live ColabProxyClient. The notebook tools must survive
    that copy instead of being silently dropped by FastMCP.
    """
    from contextlib import AsyncExitStack
    from fastmcp import FastMCP, Client
    from fastmcp.server.proxy import FastMCPProxy
    from fastmcp.server.middleware.tool_injection import ToolInjectionMiddleware

    notebook = FastMCP("colab-tab")

    @notebook.tool
    def add_code_cell(source: str) -> str:
        return source

    async with AsyncExitStack() as stack:
        client = session.ColabProxyClient(mock_wss)
        client.proxy_mcp_client = await stack.enter_async_context(Client(notebook))
        client._start_task = asyncio.create_task(asyncio.sleep(10))
        stack.callback(client._start_task.cancel)
        mock_wss.connection_live.set()
        assert client.is_connected()

        parent = FastMCP("ColabMCP")
        parent.mount(FastMCPProxy(client_factory=client.client_factory))
        parent.add_middleware(session.ColabProxyMiddleware(client))
        parent.add_middleware(
            ToolInjectionMiddleware(
                tools=[session.check_session_proxy_tool, session.connection_status_tool]
            )
        )

        async with Client(parent) as c:
            names = {tool.name for tool in await c.list_tools()}

    assert "add_code_cell" in names
    assert session.INJECTED_TOOL_NAME in names
    assert session.STATUS_TOOL_NAME in names


class TestProxyClientLifecycle:
    @pytest.mark.asyncio
    async def test_await_connection_timeout_does_not_cancel_start_task(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        client._start_task = asyncio.create_task(asyncio.sleep(10))
        with patch("colab_mcp.session.UI_CONNECTION_TIMEOUT", 0.05):
            await client.await_proxy_connection()
        assert not client._start_task.cancelled()
        assert not client._start_task.done()
        client._start_task.cancel()

    @pytest.mark.asyncio
    async def test_await_connection_survives_cancellation_of_caller(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        client._start_task = asyncio.create_task(asyncio.sleep(10))
        with patch("colab_mcp.session.UI_CONNECTION_TIMEOUT", 5):
            waiter = asyncio.create_task(client.await_proxy_connection())
            await asyncio.sleep(0.01)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert not client._start_task.cancelled()
        client._start_task.cancel()

    @pytest.mark.asyncio
    async def test_ensure_ready_recycles_dead_session(self, mock_wss):
        mock_wss.reset_streams = Mock()
        client = session.ColabProxyClient(mock_wss)

        async def died():
            raise RuntimeError("init cancelled")

        client._start_task = asyncio.create_task(died())
        await asyncio.sleep(0)
        assert client.session_is_dead() is True
        waiting = asyncio.Event()

        async def start_and_wait():
            await waiting.wait()

        with patch.object(client, "_start_proxy_client", start_and_wait):
            await client.ensure_ready()
            await asyncio.sleep(0)
            mock_wss.reset_streams.assert_called_once()
            assert client.session_is_dead() is False
            client._start_task.cancel()

    @pytest.mark.asyncio
    async def test_handle_disconnect_schedules_single_recycle(self, mock_wss):
        mock_wss.reset_streams = Mock()
        client = session.ColabProxyClient(mock_wss)
        with patch.object(client, "_start_proxy_client", AsyncMock()):
            client.handle_disconnect()
            first = client._recycle_task
            client.handle_disconnect()
            assert client._recycle_task is first
            await first
        mock_wss.reset_streams.assert_called_once()


class TestAutoReopen:
    @pytest.mark.asyncio
    async def test_reopens_last_notebook_for_proxied_tool(
        self, mock_proxy_client, mock_webbrowser
    ):
        mock_proxy_client.requested_notebook_url = (
            "https://colab.research.google.com/drive/abc"
        )
        # constructor, on_call_tool guard, then the post-wait check
        mock_proxy_client.is_connected.side_effect = [False, False, True]
        mock_proxy_client.await_proxy_connection = AsyncMock()
        mock_proxy_client.ensure_ready = AsyncMock()
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.message.name = "run_cell"
        context.fastmcp_context.report_progress = AsyncMock()
        call_next = AsyncMock(return_value="ran")

        with patch("colab_mcp.session.AUTO_REOPEN", True):
            assert await middleware.on_call_tool(context, call_next) == "ran"
        mock_webbrowser.assert_called_once()
        assert mock_webbrowser.call_args[0][0].startswith(
            "https://colab.research.google.com/drive/abc#"
        )
        mock_proxy_client.mark_connected.assert_called_once()
        call_next.assert_awaited_once_with(context)

    @pytest.mark.asyncio
    async def test_reopen_failure_raises_tool_error(
        self, mock_proxy_client, mock_webbrowser
    ):
        from fastmcp.exceptions import ToolError

        mock_proxy_client.requested_notebook_url = (
            "https://colab.research.google.com/drive/abc"
        )
        mock_proxy_client.is_connected.return_value = False
        mock_proxy_client.await_proxy_connection = AsyncMock()
        mock_proxy_client.ensure_ready = AsyncMock()
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.message.name = "run_cell"
        context.fastmcp_context.report_progress = AsyncMock()
        call_next = AsyncMock()
        with patch("colab_mcp.session.AUTO_REOPEN", True):
            with pytest.raises(ToolError, match="did not reconnect"):
                await middleware.on_call_tool(context, call_next)
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_notebook_from_previous_process_is_not_reopened(
        self, mock_proxy_client, mock_webbrowser
    ):
        mock_proxy_client.previous_notebook_url = (
            "https://colab.research.google.com/drive/old"
        )
        mock_proxy_client.is_connected.return_value = False
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.message.name = "run_code"
        call_next = AsyncMock(return_value="ran")

        async def passthrough(_wss, call, _message):
            return await call

        with (
            patch("colab_mcp.session.AUTO_REOPEN", True),
            patch("colab_mcp.session.until_disconnect", passthrough),
        ):
            assert await middleware.on_call_tool(context, call_next) == "ran"
        mock_webbrowser.assert_not_called()

    @pytest.mark.asyncio
    async def test_injected_tools_never_trigger_reopen(
        self, mock_proxy_client, mock_webbrowser
    ):
        mock_proxy_client.requested_notebook_url = (
            "https://colab.research.google.com/drive/abc"
        )
        mock_proxy_client.is_connected.return_value = False
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.message.name = session.STATUS_TOOL_NAME
        call_next = AsyncMock(return_value="status")
        assert await middleware.on_call_tool(context, call_next) == "status"
        mock_webbrowser.assert_not_called()


class TestPersistedIdentity:
    @pytest.mark.asyncio
    async def test_token_and_port_survive_restart(self, tmp_path):
        first = session.ColabSessionProxy(state_dir=tmp_path)
        await first.start_proxy_server()
        token, port = first.wss.token, first.wss.port
        assert port > 0
        first._persist("https://colab.research.google.com/drive/abc")
        await first.cleanup()

        second = session.ColabSessionProxy(state_dir=tmp_path)
        await second.start_proxy_server()
        try:
            assert second.wss.token == token
            assert second.wss.port == port
            client = second.middleware[0].proxy_client
            # Reported, but not a target for auto re-open after a restart.
            assert (
                client.previous_notebook_url
                == "https://colab.research.google.com/drive/abc"
            )
            assert client.requested_notebook_url is None
            assert (tmp_path / "connection.json").stat().st_mode & 0o777 == 0o600
        finally:
            await second.cleanup()

    @pytest.mark.asyncio
    async def test_rotate_token_keeps_port_and_last_notebook(self, tmp_path):
        first = session.ColabSessionProxy(state_dir=tmp_path)
        await first.start_proxy_server()
        token, port = first.wss.token, first.wss.port
        first._persist("https://colab.research.google.com/drive/abc")
        await first.cleanup()

        second = session.ColabSessionProxy(state_dir=tmp_path, rotate_token=True)
        await second.start_proxy_server()
        try:
            assert second.wss.token != token
            assert second.wss.port == port
            saved = load_state(tmp_path)
            assert saved.token == second.wss.token
            assert (
                saved.last_notebook_url == "https://colab.research.google.com/drive/abc"
            )
        finally:
            await second.cleanup()

    @pytest.mark.asyncio
    async def test_expired_token_is_replaced(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COLAB_MCP_TOKEN_MAX_AGE_DAYS", "30")
        save_state(
            tmp_path, ConnectionState(token="old-token", port=1, token_created=0.0)
        )
        proxy = session.ColabSessionProxy(state_dir=tmp_path, port=0)
        await proxy.start_proxy_server()
        try:
            assert proxy.wss.token != "old-token"
            assert load_state(tmp_path).token_created > 0
        finally:
            await proxy.cleanup()

    @pytest.mark.asyncio
    async def test_unexpired_token_keeps_its_age(self, tmp_path):
        created = time.time() - 86400
        save_state(
            tmp_path, ConnectionState(token="tok", port=1, token_created=created)
        )
        proxy = session.ColabSessionProxy(state_dir=tmp_path, port=0)
        await proxy.start_proxy_server()
        try:
            assert proxy.wss.token == "tok"
            assert load_state(tmp_path).token_created == created
        finally:
            await proxy.cleanup()

    @pytest.mark.asyncio
    async def test_ephemeral_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COLAB_MCP_STATE_DIR", str(tmp_path))
        proxy = session.ColabSessionProxy(ephemeral=True)
        await proxy.start_proxy_server()
        await proxy.cleanup()
        assert not (tmp_path / "connection.json").exists()


class TestUntilDisconnect:
    @pytest.mark.asyncio
    async def test_returns_result_when_call_finishes(self):
        wss = Mock()
        wss.disconnected = asyncio.Event()

        async def call():
            await asyncio.sleep(0.01)
            return "done"

        assert await session.until_disconnect(wss, call(), "dropped during x") == "done"

    @pytest.mark.asyncio
    async def test_fails_fast_when_tab_drops_mid_call(self):
        from fastmcp.exceptions import ToolError

        wss = Mock()
        wss.disconnected = asyncio.Event()
        cancelled = []

        async def call():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        task = asyncio.create_task(
            session.until_disconnect(wss, call(), "dropped during x")
        )
        await asyncio.sleep(0.01)
        wss.disconnected.set()
        with pytest.raises(ToolError, match="dropped during x"):
            await task
        assert cancelled == [True]

    @pytest.mark.asyncio
    async def test_rejects_immediately_when_already_disconnected(self):
        from fastmcp.exceptions import ToolError

        wss = Mock()
        wss.disconnected = asyncio.Event()
        wss.disconnected.set()

        async def call():
            return "never"

        coro = call()
        with pytest.raises(ToolError, match="before x"):
            await session.until_disconnect(wss, coro, "dropped during x")
        coro.close()
