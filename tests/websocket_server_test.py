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
from colab_mcp.websocket_server import ColabWebSocketServer
from mcp.types import JSONRPCRequest, JSONRPCResponse, JSONRPCMessage
from mcp.shared.message import SessionMessage
import pytest
import websockets


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin_domain", ["https://colab.google.com", "https://colab.research.google.com"]
)
async def test_successful_connection(origin_domain):
    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin=origin_domain,
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()
        assert server.connection_lock.locked()

        await client.close()
        await client.wait_closed()
        await asyncio.sleep(1)  # Allow server to update state

        assert not server.connection_live.is_set()
        assert not server.connection_lock.locked()


@pytest.mark.asyncio
async def test_unauthorized_origin_rejected():
    async with ColabWebSocketServer() as server:
        with pytest.raises(websockets.exceptions.InvalidStatus):
            await websockets.connect(
                f"ws://localhost:{server.port}",
                origin="https://wrong.com",
                subprotocols=["mcp"],
                additional_headers={"Authorization": f"Bearer {server.token}"},
            )
        assert not server.connection_live.is_set()


@pytest.mark.asyncio
async def test_second_connection_rejected():
    async with ColabWebSocketServer() as server:
        client1 = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()

        client2 = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )

        with pytest.raises(
            websockets.exceptions.ConnectionClosed,
            match="Server is busy",
            check=lambda e: e.rcvd.code == 1013,
        ):
            # assert we cannot ping via the second client
            await client2.ping()

        # assert we can ping via the original client
        pong = await client1.ping()
        pong_latency = await pong
        assert pong_latency > 0
        await client1.close()


@pytest.mark.asyncio
async def test_incoming_message_handling():
    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()

        test_message = JSONRPCResponse(
            jsonrpc="2.0",
            id="abc",
            result={"result": "success"},
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        await client.send(test_message.model_dump_json())

        received_msg = await asyncio.wait_for(server.read_stream.receive(), timeout=1)
        test_json_message = JSONRPCMessage(test_message)
        assert received_msg.message == test_json_message

        await client.close()


@pytest.mark.asyncio
async def test_outgoing_message_handling():
    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()

        test_message = JSONRPCRequest(
            jsonrpc="2.0",
            id="abc",
            method="test_method",
            params={"bar": "baz"},
        )
        await server.write_stream.send(SessionMessage(test_message))

        received_msg_str = await asyncio.wait_for(client.recv(), timeout=1)
        received_msg = JSONRPCRequest.model_validate_json(received_msg_str)
        assert received_msg == test_message

        await client.close()


@pytest.mark.asyncio
async def test_malformed_incoming_message():
    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()

        await client.send("this is not json")

        received_item = await asyncio.wait_for(server.read_stream.receive(), timeout=1)
        assert isinstance(received_item, Exception)

        await client.close()


@pytest.mark.asyncio
async def test_bad_token():
    with pytest.raises(
        websockets.exceptions.InvalidStatus,
        check=lambda e: e.response.status_code == 403,
    ):
        async with ColabWebSocketServer() as server:
            await websockets.connect(
                f"ws://localhost:{server.port}",
                origin="https://colab.google.com",
                subprotocols=["mcp"],
                additional_headers={"Authorization": "Bearer bad_token"},
            )


@pytest.mark.asyncio
async def test_no_auth():
    with pytest.raises(
        websockets.exceptions.InvalidStatus,
        check=lambda e: e.response.status_code == 401,
    ):
        async with ColabWebSocketServer() as server:
            await websockets.connect(
                f"ws://localhost:{server.port}",
                origin="https://colab.google.com",
                subprotocols=["mcp"],
            )


@pytest.mark.asyncio
async def test_malformed_auth_header():
    with pytest.raises(
        websockets.exceptions.InvalidStatus,
        check=lambda e: e.response.status_code == 400,
    ):
        async with ColabWebSocketServer() as server:
            await websockets.connect(
                f"ws://localhost:{server.port}",
                origin="https://colab.google.com",
                subprotocols=["mcp"],
                additional_headers={"Authorization": f"Bearer?{server.token}"},
            )


@pytest.mark.asyncio
async def test_token_in_url():
    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}?access_token={server.token}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
        )
        assert server.connection_live.is_set()
        assert server.connection_lock.locked()

        await client.close()
        await client.wait_closed()
        await asyncio.sleep(1)  # Allow server to update state

        assert not server.connection_live.is_set()
        assert not server.connection_lock.locked()


@pytest.mark.asyncio
async def test_disconnect_drops_active_tab_and_allows_reconnect():
    async with ColabWebSocketServer() as server:
        client1 = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()

        await server.disconnect()
        await client1.wait_closed()
        assert not server.connection_live.is_set()
        assert not server.connection_lock.locked()

        old_read, old_write = server.read_stream, server.write_stream
        server.reset_streams()
        assert (
            server.read_stream is not old_read and server.write_stream is not old_write
        )

        client2 = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        assert server.connection_live.is_set()
        await client2.close()
        await client2.wait_closed()


@pytest.mark.asyncio
async def test_fixed_port_and_token_are_honoured_and_fallback_when_taken():
    async with ColabWebSocketServer(token="fixed-token") as first:
        assert first.token == "fixed-token"
        wanted = first.port
        # The port is busy: a second server asking for it must fall back, not crash.
        async with ColabWebSocketServer(port=wanted) as second:
            assert second.port != wanted
            assert second.port > 0
    async with ColabWebSocketServer(port=wanted, token="fixed-token") as third:
        assert third.port == wanted


@pytest.mark.asyncio
async def test_on_disconnect_callback_fires_when_tab_drops():
    fired = []
    async with ColabWebSocketServer(on_disconnect=lambda: fired.append(True)) as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
        )
        await client.close()
        await client.wait_closed()
        for _ in range(50):
            if fired:
                break
            await asyncio.sleep(0.02)
    assert fired == [True]


@pytest.mark.asyncio
async def test_large_frames_are_accepted():
    from colab_mcp.websocket_server import MAX_MESSAGE_BYTES
    from mcp.types import JSONRPCMessage, JSONRPCNotification

    async with ColabWebSocketServer() as server:
        client = await websockets.connect(
            f"ws://localhost:{server.port}",
            origin="https://colab.google.com",
            subprotocols=["mcp"],
            additional_headers={"Authorization": f"Bearer {server.token}"},
            max_size=MAX_MESSAGE_BYTES,
        )
        assert not server.disconnected.is_set()
        big = JSONRPCMessage(
            JSONRPCNotification(
                jsonrpc="2.0", method="big", params={"blob": "x" * (3 * 1024 * 1024)}
            )
        )
        reader = asyncio.create_task(server.read_stream.receive())
        await client.send(big.model_dump_json())
        received = await asyncio.wait_for(reader, 5)
        assert len(received.message.root.params["blob"]) == 3 * 1024 * 1024
        await client.close()
        await client.wait_closed()
        for _ in range(50):
            if server.disconnected.is_set():
                break
            await asyncio.sleep(0.02)
        assert server.disconnected.is_set()
