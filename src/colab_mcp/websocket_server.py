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

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
import asyncio
import logging
import mcp.types as types
from mcp.shared.message import SessionMessage
from pydantic_core import ValidationError
import secrets
import websockets
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.typing import Subprotocol


COLAB = "https://colab.research.google.com"
COLAB_ALT_DOMAIN = "https://colab.google.com"
SCRATCH_PATH = "/notebooks/empty.ipynb"
# Largest single MCP message the bridge accepts. The library default of 1 MiB
# dropped the connection on a ~400 KB cell output; notebook outputs and file
# transfer chunks can legitimately be larger.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class ColabWebSocketServer:
    """
    A WebSocket server designed to accept a single connection specifically
    from a Google Colab session (colab.google.com).
    """

    def __init__(
        self,
        host="localhost",
        port: int = 0,
        token: str | None = None,
        on_disconnect=None,
    ):
        self.host = host
        self.port = 0
        self.requested_port = port
        # Called (synchronously) after a connected tab drops, so the owner can
        # recycle the MCP session that was bound to it.
        self.on_disconnect = on_disconnect
        self.connection_lock = asyncio.Lock()
        self.connection_live = asyncio.Event()
        # Set while no tab is attached; lets an in-flight call notice a drop.
        self.disconnected = asyncio.Event()
        self.disconnected.set()
        self.allowed_origins = [COLAB, COLAB_ALT_DOMAIN]
        self._server: websockets.Server | None = None

        self.read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
        self._read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]
        self.write_stream: MemoryObjectSendStream[SessionMessage]
        self._write_stream_reader: MemoryObjectReceiveStream[SessionMessage]
        self.reset_streams()
        self.token = token or secrets.token_urlsafe(16)
        self._active_connection: ServerConnection | None = None

    def reset_streams(self):
        """Create fresh MCP message streams.

        An MCP ClientSession closes the streams it was given when it shuts
        down, so switching to a different notebook (which starts a new
        session) needs new streams before the next browser tab connects.
        """
        self._read_stream_writer, self.read_stream = anyio.create_memory_object_stream(
            0
        )
        self.write_stream, self._write_stream_reader = (
            anyio.create_memory_object_stream(0)
        )

    async def disconnect(self):
        """Close the connected Colab tab's socket, if any, and wait for it to drop."""
        connection = self._active_connection
        if connection is None:
            return
        await connection.close(code=1000, reason="Switching notebook")
        # The handler clears connection_live in its finally block; give it a moment.
        for _ in range(50):
            if not self.connection_live.is_set():
                break
            await asyncio.sleep(0.02)

    async def _read_from_socket(self, websocket):
        """Listens to the socket and puts messages into the read stream."""
        try:
            await self._pump_socket(websocket)
        except websockets.exceptions.ConnectionClosed as e:
            logging.info(f"Colab tab connection closed: {e.code} - {e.reason}")

    async def _pump_socket(self, websocket):
        async for msg in websocket:
            try:
                client_message = types.JSONRPCMessage.model_validate_json(msg)
            except ValidationError as exc:
                await self._read_stream_writer.send(exc)
                continue
            await self._read_stream_writer.send(SessionMessage(client_message))

    async def _write_to_socket(self, websocket):
        """Reads from the write stream and sends over the socket."""
        try:
            while True:
                # Wait for a message from the application
                msg = await self._write_stream_reader.receive()

                try:
                    json_obj = msg.message.model_dump_json(
                        by_alias=True, exclude_none=True
                    )
                    await websocket.send(json_obj)
                except ConnectionClosed:
                    break
        except (anyio.ClosedResourceError, anyio.EndOfStream):
            # server closed write stream
            pass

    def _validate_authorization(self, websocket: ServerConnection, request: Request):
        if request.path.find(f"access_token={self.token}") != -1:
            return None
        try:
            headers: Headers = request.headers
            auth_header = headers.get("Authorization")
            if not auth_header:
                return Response(401, "Missing authorization", Headers([]))
            scheme, token = auth_header.split(None, 1)
            if scheme.lower() != "bearer":
                return Response(400, "Invalid authorization header", Headers([]))
        except ValueError:
            return Response(400, "Invalid header format", Headers([]))
        if token == self.token:
            return None
        return Response(403, "Bad authorization token", Headers([]))

    async def _connection_handler(self, websocket: ServerConnection):
        """
        Handles incoming websocket connections.
        Validates Origin and ensures single-client exclusivity.
        """
        if self.connection_lock.locked():
            logging.warning(
                f"Connection rejected: {websocket.remote_address}. A client is already connected"
            )
            await websocket.close(code=1013, reason="Server is busy")
            return

        async with self.connection_lock:
            try:
                self._active_connection = websocket
                self.disconnected.clear()
                self.connection_live.set()

                reading_task = asyncio.create_task(self._read_from_socket(websocket))
                writing_task = asyncio.create_task(self._write_to_socket(websocket))
                _, pending = await asyncio.wait(
                    [reading_task, writing_task], return_when=asyncio.FIRST_COMPLETED
                )

                for task in pending:
                    task.cancel()

            except websockets.exceptions.ConnectionClosed as e:
                logging.info(f"Connection closed: {e.code} - {e.reason}")
                await self._read_stream_writer.send(
                    Exception("Colab Frontend disconnected")
                )
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
            finally:
                self._active_connection = None
                self.connection_live.clear()
                self.disconnected.set()
                if self.on_disconnect is not None:
                    try:
                        self.on_disconnect()
                    except Exception as exc:  # pragma: no cover - defensive
                        logging.error(f"on_disconnect handler failed: {exc}")

    async def _serve(self, port: int):
        return await websockets.serve(
            self._connection_handler,
            host=self.host,
            port=port,
            subprotocols=[Subprotocol("mcp")],
            origins=self.allowed_origins,
            process_request=self._validate_authorization,
            max_size=MAX_MESSAGE_BYTES,
        )

    async def __aenter__(self):
        try:
            self._server = await self._serve(self.requested_port)
        except OSError as exc:
            if not self.requested_port:
                raise
            logging.warning(
                f"Port {self.requested_port} unavailable ({exc}); a connected tab will need to be re-opened"
            )
            self._server = await self._serve(0)
        self.port = self._server.sockets[0].getsockname()[1]
        logging.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logging.info("Closing WebSocket server")
        if self._server:
            self._server.close()
            self.write_stream.close()
            self.read_stream.close()
            await self._server.wait_closed()
