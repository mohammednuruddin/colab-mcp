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

import argparse
import asyncio
import datetime
import logging
import tempfile
import sys

from fastmcp import FastMCP
from fastmcp.utilities import logging as fastmcp_logger

from pathlib import Path

from colab_mcp.session import SERVER_INFO, ColabSessionProxy


mcp = FastMCP(name="ColabMCP")


def init_logger(logdir):
    log_filename = datetime.datetime.now().strftime(
        f"{logdir}/colab-mcp.%Y-%m-%d_%H-%M-%S.log"
    )
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_filename,
        level=logging.INFO,  # Set the minimum logging level to capture
    )
    fastmcp_logger.get_logger("colab-mcp").info("logging to %s" % log_filename)


def parse_args(v):
    parser = argparse.ArgumentParser(
        description="ColabMCP is an MCP server that lets you interact with Colab."
    )
    parser.add_argument(
        "-l",
        "--log",
        help="if set, use this directory as a location for logfiles (if unset, will log to %s/colab-mcp-logs/)"
        % tempfile.gettempdir(),
        action="store",
        default=tempfile.mkdtemp(prefix="colab-mcp-logs-"),
    )
    parser.add_argument(
        "--state-dir",
        help="directory for the persisted connection token/port (default: ~/.colab-mcp, or COLAB_MCP_STATE_DIR)",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--ephemeral",
        help="do not persist the connection token/port; a connected tab will not survive a restart",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--rotate-token",
        help="issue a new persisted connection token now; connected tabs must be re-opened "
        "(tokens also rotate after COLAB_MCP_TOKEN_MAX_AGE_DAYS, default 30)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--port",
        help="fixed websocket port for the Colab tab to connect to (default: the persisted port, else any free port)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-p",
        "--enable-proxy",
        help="if set, enable the runtime proxy (enabled by default).",
        action="store_true",
        default=True,
    )
    return parser.parse_args(v)


async def main_async():
    args = parse_args(sys.argv[1:])
    init_logger(args.log)
    SERVER_INFO["log_dir"] = args.log

    if args.enable_proxy:
        logging.info("enabling session proxy tools")
        session_mcp = ColabSessionProxy(
            state_dir=Path(args.state_dir).expanduser() if args.state_dir else None,
            ephemeral=args.ephemeral,
            port=args.port,
            rotate_token=args.rotate_token,
        )
        await session_mcp.start_proxy_server()
        mcp.mount(session_mcp.proxy_server)
        for middleware in session_mcp.middleware:
            mcp.add_middleware(middleware)

    try:
        await mcp.run_async()

    finally:
        if args.enable_proxy:
            await session_mcp.cleanup()


def main() -> None:
    asyncio.run(main_async())
