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

"""Persisted connection identity so a Colab tab can outlive a server restart.

Upstream generates a fresh token and port on every start, which strands any
tab that was connected. Keeping both on disk (owner-readable only) lets the
same tab reconnect to the next server process without a new browser round
trip. Pass ``--ephemeral`` to get the upstream behaviour back.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

DEFAULT_STATE_DIR = Path.home() / ".colab-mcp"
STATE_FILE = "connection.json"


@dataclass
class ConnectionState:
    token: str
    port: int
    last_notebook_url: str | None = None


def default_state_dir() -> Path:
    override = os.environ.get("COLAB_MCP_STATE_DIR")
    return Path(override).expanduser() if override else DEFAULT_STATE_DIR


def load_state(state_dir: Path) -> ConnectionState | None:
    path = state_dir / STATE_FILE
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logging.warning("Ignoring unreadable connection state at %s: %s", path, exc)
        return None
    token = raw.get("token")
    port = raw.get("port")
    if (
        not isinstance(token, str)
        or not token
        or not isinstance(port, int)
        or not (0 < port < 65536)
    ):
        logging.warning("Ignoring malformed connection state at %s", path)
        return None
    notebook = raw.get("last_notebook_url")
    return ConnectionState(
        token=token,
        port=port,
        last_notebook_url=notebook if isinstance(notebook, str) else None,
    )


def save_state(state_dir: Path, state: ConnectionState) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / STATE_FILE
    payload = json.dumps(
        {
            "token": state.token,
            "port": state.port,
            "last_notebook_url": state.last_notebook_url,
        },
        indent=2,
    )
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(payload)
    os.replace(tmp, path)


def new_token() -> str:
    return secrets.token_urlsafe(16)
