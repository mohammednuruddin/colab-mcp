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

A persisted token also outlives its exposure: it travels in the URL fragment
of every Colab link the server opens, so it lands in browser history. It is
replaced once it is older than ``COLAB_MCP_TOKEN_MAX_AGE_DAYS`` (default 30, 0
never) or on ``--rotate-token``.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_STATE_DIR = Path.home() / ".colab-mcp"
STATE_FILE = "connection.json"
DEFAULT_TOKEN_MAX_AGE_DAYS = 30.0


@dataclass
class ConnectionState:
    token: str
    port: int
    last_notebook_url: str | None = None
    # When the token was issued (epoch seconds).
    token_created: float = field(default_factory=time.time)

    def token_expired(self, max_age_days: float, now: float | None = None) -> bool:
        if max_age_days <= 0:
            return False
        age = (time.time() if now is None else now) - self.token_created
        return age > max_age_days * 86400


def default_state_dir() -> Path:
    override = os.environ.get("COLAB_MCP_STATE_DIR")
    return Path(override).expanduser() if override else DEFAULT_STATE_DIR


def token_max_age_days() -> float:
    raw = os.environ.get("COLAB_MCP_TOKEN_MAX_AGE_DAYS")
    if raw is None:
        return DEFAULT_TOKEN_MAX_AGE_DAYS
    try:
        return float(raw)
    except ValueError:
        logging.warning(
            "Ignoring non-numeric COLAB_MCP_TOKEN_MAX_AGE_DAYS=%r; using %s",
            raw,
            DEFAULT_TOKEN_MAX_AGE_DAYS,
        )
        return DEFAULT_TOKEN_MAX_AGE_DAYS


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
    created = raw.get("token_created")
    state = ConnectionState(
        token=token,
        port=port,
        last_notebook_url=notebook if isinstance(notebook, str) else None,
    )
    # A file written before token_created existed keeps its token: its age is
    # unknown, and rotating it would strand a tab that is connected right now.
    if isinstance(created, (int, float)) and not isinstance(created, bool):
        state.token_created = float(created)
    return state


def save_state(state_dir: Path, state: ConnectionState) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / STATE_FILE
    payload = json.dumps(
        {
            "token": state.token,
            "port": state.port,
            "last_notebook_url": state.last_notebook_url,
            "token_created": state.token_created,
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
