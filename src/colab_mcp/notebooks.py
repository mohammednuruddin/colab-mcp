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

"""Turn the notebook links people actually paste into Colab notebook URLs.

Colab opens notebooks from Drive, GitHub, and gists through its own URL
scheme. The agent gets handed whatever the user copied — a Drive share link, a
GitHub blob URL, a raw.githubusercontent link — so this module maps all of
those onto the ``colab.research.google.com`` form the browser connection needs.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit, parse_qs

COLAB = "https://colab.research.google.com"
COLAB_HOSTS = {"colab.research.google.com", "colab.google.com"}
SCRATCH_PATH = "/notebooks/empty.ipynb"

_DRIVE_FILE = re.compile(r"^/file/d/(?P<id>[A-Za-z0-9_-]+)")
_DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")
_GITHUB_BLOB = re.compile(
    r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<rest>.+\.ipynb)$"
)
_RAW_GITHUB = re.compile(r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/(?P<rest>.+\.ipynb)$")
_GIST = re.compile(r"^/(?P<user>[^/]+)/(?P<id>[0-9a-f]+)")


class UnsupportedNotebookUrl(ValueError):
    """The link is not something Colab can open as a notebook."""


def scratch_notebook_url() -> str:
    return f"{COLAB}{SCRATCH_PATH}"


def normalize_notebook_url(link: str) -> str:
    """Return the Colab URL for a notebook link, or raise UnsupportedNotebookUrl.

    Accepted inputs:
      * any ``colab.research.google.com`` / ``colab.google.com`` notebook URL
      * Google Drive share links (``/file/d/<id>``, ``open?id=<id>``) or a bare Drive file id
      * GitHub blob URLs ending in ``.ipynb`` and ``raw.githubusercontent.com`` links
      * GitHub gist URLs

    Any URL fragment is dropped: the connection parameters go there.
    """
    link = link.strip()
    if not link:
        raise UnsupportedNotebookUrl("notebook_url is empty")
    if _DRIVE_ID.match(link) and "/" not in link and "." not in link:
        return f"{COLAB}/drive/{link}"

    parts = urlsplit(link if "://" in link else f"https://{link}")
    host = parts.netloc.lower()
    path = parts.path

    if host in COLAB_HOSTS:
        if not path or path == "/":
            raise UnsupportedNotebookUrl("Colab URL does not point at a notebook")
        return urlunsplit(("https", "colab.research.google.com", path, parts.query, ""))

    if host in {"drive.google.com", "docs.google.com"}:
        match = _DRIVE_FILE.match(path)
        file_id = (
            match.group("id") if match else parse_qs(parts.query).get("id", [None])[0]
        )
        if not file_id:
            raise UnsupportedNotebookUrl("Drive link has no file id")
        return f"{COLAB}/drive/{file_id}"

    if host in {"github.com", "www.github.com"}:
        match = _GITHUB_BLOB.match(path)
        if not match:
            raise UnsupportedNotebookUrl(
                "GitHub link must be a blob URL to an .ipynb file"
            )
        return f"{COLAB}/github/{match['owner']}/{match['repo']}/blob/{match['rest']}"

    if host == "raw.githubusercontent.com":
        match = _RAW_GITHUB.match(path)
        if not match:
            raise UnsupportedNotebookUrl("raw GitHub link must point at an .ipynb file")
        return f"{COLAB}/github/{match['owner']}/{match['repo']}/blob/{match['rest']}"

    if host == "gist.github.com":
        match = _GIST.match(path)
        if not match:
            raise UnsupportedNotebookUrl("gist link must be /<user>/<gist id>")
        return f"{COLAB}/gist/{match['user']}/{match['id']}"

    raise UnsupportedNotebookUrl(
        f"Cannot open '{link}' in Colab. Use a Colab, Google Drive, GitHub (.ipynb), or gist link."
    )


def connection_url(notebook_url: str, token: str, port: int) -> str:
    """The URL the browser must load so the Colab tab connects back to this server."""
    return f"{notebook_url}#mcpProxyToken={token}&mcpProxyPort={port}"
