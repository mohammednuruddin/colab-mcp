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

"""Higher-level notebook tools composed from Colab's cell primitives.

Colab's in-browser MCP server exposes seven primitives: get_cells,
add_code_cell, add_text_cell, update_cell, move_cell, delete_cell and a
blocking run_code_cell. Everything here is built on those, server-side, so an
agent gets background runs, whole-notebook execution, ad-hoc code, file
transfer and export without hand-rolling the sequencing on every turn.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp.dependencies import CurrentContext
from fastmcp.exceptions import ToolError
from fastmcp.server.context import Context
from fastmcp.tools.tool import Tool, ToolResult
from mcp.types import ImageContent, TextContent

PROXY_CLIENT_KEY = "proxy_client"

# Raw bytes per transfer cell. Base64 inflates by 4/3, so this keeps each
# cell's source or output comfortably under what the Colab bridge handles.
UPLOAD_CHUNK_BYTES = 384 * 1024
DOWNLOAD_CHUNK_BYTES = 192 * 1024
MAX_JOBS = 50
MAX_IMAGES_PER_RESULT = 4
MAX_IMAGE_BYTES = 2 * 1024 * 1024


# -- connection access ---------------------------------------------------------


def _proxy(ctx: Context):
    proxy = ctx.get_state(PROXY_CLIENT_KEY)
    if proxy is None or not proxy.is_connected() or proxy.proxy_mcp_client is None:
        message = (
            "No Colab tab is connected. Call open_colab_browser_connection (optionally with "
            "notebook_url) and have the user accept the connection first."
        )
        previous = getattr(proxy, "previous_notebook_url", None)
        if isinstance(previous, str) and previous:
            message += (
                f" Before the server restarted the connected notebook was {previous}."
            )
        raise ToolError(message)
    return proxy


def _client(ctx: Context):
    proxy = _proxy(ctx)
    client = proxy.proxy_mcp_client
    # Tag the client with its websocket server so _call can fail fast on a drop,
    # and with the lock that serialises execution on its runtime.
    with contextlib.suppress(AttributeError):
        client._colab_wss = proxy.wss
        if getattr(client, "_colab_run_lock", None) is None:
            client._colab_run_lock = asyncio.Lock()
    return client


def _run_lock(client):
    """One execution at a time per runtime.

    A background job and a foreground call would otherwise run cells on the
    same kernel concurrently, and two temporary cells placed from the same
    stale cell count would collide. The lock covers one cell's execution, not
    a whole job, so a foreground call can still run between a job's cells.
    """
    lock = getattr(client, "_colab_run_lock", None)
    return lock if lock is not None else contextlib.nullcontext()


async def _call(client, name: str, args: dict[str, Any]) -> Any:
    """Call a Colab primitive and return its structured payload (or raise ToolError)."""
    wss = getattr(client, "_colab_wss", None)
    if wss is not None:
        from colab_mcp.session import until_disconnect

        result = await until_disconnect(
            wss,
            client.call_tool(name, args, raise_on_error=False),
            f"Colab tab disconnected during {name}",
        )
    else:
        result = await client.call_tool(name, args, raise_on_error=False)
    if getattr(result, "is_error", False):
        text = " ".join(getattr(c, "text", "") for c in (result.content or []))
        raise ToolError(f"Colab {name} failed: {text or 'unknown error'}")
    payload = result.structured_content
    if payload is None and result.content:
        try:
            payload = json.loads(result.content[0].text)
        except (ValueError, AttributeError):
            payload = {"text": getattr(result.content[0], "text", "")}
    return payload


# -- output handling -----------------------------------------------------------


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _join(text: Any) -> str:
    if isinstance(text, list):
        return "".join(str(t) for t in text)
    return "" if text is None else str(text)


def _plain(text: Any) -> str:
    return _ANSI.sub("", _join(text))


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    head = limit // 2
    tail = limit - head
    return (
        f"{text[:head]}\n... [{len(text) - limit} chars truncated] ...\n{text[-tail:]}",
        True,
    )


def summarize_outputs(
    outputs: list[dict] | None, max_chars: int = 4000
) -> dict[str, Any]:
    """Collapse nbformat outputs into stdout/stderr/result/error plus image blobs."""
    stdout, stderr, results, images = [], [], [], []
    error = None
    for out in outputs or []:
        kind = out.get("output_type")
        if kind == "stream":
            (stderr if out.get("name") == "stderr" else stdout).append(
                _join(out.get("text"))
            )
        elif kind in ("execute_result", "display_data"):
            data = out.get("data") or {}
            for mime, blob in data.items():
                if mime.startswith("image/") and len(images) < MAX_IMAGES_PER_RESULT:
                    raw = _join(blob).strip()
                    if len(raw) * 3 // 4 <= MAX_IMAGE_BYTES:
                        images.append({"mime": mime, "base64": raw})
            if "text/plain" in data:
                results.append(_join(data["text/plain"]))
        elif kind == "error":
            error = {
                "ename": out.get("ename"),
                "evalue": out.get("evalue"),
                "traceback": _truncate(
                    "\n".join(_plain(line) for line in out.get("traceback") or []),
                    max_chars,
                )[0],
            }
    out_text, t1 = _truncate("".join(stdout), max_chars)
    err_text, t2 = _truncate("".join(stderr), max_chars)
    res_text, t3 = _truncate("\n".join(results), max_chars)
    summary: dict[str, Any] = {
        "ok": error is None,
        "stdout": out_text,
        "stderr": err_text,
        "result": res_text,
        "error": error,
        "image_count": len(images),
        "truncated": t1 or t2 or t3,
    }
    summary["_images"] = images
    return summary


def _result_with_images(payload: dict[str, Any], images: list[dict]) -> ToolResult:
    content: list = [TextContent(type="text", text=json.dumps(payload))]
    for image in images:
        content.append(
            ImageContent(type="image", data=image["base64"], mimeType=image["mime"])
        )
    return ToolResult(content=content, structured_content=payload)


def _strip_images(summary: dict[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    images = summary.pop("_images", [])
    return summary, images


# -- cells ---------------------------------------------------------------------


async def _cells(client, include_outputs: bool = False) -> list[dict]:
    payload = await _call(client, "get_cells", {"includeOutputs": include_outputs})
    cells = payload.get("cells") if isinstance(payload, dict) else None
    if not isinstance(cells, list):
        raise ToolError("Colab get_cells returned an unexpected payload")
    return cells


def _source(cell: dict) -> str:
    return _join(cell.get("source"))


async def _add_temp_cell(client, code: str) -> str:
    cells = await _cells(client)
    payload = await _call(
        client,
        "add_code_cell",
        {"cellIndex": len(cells), "language": "python", "code": code},
    )
    cell_id = payload.get("newCellId") if isinstance(payload, dict) else None
    if not cell_id:
        raise ToolError("Colab add_code_cell did not return a cell id")
    return cell_id


async def _delete_quiet(client, cell_id: str) -> None:
    try:
        await _call(client, "delete_cell", {"cellId": cell_id})
    except ToolError as exc:
        logging.warning("Could not delete temporary cell %s: %s", cell_id, exc)


async def _execute(client, cell_id: str, max_chars: int) -> dict[str, Any]:
    payload = await _call(client, "run_code_cell", {"cellId": cell_id})
    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    return summarize_outputs(outputs or [], max_chars)


async def _run(client, cell_id: str, max_chars: int) -> dict[str, Any]:
    async with _run_lock(client):
        return await _execute(client, cell_id, max_chars)


async def _run_temp(
    client, code: str, max_chars: int, keep: bool = False
) -> dict[str, Any]:
    # Held from placement to deletion so no other temporary cell is added at
    # the same index and nothing else executes in between.
    async with _run_lock(client):
        cell_id = await _add_temp_cell(client, code)
        try:
            summary = await _execute(client, cell_id, max_chars)
        finally:
            if not keep:
                await _delete_quiet(client, cell_id)
    summary["cell_id"] = cell_id if keep else None
    return summary


# -- jobs ----------------------------------------------------------------------


@dataclass
class Job:
    id: str
    kind: str
    state: str = "running"  # running | done | failed | cancelled
    started: float = field(default_factory=time.time)
    finished: float | None = None
    total: int = 0
    current: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    cancel_requested: bool = False
    task: asyncio.Task | None = None

    def view(self, max_chars: int) -> dict[str, Any]:
        results = []
        for r in self.results:
            r = dict(r)
            r.pop("_images", None)
            for key in ("stdout", "stderr", "result"):
                if isinstance(r.get(key), str):
                    r[key], _ = _truncate(r[key], max_chars)
            results.append(r)
        return {
            "job_id": self.id,
            "kind": self.kind,
            "state": self.state,
            "started": self.started,
            "finished": self.finished,
            "elapsed_seconds": round((self.finished or time.time()) - self.started, 1),
            "total_cells": self.total,
            "completed_cells": len(self.results),
            "current_cell": self.current,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "results": results,
        }


JOBS: dict[str, Job] = {}


def _register(job: Job) -> None:
    JOBS[job.id] = job
    while len(JOBS) > MAX_JOBS:
        oldest = next(iter(JOBS))
        if JOBS[oldest].state == "running":
            break
        del JOBS[oldest]


async def _run_sequence(
    client, job: Job, cell_ids: list[str], stop_on_error: bool, max_chars: int
) -> None:
    job.total = len(cell_ids)
    try:
        for cell_id in cell_ids:
            if job.cancel_requested:
                job.state = "cancelled"
                break
            job.current = cell_id
            summary = await _run(client, cell_id, max_chars)
            summary["cell_id"] = cell_id
            job.results.append(summary)
            if not summary["ok"] and stop_on_error:
                job.state = "failed"
                job.error = f"cell {cell_id} raised {summary['error']['ename']}: {summary['error']['evalue']}"
                break
        else:
            job.state = "done" if all(r["ok"] for r in job.results) else "failed"
    except asyncio.CancelledError:
        job.state = "cancelled"
        raise
    except Exception as exc:  # the bridge dropped, or Colab refused a call
        job.state = "failed"
        job.error = str(exc)
    finally:
        job.current = None
        job.finished = time.time()


def _start_job(kind: str, coro) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], kind=kind)
    job.task = asyncio.create_task(coro(job))
    _register(job)
    return job


async def _select_cells(
    client, cell_ids: list[str] | None, start_index: int | None, end_index: int | None
) -> list[str]:
    cells = await _cells(client)
    if cell_ids:
        known = {c.get("id") for c in cells}
        missing = [c for c in cell_ids if c not in known]
        if missing:
            raise ToolError(f"Unknown cell ids: {missing}")
        return list(cell_ids)
    start = 0 if start_index is None else max(0, start_index)
    end = len(cells) if end_index is None else min(len(cells), end_index)
    return [
        c["id"]
        for c in cells[start:end]
        if c.get("cell_type") == "code" and c.get("id")
    ]


# -- tools ---------------------------------------------------------------------


async def get_notebook_outline(ctx: Context = CurrentContext()) -> dict:
    """List cells as index, id, type, first source line and whether outputs exist."""
    client = _client(ctx)
    cells = await _cells(client, include_outputs=True)
    outline = []
    for index, cell in enumerate(cells):
        source = _source(cell)
        first = source.strip().splitlines()[0] if source.strip() else ""
        outline.append(
            {
                "index": index,
                "id": cell.get("id"),
                "type": cell.get("cell_type"),
                "first_line": first[:120],
                "lines": source.count("\n") + (1 if source else 0),
                "has_outputs": bool(cell.get("outputs")),
                "has_error": any(
                    o.get("output_type") == "error" for o in cell.get("outputs") or []
                ),
            }
        )
    return {"cell_count": len(cells), "cells": outline}


async def run_cells(
    cell_ids: list[str] | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    stop_on_error: bool = True,
    background: bool = False,
    max_output_chars: int = 4000,
    ctx: Context = CurrentContext(),
) -> dict:
    """Run code cells in order: by id list, or by index range (end exclusive), or all.

    Each cell blocks until it finishes. For anything that may take more than a
    minute set background=True and poll job_status; Colab cannot be
    interrupted through this bridge, so cancel only stops scheduling the next
    cell.
    """
    client = _client(ctx)
    ids = await _select_cells(client, cell_ids, start_index, end_index)
    if not ids:
        raise ToolError("No code cells matched")
    kind = "run_cells"
    if background:
        job = _start_job(
            kind,
            lambda j: _run_sequence(client, j, ids, stop_on_error, max_output_chars),
        )
        return {
            "job_id": job.id,
            "state": job.state,
            "total_cells": len(ids),
            "poll_with": "job_status",
        }
    job = Job(id=uuid.uuid4().hex[:12], kind=kind)
    _register(job)
    await _run_sequence(client, job, ids, stop_on_error, max_output_chars)
    return job.view(max_output_chars)


async def run_notebook(
    stop_on_error: bool = True,
    background: bool = True,
    max_output_chars: int = 2000,
    ctx: Context = CurrentContext(),
) -> dict:
    """Run every code cell from the top. Background by default: poll job_status."""
    return await run_cells(
        stop_on_error=stop_on_error,
        background=background,
        max_output_chars=max_output_chars,
        ctx=ctx,
    )


async def run_code(
    code: str,
    keep_cell: bool = False,
    background: bool = False,
    max_output_chars: int = 8000,
    ctx: Context = CurrentContext(),
) -> ToolResult:
    """Run ad-hoc Python in the notebook's runtime via a temporary cell at the end.

    The cell is deleted afterwards unless keep_cell=True. Images in the output
    (plots) are returned as image content. background=True returns a job id.
    """
    client = _client(ctx)
    if not code.strip():
        raise ToolError("code is empty")
    if background:

        async def body(job: Job):
            job.total = 1
            try:
                summary = await _run_temp(
                    client, code, max_output_chars, keep=keep_cell
                )
                job.results.append(summary)
                job.state = "done" if summary["ok"] else "failed"
                if not summary["ok"]:
                    job.error = (
                        f"{summary['error']['ename']}: {summary['error']['evalue']}"
                    )
            except Exception as exc:
                job.state = "failed"
                job.error = str(exc)
            finally:
                job.finished = time.time()

        job = _start_job("run_code", body)
        return ToolResult(
            content=[
                TextContent(
                    type="text", text=json.dumps({"job_id": job.id, "state": "running"})
                )
            ],
            structured_content={
                "job_id": job.id,
                "state": "running",
                "poll_with": "job_status",
            },
        )
    summary = await _run_temp(client, code, max_output_chars, keep=keep_cell)
    summary, images = _strip_images(summary)
    return _result_with_images(summary, images)


async def job_status(
    job_id: str, max_output_chars: int = 2000, ctx: Context = CurrentContext()
) -> dict:
    """Progress and per-cell results of a background run."""
    job = JOBS.get(job_id)
    if job is None:
        raise ToolError(f"Unknown job {job_id}")
    return job.view(max_output_chars)


async def list_jobs(ctx: Context = CurrentContext()) -> dict:
    """All known background runs, newest last."""
    return {"jobs": [job.view(200) | {"results": None} for job in JOBS.values()]}


async def cancel_job(job_id: str, ctx: Context = CurrentContext()) -> dict:
    """Stop a background run after its current cell. The running cell itself cannot be interrupted."""
    job = JOBS.get(job_id)
    if job is None:
        raise ToolError(f"Unknown job {job_id}")
    if job.state != "running":
        return {"job_id": job.id, "state": job.state, "note": "job already finished"}
    job.cancel_requested = True
    return {"job_id": job.id, "state": job.state, "cancel_requested": True}


# -- file transfer -------------------------------------------------------------


def _local(path: str) -> Path:
    return Path(path).expanduser().resolve()


_HASH_SNIPPET = """
import hashlib, os, json
_p = os.path.expanduser({path!r})
_h = hashlib.sha256()
with open(_p, 'rb') as _f:
    for _chunk in iter(lambda: _f.read(1 << 20), b''):
        _h.update(_chunk)
print(json.dumps({{"sha256": _h.hexdigest(), "size": os.path.getsize(_p)}}))
"""


async def _remote_hash(client, remote_path: str) -> dict:
    summary = await _run_temp(client, _HASH_SNIPPET.format(path=remote_path), 2000)
    if not summary["ok"]:
        raise ToolError(
            f"Could not hash {remote_path} on the runtime: {summary['error']['evalue']}"
        )
    try:
        return json.loads(summary["stdout"].strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ToolError(f"Unexpected hash output: {summary['stdout'][:200]}") from exc


async def upload_file(
    local_path: str,
    remote_path: str,
    overwrite: bool = True,
    ctx: Context = CurrentContext(),
) -> dict:
    """Copy a local file into the notebook runtime's filesystem (e.g. /content/data.csv).

    Transfers in base64 chunks through temporary cells and verifies the
    SHA-256 on the runtime afterwards.
    """
    client = _client(ctx)
    src = _local(local_path)
    if not src.is_file():
        raise ToolError(f"Local file not found: {src}")
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not overwrite:
        probe = await _run_temp(
            client,
            f"import os; print(os.path.exists(os.path.expanduser({remote_path!r})))",
            200,
        )
        if probe["ok"] and probe["stdout"].strip() == "True":
            raise ToolError(f"{remote_path} exists on the runtime and overwrite=False")
    chunks = [
        data[i : i + UPLOAD_CHUNK_BYTES]
        for i in range(0, len(data), UPLOAD_CHUNK_BYTES)
    ] or [b""]
    for index, chunk in enumerate(chunks):
        mode = "wb" if index == 0 else "ab"
        code = (
            "import base64, os\n"
            f"_p = os.path.expanduser({remote_path!r})\n"
            "os.makedirs(os.path.dirname(_p) or '.', exist_ok=True)\n"
            f"with open(_p, {mode!r}) as _f:\n"
            f"    _f.write(base64.b64decode({base64.b64encode(chunk).decode()!r}))\n"
            f"print('chunk', {index}, 'ok')\n"
        )
        summary = await _run_temp(client, code, 500)
        if not summary["ok"]:
            raise ToolError(
                f"Upload chunk {index + 1}/{len(chunks)} failed: {summary['error']['evalue']}"
            )
    remote = await _remote_hash(client, remote_path)
    if remote["sha256"] != digest:
        raise ToolError(
            f"Upload verification failed: local sha256 {digest} != remote {remote['sha256']}"
        )
    return {
        "remote_path": remote_path,
        "bytes": len(data),
        "sha256": digest,
        "chunks": len(chunks),
    }


_READ_SNIPPET = """
import base64, os
_p = os.path.expanduser({path!r})
with open(_p, 'rb') as _f:
    _f.seek({offset})
    _d = _f.read({size})
print('<<B64>>' + base64.b64encode(_d).decode() + '<<END>>')
"""


async def download_file(
    remote_path: str,
    local_path: str,
    overwrite: bool = False,
    ctx: Context = CurrentContext(),
) -> dict:
    """Copy a file from the notebook runtime to this machine, verifying its SHA-256."""
    client = _client(ctx)
    dst = _local(local_path)
    if dst.exists() and not overwrite:
        raise ToolError(f"{dst} exists locally and overwrite=False")
    remote = await _remote_hash(client, remote_path)
    size = int(remote["size"])
    buffer = bytearray()
    offset = 0
    chunks = 0
    while offset < size or (size == 0 and chunks == 0):
        code = _READ_SNIPPET.format(
            path=remote_path, offset=offset, size=DOWNLOAD_CHUNK_BYTES
        )
        summary = await _run_temp(client, code, 0)
        if not summary["ok"]:
            raise ToolError(
                f"Download failed at offset {offset}: {summary['error']['evalue']}"
            )
        text = summary["stdout"]
        start, end = text.find("<<B64>>"), text.rfind("<<END>>")
        if start < 0 or end < 0:
            raise ToolError(
                f"Download chunk at offset {offset} was truncated by the bridge; lower DOWNLOAD_CHUNK_BYTES"
            )
        piece = base64.b64decode(text[start + 7 : end].strip())
        if not piece and size > 0:
            raise ToolError(f"Empty chunk at offset {offset}")
        buffer.extend(piece)
        offset += len(piece)
        chunks += 1
        if size == 0:
            break
    digest = hashlib.sha256(bytes(buffer)).hexdigest()
    if digest != remote["sha256"]:
        raise ToolError(
            f"Download verification failed: remote sha256 {remote['sha256']} != local {digest}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(bytes(buffer))
    return {
        "local_path": str(dst),
        "bytes": len(buffer),
        "sha256": digest,
        "chunks": chunks,
    }


# -- export --------------------------------------------------------------------


def to_ipynb(cells: list[dict], include_outputs: bool = True) -> dict:
    out_cells = []
    for cell in cells:
        kind = cell.get("cell_type", "code")
        entry: dict[str, Any] = {
            "cell_type": kind,
            "metadata": {"id": cell.get("id")} if cell.get("id") else {},
            "source": _source(cell),
        }
        if kind == "code":
            entry["execution_count"] = cell.get("execution_count")
            entry["outputs"] = (cell.get("outputs") or []) if include_outputs else []
        out_cells.append(entry)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "cells": out_cells,
    }


async def export_notebook(
    local_path: str, include_outputs: bool = True, ctx: Context = CurrentContext()
) -> dict:
    """Save the connected notebook as an .ipynb file on this machine."""
    client = _client(ctx)
    cells = await _cells(client, include_outputs=include_outputs)
    dst = _local(local_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(to_ipynb(cells, include_outputs), indent=1))
    return {
        "local_path": str(dst),
        "cell_count": len(cells),
        "bytes": dst.stat().st_size,
    }


NOTEBOOK_TOOLS = [
    Tool.from_function(fn=get_notebook_outline, name="get_notebook_outline"),
    Tool.from_function(fn=run_cells, name="run_cells"),
    Tool.from_function(fn=run_notebook, name="run_notebook"),
    Tool.from_function(fn=run_code, name="run_code"),
    Tool.from_function(fn=job_status, name="job_status"),
    Tool.from_function(fn=list_jobs, name="list_jobs"),
    Tool.from_function(fn=cancel_job, name="cancel_job"),
    Tool.from_function(fn=upload_file, name="upload_file"),
    Tool.from_function(fn=download_file, name="download_file"),
    Tool.from_function(fn=export_notebook, name="export_notebook"),
]

# Tools that must not trigger an automatic tab re-open (they do not need Colab).
PASSIVE_TOOL_NAMES = frozenset({"job_status", "list_jobs", "cancel_job"})
