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

"""Tests for the composite notebook tools against a fake Colab bridge.

The fake keeps a cell list and really executes python for run_code_cell,
producing nbformat outputs, so the upload/download snippets are exercised
for real against a temporary directory.
"""

import asyncio
import contextlib
import io
import json
import traceback
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastmcp.exceptions import ToolError
from mcp.types import TextContent

from colab_mcp import notebook_ops as ops


class FakeColab:
    def __init__(self, cells=None, fail_on=None):
        self.cells = cells or []
        self.counter = 0
        self.calls = []
        self.fail_on = fail_on or {}
        self.namespace = {}

    def _new_id(self):
        self.counter += 1
        return f"cell{self.counter}"

    def _find(self, cell_id):
        for cell in self.cells:
            if cell["id"] == cell_id:
                return cell
        raise KeyError(cell_id)

    def _exec(self, code):
        outputs = []
        if code in self.fail_on:
            return [
                {
                    "output_type": "error",
                    "ename": "RuntimeError",
                    "evalue": self.fail_on[code],
                    "traceback": [
                        "\x1b[0;31mRuntimeError\x1b[0m: " + self.fail_on[code]
                    ],
                }
            ]
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(code, self.namespace)
        except Exception as exc:
            outputs.append(
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
        if out.getvalue():
            outputs.insert(
                0, {"output_type": "stream", "name": "stdout", "text": [out.getvalue()]}
            )
        if err.getvalue():
            outputs.append(
                {"output_type": "stream", "name": "stderr", "text": [err.getvalue()]}
            )
        return outputs

    async def call_tool(self, name, args, raise_on_error=True):
        self.calls.append((name, args))
        await asyncio.sleep(0)
        if name == "get_cells":
            cells = [dict(c) for c in self.cells]
            if not args.get("includeOutputs"):
                for c in cells:
                    c.pop("outputs", None)
            payload = {"cells": cells}
        elif name == "add_code_cell":
            cell = {
                "cell_type": "code",
                "id": self._new_id(),
                "source": [args["code"]],
                "outputs": [],
            }
            self.cells.insert(args["cellIndex"], cell)
            payload = {"newCellId": cell["id"]}
        elif name == "delete_cell":
            self.cells.remove(self._find(args["cellId"]))
            payload = {}
        elif name == "run_code_cell":
            cell = self._find(args["cellId"])
            outputs = self._exec("".join(cell["source"]))
            cell["outputs"] = outputs
            payload = {"outputs": outputs}
        else:
            raise AssertionError(name)
        return SimpleNamespace(
            structured_content=payload,
            content=[TextContent(type="text", text=json.dumps(payload))],
            is_error=False,
        )


def make_ctx(colab, connected=True):
    proxy = Mock()
    proxy.is_connected.return_value = connected
    proxy.proxy_mcp_client = colab
    proxy.wss = SimpleNamespace(disconnected=asyncio.Event())
    if not connected:
        proxy.wss.disconnected.set()
    ctx = Mock()
    ctx.get_state.side_effect = lambda k: proxy if k == ops.PROXY_CLIENT_KEY else None
    return ctx


@pytest.fixture(autouse=True)
def clear_jobs():
    ops.JOBS.clear()


def code_cell(cid, code):
    return {"cell_type": "code", "id": cid, "source": [code], "outputs": []}


class TestSummarize:
    def test_streams_results_errors_and_images(self):
        outputs = [
            {"output_type": "stream", "name": "stdout", "text": ["a\n", "b\n"]},
            {"output_type": "stream", "name": "stderr", "text": ["warn"]},
            {"output_type": "execute_result", "data": {"text/plain": ["4"]}},
            {
                "output_type": "display_data",
                "data": {"text/plain": ["<Figure>"], "image/png": "aGVsbG8="},
            },
            {
                "output_type": "error",
                "ename": "ValueError",
                "evalue": "boom",
                "traceback": ["\x1b[0;31mValueError\x1b[0m: boom"],
            },
        ]
        s = ops.summarize_outputs(outputs)
        assert s["stdout"] == "a\nb\n"
        assert s["stderr"] == "warn"
        assert s["result"] == "4\n<Figure>"
        assert s["ok"] is False
        assert s["error"]["traceback"] == "ValueError: boom"
        assert s["image_count"] == 1 and s["_images"][0]["mime"] == "image/png"

    def test_truncation_keeps_head_and_tail(self):
        s = ops.summarize_outputs(
            [{"output_type": "stream", "name": "stdout", "text": ["x" * 100 + "END"]}],
            max_chars=40,
        )
        assert s["truncated"] is True
        assert s["stdout"].endswith("END") and "truncated" in s["stdout"]


class TestRunTools:
    @pytest.mark.asyncio
    async def test_outline(self):
        colab = FakeColab(
            [
                code_cell("a", "print(1)\nprint(2)"),
                {"cell_type": "markdown", "id": "m", "source": ["# title"]},
            ]
        )
        outline = await ops.get_notebook_outline(ctx=make_ctx(colab))
        assert outline["cell_count"] == 2
        assert outline["cells"][0] == {
            "index": 0,
            "id": "a",
            "type": "code",
            "first_line": "print(1)",
            "lines": 2,
            "has_outputs": False,
            "has_error": False,
        }
        assert outline["cells"][1]["type"] == "markdown"

    @pytest.mark.asyncio
    async def test_run_cells_in_order_and_stop_on_error(self):
        colab = FakeColab(
            [
                code_cell("a", "print('one')"),
                code_cell("b", "raise ValueError('bad')"),
                code_cell("c", "print('three')"),
            ]
        )
        view = await ops.run_cells(ctx=make_ctx(colab))
        assert view["state"] == "failed"
        assert [r["cell_id"] for r in view["results"]] == ["a", "b"]
        assert view["results"][0]["stdout"] == "one\n"
        assert "ValueError" in view["error"]

    @pytest.mark.asyncio
    async def test_run_cells_continue_past_error_by_range(self):
        colab = FakeColab(
            [
                code_cell("a", "print('one')"),
                code_cell("b", "raise ValueError('bad')"),
                code_cell("c", "print('three')"),
            ]
        )
        view = await ops.run_cells(
            start_index=1, stop_on_error=False, ctx=make_ctx(colab)
        )
        assert [r["cell_id"] for r in view["results"]] == ["b", "c"]
        assert view["state"] == "failed"
        assert view["results"][1]["stdout"] == "three\n"

    @pytest.mark.asyncio
    async def test_unknown_cell_id_rejected(self):
        colab = FakeColab([code_cell("a", "1")])
        with pytest.raises(ToolError, match="Unknown cell ids"):
            await ops.run_cells(cell_ids=["nope"], ctx=make_ctx(colab))

    @pytest.mark.asyncio
    async def test_background_job_and_cancel(self):
        colab = FakeColab([code_cell(f"c{i}", f"print({i})") for i in range(5)])
        started = await ops.run_notebook(ctx=make_ctx(colab))
        assert started["state"] == "running" and started["total_cells"] == 5
        job = ops.JOBS[started["job_id"]]
        await job.task
        status = await ops.job_status(started["job_id"])
        assert status["state"] == "done" and status["completed_cells"] == 5
        assert (await ops.list_jobs())["jobs"][0]["job_id"] == started["job_id"]
        assert (await ops.cancel_job(started["job_id"]))[
            "note"
        ] == "job already finished"
        with pytest.raises(ToolError):
            await ops.job_status("missing")

    @pytest.mark.asyncio
    async def test_cancel_stops_scheduling(self):
        colab = FakeColab([code_cell(f"c{i}", f"print({i})") for i in range(3)])
        original = colab.call_tool
        gate = asyncio.Event()

        async def slow(name, args, raise_on_error=True):
            if name == "run_code_cell":
                await gate.wait()
            return await original(name, args, raise_on_error)

        colab.call_tool = slow
        started = await ops.run_cells(background=True, ctx=make_ctx(colab))
        await asyncio.sleep(0)
        await ops.cancel_job(started["job_id"])
        gate.set()
        await ops.JOBS[started["job_id"]].task
        status = await ops.job_status(started["job_id"])
        assert status["state"] == "cancelled"
        assert status["completed_cells"] == 1

    @pytest.mark.asyncio
    async def test_run_code_temp_cell_is_removed_and_images_returned(self):
        colab = FakeColab([code_cell("a", "1")])
        result = await ops.run_code("print('hi')", ctx=make_ctx(colab))
        assert result.structured_content["stdout"] == "hi\n"
        assert [c["id"] for c in colab.cells] == ["a"]
        kept = await ops.run_code("x = 1", keep_cell=True, ctx=make_ctx(colab))
        assert kept.structured_content["cell_id"] == "cell2"
        assert [c["id"] for c in colab.cells] == ["a", "cell2"]

    @pytest.mark.asyncio
    async def test_run_code_requires_connection(self):
        with pytest.raises(ToolError, match="No Colab tab is connected"):
            await ops.run_code("1", ctx=make_ctx(FakeColab(), connected=False))


class TestTransfer:
    @pytest.mark.asyncio
    async def test_upload_download_roundtrip_in_chunks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ops, "UPLOAD_CHUNK_BYTES", 1000)
        monkeypatch.setattr(ops, "DOWNLOAD_CHUNK_BYTES", 700)
        colab = FakeColab([code_cell("a", "1")])
        data = bytes(range(256)) * 10  # 2560 bytes, not a multiple of either chunk
        src = tmp_path / "local" / "in.bin"
        src.parent.mkdir()
        src.write_bytes(data)
        remote = str(tmp_path / "runtime" / "sub" / "in.bin")

        up = await ops.upload_file(str(src), remote, ctx=make_ctx(colab))
        assert up["bytes"] == 2560 and up["chunks"] == 3
        assert open(remote, "rb").read() == data
        assert [c["id"] for c in colab.cells] == ["a"], (
            "temporary cells must be cleaned up"
        )

        dst = tmp_path / "local" / "out" / "back.bin"
        down = await ops.download_file(remote, str(dst), ctx=make_ctx(colab))
        assert down["bytes"] == 2560 and down["chunks"] == 4
        assert dst.read_bytes() == data
        assert down["sha256"] == up["sha256"]
        assert [c["id"] for c in colab.cells] == ["a"]

    @pytest.mark.asyncio
    async def test_download_refuses_overwrite_and_missing_remote(self, tmp_path):
        colab = FakeColab([])
        existing = tmp_path / "x.bin"
        existing.write_bytes(b"old")
        with pytest.raises(ToolError, match="overwrite=False"):
            await ops.download_file("/nowhere", str(existing), ctx=make_ctx(colab))
        with pytest.raises(ToolError, match="Could not hash"):
            await ops.download_file(
                str(tmp_path / "missing"),
                str(tmp_path / "new.bin"),
                ctx=make_ctx(colab),
            )

    @pytest.mark.asyncio
    async def test_upload_missing_local(self, tmp_path):
        with pytest.raises(ToolError, match="Local file not found"):
            await ops.upload_file(
                str(tmp_path / "nope"), "/content/x", ctx=make_ctx(FakeColab())
            )


class TestExport:
    @pytest.mark.asyncio
    async def test_export_writes_nbformat(self, tmp_path):
        colab = FakeColab(
            [
                code_cell("a", "print(1)"),
                {"cell_type": "markdown", "id": "m", "source": ["# hi"]},
            ]
        )
        colab.cells[0]["outputs"] = [
            {"output_type": "stream", "name": "stdout", "text": ["1\n"]}
        ]
        info = await ops.export_notebook(
            str(tmp_path / "nb" / "out.ipynb"), ctx=make_ctx(colab)
        )
        nb = json.loads((tmp_path / "nb" / "out.ipynb").read_text())
        assert info["cell_count"] == 2 and nb["nbformat"] == 4
        assert nb["cells"][0]["source"] == "print(1)" and nb["cells"][0]["outputs"][0][
            "text"
        ] == ["1\n"]
        assert (
            nb["cells"][1]["cell_type"] == "markdown"
            and "outputs" not in nb["cells"][1]
        )
