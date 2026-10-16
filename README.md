# Colab-mcp

An MCP server for bridging your local agent to a Colab session in the browser.

# Supported Clients
This MCP server requires client support for `notifications/tools/list_changed` and for the client to be running locally on your device. 

Popular clients that fit these criteria include:
- Gemini CLI
- Claude Code
- Windsurf


# Setup

- Install `uv` (`pip install uv`)
- Configure for usage (eg for mcp.json style services):

```
...
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
...
```

(If you have a non-standard default package index (**Googlers**), you may also need to add `--index https://pypi.org/simple`)

# Opening a specific notebook (local fork)

This fork extends `open_colab_browser_connection` with an optional `notebook_url`
argument so an agent can connect to the notebook you hand it instead of the
scratch notebook:

- Colab links (`colab.research.google.com/...`, `colab.google.com/...`)
- Google Drive share links or bare Drive file ids
- GitHub `.ipynb` blob or `raw.githubusercontent.com` links
- Gist links

Calling it with a different `notebook_url` while connected disconnects the current
tab and opens the new notebook. A `colab_connection_status` tool reports whether a
tab is connected and which notebook it was opened for, without touching the browser.
`COLAB_MCP_UI_TIMEOUT` (seconds, default 60) controls how long the opener waits for
you to accept the connection in the tab.

Run from this checkout with `uv run --project /path/to/colab-mcp colab-mcp`.

## Connection lifecycle (local fork)

- **Persistent identity.** The websocket token and port are saved to
  `~/.colab-mcp/connection.json` (owner-readable only; override the directory with
  `--state-dir` or `COLAB_MCP_STATE_DIR`) so a connected tab survives a server restart.
  `--ephemeral` restores the upstream behaviour of a fresh token and port each start,
  and `--port` pins the websocket port.
- **Automatic re-open.** If the tab drops and the agent calls a notebook tool, the server
  re-opens the last notebook and waits for the connection instead of failing. Set
  `COLAB_MCP_AUTO_REOPEN=0` to disable.
- **Cancellation-safe waits.** Upstream's wait for the browser used `gather` inside
  `wait_for`; a timeout or a cancelled tool call killed the long-lived MCP client, after
  which every tab was dropped the instant it connected. The wait no longer cancels the
  client, and a dead session is recycled before a tab is opened.
- `colab_connection_status` also reports the proxy port, the state and log directories,
  whether the session is dead, and the server pid.

## Composite notebook tools (local fork)

Colab's in-browser server exposes seven primitives (`get_cells`, `add_code_cell`,
`add_text_cell`, `update_cell`, `move_cell`, `delete_cell`, blocking `run_code_cell`).
This fork adds server-side tools built on them:

| Tool | What it does |
| --- | --- |
| `get_notebook_outline` | Cells as index, id, type, first line, output and error flags. Cheaper than `get_cells`. |
| `run_cells` | Run code cells by id list or index range, in order, with per-cell stdout/stderr/result/error. `background=True` returns a job id. |
| `run_notebook` | Run every code cell from the top; background by default. |
| `run_code` | Run ad-hoc Python through a temporary cell at the end (deleted afterwards unless `keep_cell`). Plots come back as image content. |
| `job_status`, `list_jobs`, `cancel_job` | Poll or stop background runs. Colab cannot interrupt a running cell through this bridge, so cancel stops after the current cell. |
| `upload_file` | Copy a local file to the runtime in base64 chunks, verified by SHA-256. |
| `download_file` | Copy a runtime file to this machine, verified by SHA-256. |
| `export_notebook` | Save the connected notebook as `.ipynb` (nbformat 4). |

Large outputs: the websocket bridge now accepts messages up to 64 MiB (the library
default of 1 MiB dropped the connection on a ~400 KB output). A proxied or composite
call fails fast with a clear error if the tab drops while the call is in flight.

# Issues & Discussions

We are using GitHub [discussions](https://github.com/googlecolab/colab-mcp/discussions) as the
place for issue discussion and feature requests. As discussions mature into action items, we
will add those items as issues. This helps us ensure that issues in the issue tracker are
well-understood, deduplicated, and actionable. For these reasons, **please do <u>NOT</u> open
issues directly.** 

# Contributing 
We unfortunately don't have the bandwidth to support review of external contributions, and we 
don't want user PRs to languish, so we aren't accepting any external contributions right now.

If you have a great idea or pain point, we would love to hear about it on our 
[discussions](https://github.com/googlecolab/colab-mcp/discussions) page - the preferred place 
for issue discussion and feature requests.

# Internal - For Colab Developers

### Prerequisites

- `uv` is required (`pip install uv`)
- Configure git hooks to run repo presubmits

```shell
git config core.hooksPath .githooks
```

### Gemini CLI setup

```
...
  "mcpServers": {
    "colab-mcp": {
      "command": "uv",
      "args": ["run", "colab-mcp"],
      "cwd": "/path/to/github/colab-mcp",
      "timeout": 30000
    }
  }
...
```
