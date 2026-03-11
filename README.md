# Autoresearch Dashboard

Web-based results explorer for [karpathy/autoresearch](https://github.com/karpathy/autoresearch) experiments.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/fastapi-latest-green)

## Features

- **val_bpb chart** — scatter plot of all experiments with best-so-far trendline
- **Live loss curve** — real-time training progress, auto-refreshes every 5s
- **Experiment table** — sortable by any column, color-coded status (keep/discard/crash)
- **Git diff viewer** — click any experiment to see what changed in train.py
- **GPU monitoring** — utilization, VRAM, temperature for all GPUs

## Setup

```bash
cd /path/to/autoresearch
pip install fastapi uvicorn
python dashboard.py
```

Or with uv (if autoresearch is already set up):

```bash
uv pip install fastapi uvicorn
uv run dashboard.py
```

Open http://localhost:8080

## Configuration

Set `REPO_DIR` environment variable to point to your autoresearch repo:

```bash
REPO_DIR=/path/to/autoresearch python dashboard.py
```

Defaults to `/root/github.com/karpathy/autoresearch`.

## How it works

Single Python file, no build step. The FastAPI server exposes:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard HTML (embedded, no static files) |
| `GET /api/results` | Parsed results.tsv as JSON |
| `GET /api/live` | Current run.log training progress |
| `GET /api/diff/<commit>` | Git diff of train.py for a commit |
| `GET /api/status` | GPU info, branch, experiment count |
| `GET /api/git-log` | Recent git history |

The frontend uses Chart.js (loaded from CDN) and vanilla JS. Auto-refreshes every 5 seconds.

## License

MIT
