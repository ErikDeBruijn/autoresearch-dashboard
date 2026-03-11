"""Autoresearch Dashboard — FastAPI server with embedded frontend."""
import asyncio
import base64
import csv
import datetime
import io
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import requests as http_requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

REPO_DIR = Path(os.environ.get("REPO_DIR", "/root/github.com/karpathy/autoresearch"))
RESULTS_TSV = REPO_DIR / "results.tsv"
RUN_LOG = REPO_DIR / "run.log"
GITHUB_REPO = os.environ.get("GITHUB_REPO", "karpathy/autoresearch")


# ---------------------------------------------------------------------------
# Fork Scanner
# ---------------------------------------------------------------------------

class ForkScanner:
    """Scans GitHub forks of the upstream repo and caches metadata."""

    def __init__(self, upstream_repo: str):
        self.upstream = upstream_repo
        self.forks: list[dict] = []
        self.last_scan: str | None = None
        self.scanning = False

    def scan(self):
        if self.scanning:
            return
        self.scanning = True
        try:
            self._do_scan()
        finally:
            self.scanning = False

    def _do_scan(self):
        all_forks: list[dict] = []
        page = 1
        while page <= 10:
            try:
                resp = http_requests.get(
                    f"https://api.github.com/repos/{self.upstream}/forks",
                    params={"page": page, "per_page": 100, "sort": "newest"},
                    headers={"Accept": "application/vnd.github.v3+json"},
                    timeout=30,
                )
            except Exception:
                break
            if resp.status_code != 200:
                break
            remaining = int(resp.headers.get("X-RateLimit-Remaining", 60))
            data = resp.json()
            if not data:
                break
            all_forks.extend(data)
            page += 1
            if len(data) < 100 or remaining < 10:
                break

        now = datetime.datetime.now(datetime.timezone.utc)
        processed = []
        enrich_count = 0

        for fork in all_forks:
            try:
                pushed = datetime.datetime.fromisoformat(
                    fork["pushed_at"].replace("Z", "+00:00")
                )
            except Exception:
                pushed = now
            age_seconds = (now - pushed).total_seconds()

            entry = {
                "full_name": fork["full_name"],
                "owner": fork["owner"]["login"],
                "repo": fork["name"],
                "html_url": fork["html_url"],
                "pushed_at": fork["pushed_at"],
                "stargazers_count": fork.get("stargazers_count", 0),
                "description": fork.get("description") or "",
                "default_branch": fork.get("default_branch", "master"),
                "age_seconds": age_seconds,
            }

            # Enrich recent forks (< 7 days), max 15 to stay within rate limits
            if age_seconds < 7 * 86400 and enrich_count < 15:
                self._enrich_fork(entry, fork)
                enrich_count += 1

            processed.append(entry)

        processed.sort(key=lambda x: x["pushed_at"], reverse=True)
        self.forks = processed
        self.last_scan = now.isoformat()

    def _enrich_fork(self, entry: dict, fork_data: dict):
        owner = fork_data["owner"]["login"]
        default_branch = fork_data.get("default_branch", "master")

        # Compare with upstream
        try:
            resp = http_requests.get(
                f"https://api.github.com/repos/{self.upstream}/compare/master...{owner}:{default_branch}",
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=30,
            )
            if resp.status_code == 200:
                compare = resp.json()
                files = compare.get("files", [])
                entry["additions"] = sum(f.get("additions", 0) for f in files)
                entry["deletions"] = sum(f.get("deletions", 0) for f in files)
                entry["changed_files"] = len(files)
                entry["train_py_changed"] = any(
                    f["filename"] == "train.py" for f in files
                )
        except Exception:
            pass

        # Check results.tsv and parse best val_bpb
        try:
            resp = http_requests.get(
                f"https://api.github.com/repos/{entry['full_name']}/contents/results.tsv",
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=15,
            )
            if resp.status_code == 200:
                entry["has_results"] = True
                content = resp.json().get("content", "")
                decoded = base64.b64decode(content).decode("utf-8")
                reader = csv.DictReader(io.StringIO(decoded), delimiter="\t")
                best_bpb = None
                for row in reader:
                    try:
                        bpb = float(row.get("val_bpb", 0))
                        if bpb > 0 and (best_bpb is None or bpb < best_bpb):
                            best_bpb = bpb
                    except (ValueError, TypeError):
                        pass
                if best_bpb is not None:
                    entry["best_val_bpb"] = best_bpb
        except Exception:
            pass


fork_scanner = ForkScanner(GITHUB_REPO)


async def periodic_scan():
    await asyncio.sleep(10)
    await asyncio.to_thread(fork_scanner.scan)
    while True:
        await asyncio.sleep(3600)
        await asyncio.to_thread(fork_scanner.scan)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(periodic_scan())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Existing API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/results")
def get_results():
    if not RESULTS_TSV.exists():
        return []
    rows = []
    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                r["val_bpb"] = float(r.get("val_bpb", 0))
                r["memory_gb"] = float(r.get("memory_gb", 0))
            except (ValueError, TypeError):
                pass
            rows.append(r)
    return rows


@app.get("/api/live")
def get_live():
    if not RUN_LOG.exists():
        return {"steps": [], "summary": None, "running": False}

    text = RUN_LOG.read_text()
    step_pattern = re.compile(
        r"step (\d+) \(([\d.]+)%\) \| loss: ([\d.]+) \| lrm: ([\d.]+) \| "
        r"dt: (\d+)ms \| tok/sec: ([\d,]+) \| mfu: ([\d.]+)% \| epoch: (\d+) \| remaining: (\d+)s"
    )
    steps = []
    for m in step_pattern.finditer(text):
        steps.append({
            "step": int(m.group(1)),
            "progress": float(m.group(2)),
            "loss": float(m.group(3)),
            "lrm": float(m.group(4)),
            "dt_ms": int(m.group(5)),
            "tok_sec": int(m.group(6).replace(",", "")),
            "mfu": float(m.group(7)),
            "epoch": int(m.group(8)),
            "remaining": int(m.group(9)),
        })

    summary = None
    summary_pattern = re.compile(r"^(val_bpb|training_seconds|peak_vram_mb|mfu_percent|total_tokens_M|num_steps|num_params_M|depth):\s+(.+)$", re.MULTILINE)
    matches = dict(summary_pattern.findall(text))
    if matches:
        summary = {k: float(v) for k, v in matches.items()}

    try:
        result = subprocess.run(["pgrep", "-f", "uv run train.py"], capture_output=True, timeout=5)
        running = result.returncode == 0
    except Exception:
        running = False

    return {"steps": steps, "summary": summary, "running": running}


@app.get("/api/diff/{commit}")
def get_diff(commit: str):
    if not re.match(r"^[a-f0-9]{7,40}$", commit):
        return JSONResponse({"error": "invalid commit"}, status_code=400)
    try:
        result = subprocess.run(
            ["git", "diff", f"{commit}~1", commit, "--", "train.py"],
            capture_output=True, text=True, cwd=REPO_DIR, timeout=10
        )
        return {"commit": commit, "diff": result.stdout or "(no changes to train.py)"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/status")
def get_status():
    gpu_info = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "PATH": "/opt/nvidia-libs:/usr/local/bin:/usr/bin:/bin"}
        )
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpu_info.append({
                    "index": int(parts[0]), "name": parts[1],
                    "util_pct": int(parts[2]), "mem_used_mb": int(parts[3]),
                    "mem_total_mb": int(parts[4]), "temp_c": int(parts[5]),
                })
    except Exception:
        pass

    branch = ""
    try:
        result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=REPO_DIR, timeout=5)
        branch = result.stdout.strip()
    except Exception:
        pass

    exp_count = 0
    if RESULTS_TSV.exists():
        exp_count = max(0, sum(1 for _ in open(RESULTS_TSV)) - 1)

    return {
        "gpus": gpu_info,
        "branch": branch,
        "experiment_count": exp_count,
        "github_repo": GITHUB_REPO,
        "fork_count": len(fork_scanner.forks),
        "last_scan": fork_scanner.last_scan,
        "scanning": fork_scanner.scanning,
    }


@app.get("/api/git-log")
def get_git_log():
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-50"],
            capture_output=True, text=True, cwd=REPO_DIR, timeout=10
        )
        commits = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(" ", 1)
                commits.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})
        return commits
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Fork API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/forks")
def get_forks():
    return {
        "forks": fork_scanner.forks,
        "last_scan": fork_scanner.last_scan,
        "scanning": fork_scanner.scanning,
        "total": len(fork_scanner.forks),
    }


@app.get("/api/forks/{owner}/{repo}/diff")
def get_fork_diff(owner: str, repo: str):
    if not re.match(r"^[\w.-]+$", owner) or not re.match(r"^[\w.-]+$", repo):
        return JSONResponse({"error": "invalid owner/repo"}, status_code=400)

    # Find default_branch from cached data
    fork_entry = next(
        (f for f in fork_scanner.forks if f["owner"] == owner and f["repo"] == repo),
        None,
    )
    ref = fork_entry["default_branch"] if fork_entry else "master"

    try:
        resp = http_requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/compare/master...{owner}:{ref}",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=30,
        )
        if resp.status_code != 200:
            return JSONResponse(
                {"error": f"GitHub API returned {resp.status_code}"},
                status_code=502,
            )
        data = resp.json()
        train_py_patch = None
        for f in data.get("files", []):
            if f["filename"] == "train.py":
                train_py_patch = f.get("patch", "(no patch available)")
                break
        return {
            "owner": owner,
            "repo": repo,
            "diff": train_py_patch or "(no changes to train.py)",
            "total_commits": data.get("total_commits", 0),
            "compare_url": data.get("html_url", ""),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/forks/scan")
async def trigger_scan():
    if fork_scanner.scanning:
        return {"status": "already_scanning"}
    asyncio.create_task(asyncio.to_thread(fork_scanner.scan))
    return {"status": "scan_started"}


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autoresearch Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --border: #30363d;
    --text: #e6edf3; --text2: #8b949e; --green: #3fb950; --red: #f85149;
    --blue: #58a6ff; --orange: #d29922; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; }
  .header { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
  .header h1 { font-size: 16px; font-weight: 600; }
  .header .meta { display: flex; gap: 16px; align-items: center; font-size: 13px; color: var(--text2); flex-wrap: wrap; }
  .gpu-badge { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }
  .gpu-badge.active { background: rgba(63,185,80,0.15); color: var(--green); }
  .gpu-badge.idle { background: rgba(139,148,158,0.15); color: var(--text2); }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .status-dot.running { background: var(--green); animation: pulse 2s infinite; }
  .status-dot.idle { background: var(--text2); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); }
  .chart-panel { background: var(--bg); padding: 16px; }
  .chart-panel h2 { font-size: 13px; color: var(--text2); margin-bottom: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
  canvas { width: 100% !important; height: 280px !important; }

  /* Tabs */
  .tabs { display: flex; gap: 0; background: var(--bg2); border-bottom: 1px solid var(--border); }
  .tab-btn { background: none; border: none; color: var(--text2); padding: 10px 20px; font-size: 13px; cursor: pointer; border-bottom: 2px solid transparent; font-weight: 500; font-family: inherit; }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { color: var(--text); border-bottom-color: var(--blue); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .table-section { padding: 0; }
  .table-section h2 { font-size: 13px; color: var(--text2); padding: 16px 20px 8px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; padding: 8px 12px; font-size: 12px; color: var(--text2); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); cursor: pointer; user-select: none; }
  th:hover { color: var(--text); }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
  tr:hover td { background: var(--bg2); }
  tr.selected td { background: var(--bg3); }
  .status-keep { color: var(--green); }
  .status-discard { color: var(--text2); }
  .status-crash { color: var(--red); }
  .val-improved { color: var(--green); font-weight: 600; }
  .commit-hash { font-family: 'SF Mono', 'Fira Code', monospace; color: var(--blue); text-decoration: none; }
  .commit-hash:hover { text-decoration: underline; }

  /* Fork-specific */
  .fork-link { color: var(--blue); text-decoration: none; }
  .fork-link:hover { text-decoration: underline; }
  .badge-train { background: rgba(88,166,255,0.15); color: var(--blue); padding: 1px 6px; border-radius: 8px; font-size: 11px; margin-left: 6px; }
  .changes-add { color: var(--green); }
  .changes-del { color: var(--red); }
  .btn-small { background: var(--bg3); border: 1px solid var(--border); color: var(--text2); padding: 3px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; text-decoration: none; display: inline-block; font-family: inherit; }
  .btn-small:hover { color: var(--text); border-color: var(--text2); }
  .btn-small:disabled { opacity: 0.5; cursor: default; }
  .forks-header { display: flex; align-items: center; justify-content: space-between; padding: 12px 20px 8px; }
  .forks-header h2 { font-size: 13px; color: var(--text2); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin: 0; padding: 0; }
  .scan-info { font-size: 12px; color: var(--text2); display: flex; align-items: center; gap: 10px; }

  .diff-panel { background: var(--bg2); border-top: 1px solid var(--border); padding: 16px 20px; display: none; }
  .diff-panel.visible { display: block; }
  .diff-panel h2 { font-size: 13px; color: var(--text2); margin-bottom: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
  .diff-header { display: flex; align-items: center; gap: 12px; }
  .diff-content { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; line-height: 1.6; white-space: pre; overflow-x: auto; max-height: 400px; overflow-y: auto; padding: 12px; background: var(--bg); border-radius: 6px; border: 1px solid var(--border); }
  .diff-content .add { color: var(--green); }
  .diff-content .del { color: var(--red); }
  .diff-content .hunk { color: var(--purple); }

  .empty-state { color: var(--text2); text-align: center; padding: 40px; font-style: italic; }
  @media (max-width: 800px) { .charts { grid-template-columns: 1fr; } }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body>

<div class="header">
  <h1>Autoresearch Dashboard</h1>
  <div class="meta">
    <span id="status-indicator"><span class="status-dot idle"></span>Checking...</span>
    <span id="branch-info"></span>
    <span id="exp-count"></span>
    <span id="fork-count-badge"></span>
    <span id="gpu-badges"></span>
  </div>
</div>

<div class="charts">
  <div class="chart-panel">
    <h2>val_bpb over experiments</h2>
    <canvas id="bpbChart"></canvas>
  </div>
  <div class="chart-panel">
    <h2>Live training — loss curve</h2>
    <canvas id="lossChart"></canvas>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('experiments')">Experiments</button>
  <button class="tab-btn" onclick="switchTab('forks')" id="forks-tab-btn">Forks</button>
</div>

<div id="tab-experiments" class="tab-content active">
  <div class="table-section">
    <table id="results-table">
      <thead>
        <tr>
          <th data-col="idx">#</th>
          <th data-col="commit">Commit</th>
          <th data-col="val_bpb">val_bpb</th>
          <th data-col="memory_gb">VRAM (GB)</th>
          <th data-col="status">Status</th>
          <th data-col="description">Description</th>
        </tr>
      </thead>
      <tbody id="results-body"></tbody>
    </table>
    <div id="empty-results" class="empty-state">No experiments yet — waiting for results...</div>
  </div>
</div>

<div id="tab-forks" class="tab-content">
  <div class="forks-header">
    <h2>Fork Ecosystem</h2>
    <div class="scan-info">
      <span id="last-scan-time"></span>
      <button class="btn-small" onclick="triggerScan()" id="scan-btn">Scan now</button>
    </div>
  </div>
  <div class="table-section">
    <table id="forks-table">
      <thead>
        <tr>
          <th data-fcol="full_name">Fork</th>
          <th data-fcol="pushed_at">Last push</th>
          <th data-fcol="stargazers_count">Stars</th>
          <th data-fcol="changes">Changes</th>
          <th data-fcol="best_val_bpb">Best val_bpb</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody id="forks-body"></tbody>
    </table>
    <div id="empty-forks" class="empty-state">No forks scanned yet — first scan runs shortly after startup...</div>
  </div>
</div>

<div class="diff-panel" id="diff-panel">
  <h2 class="diff-header">
    <span>Diff — <span id="diff-commit"></span></span>
    <span id="diff-github-link"></span>
  </h2>
  <div class="diff-content" id="diff-content"></div>
</div>

<script>
const REFRESH_INTERVAL = 5000;
const FORK_REFRESH_INTERVAL = 60000;
const GITHUB_REPO = '__GITHUB_REPO__';
let bpbChart, lossChart;
let results = [];
let forksData = { forks: [], last_scan: null, scanning: false, total: 0 };
let sortCol = null, sortAsc = true;
let forkSortCol = null, forkSortAsc = true;
let activeTab = 'experiments';

// --- Tabs ---
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector(`.tab-btn[onclick*="${tab}"]`).classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
  // Hide diff panel on tab switch
  document.getElementById('diff-panel').classList.remove('visible');
}

// --- Charts ---
function initCharts() {
  const baseOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', font: { size: 11 } } },
      y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', font: { size: 11 } } }
    }
  };

  bpbChart = new Chart(document.getElementById('bpbChart'), {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      ...baseOpts,
      scales: {
        ...baseOpts.scales,
        x: { ...baseOpts.scales.x, title: { display: true, text: 'Experiment #', color: '#8b949e' } },
        y: { ...baseOpts.scales.y, title: { display: true, text: 'val_bpb', color: '#8b949e' }, reverse: false }
      },
      plugins: {
        ...baseOpts.plugins,
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const r = results[ctx.dataIndex];
              return r ? `${r.description} — ${r.val_bpb}` : '';
            }
          }
        }
      }
    }
  });

  lossChart = new Chart(document.getElementById('lossChart'), {
    type: 'line',
    data: { labels: [], datasets: [{ data: [], borderColor: '#58a6ff', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.1 }] },
    options: {
      ...baseOpts,
      scales: {
        ...baseOpts.scales,
        x: { ...baseOpts.scales.x, title: { display: true, text: 'Step', color: '#8b949e' } },
        y: { ...baseOpts.scales.y, title: { display: true, text: 'Loss', color: '#8b949e' } }
      },
      animation: false
    }
  });
}

function updateBpbChart() {
  if (!results.length) return;
  const keep = [], discard = [], crash = [];
  const bestLine = [];
  let bestBpb = Infinity;

  results.forEach((r, i) => {
    const point = { x: i + 1, y: r.val_bpb };
    if (r.status === 'keep') { keep.push(point); if (r.val_bpb < bestBpb) bestBpb = r.val_bpb; }
    else if (r.status === 'crash') crash.push({ x: i + 1, y: r.val_bpb || null });
    else discard.push(point);
    bestLine.push({ x: i + 1, y: bestBpb === Infinity ? null : bestBpb });
  });

  bpbChart.data.datasets = [
    { label: 'Best', data: bestLine, type: 'line', borderColor: '#3fb950', borderWidth: 1.5, borderDash: [4,4], pointRadius: 0, fill: false },
    { label: 'Keep', data: keep, backgroundColor: '#3fb950', pointRadius: 5, pointHoverRadius: 7 },
    { label: 'Discard', data: discard, backgroundColor: '#8b949e', pointRadius: 4, pointHoverRadius: 6 },
    { label: 'Crash', data: crash.filter(p => p.y), backgroundColor: '#f85149', pointRadius: 4, pointStyle: 'crossRot' },
  ];
  bpbChart.update();
}

function updateLossChart(steps) {
  if (!steps.length) return;
  const maxPoints = 300;
  const stride = Math.max(1, Math.floor(steps.length / maxPoints));
  const sampled = steps.filter((_, i) => i % stride === 0 || i === steps.length - 1);

  lossChart.data.labels = sampled.map(s => s.step);
  lossChart.data.datasets[0].data = sampled.map(s => s.loss);
  lossChart.update();
}

// --- Experiments Table ---
function renderTable() {
  const tbody = document.getElementById('results-body');
  const empty = document.getElementById('empty-results');

  if (!results.length) { tbody.innerHTML = ''; empty.style.display = 'block'; return; }
  empty.style.display = 'none';

  let sorted = results.map((r, i) => ({ ...r, idx: i + 1 }));
  if (sortCol) {
    sorted.sort((a, b) => {
      let va = a[sortCol], vb = b[sortCol];
      if (typeof va === 'number' && typeof vb === 'number') return sortAsc ? va - vb : vb - va;
      return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });
  }

  const bestBpb = Math.min(...results.filter(r => r.status === 'keep' && r.val_bpb > 0).map(r => r.val_bpb));

  tbody.innerHTML = sorted.map(r => {
    const commitLink = r.commit
      ? `<a class="commit-hash" href="https://github.com/${GITHUB_REPO}/commit/${r.commit}" target="_blank" onclick="event.stopPropagation()">${r.commit}</a>`
      : '-';
    return `
    <tr data-commit="${r.commit}" onclick="showDiff('${r.commit}', this)">
      <td>${r.idx}</td>
      <td>${commitLink}</td>
      <td class="${r.val_bpb === bestBpb && r.status === 'keep' ? 'val-improved' : ''}">${r.val_bpb ? r.val_bpb.toFixed(6) : '-'}</td>
      <td>${r.memory_gb ? r.memory_gb.toFixed(1) : '-'}</td>
      <td class="status-${r.status}">${r.status}</td>
      <td>${r.description || ''}</td>
    </tr>`;
  }).join('');
}

document.querySelectorAll('th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (sortCol === col) sortAsc = !sortAsc;
    else { sortCol = col; sortAsc = true; }
    renderTable();
  });
});

// --- Forks Table ---
function relativeTime(isoStr) {
  if (!isoStr) return '-';
  const diff = (Date.now() - new Date(isoStr)) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  return Math.floor(diff / 86400) + 'd ago';
}

function renderForks() {
  const tbody = document.getElementById('forks-body');
  const empty = document.getElementById('empty-forks');
  const forks = forksData.forks || [];

  // Update tab button text
  document.getElementById('forks-tab-btn').textContent = `Forks (${forks.length})`;

  // Update last scan time
  const scanEl = document.getElementById('last-scan-time');
  if (forksData.scanning) {
    scanEl.textContent = 'Scanning...';
  } else if (forksData.last_scan) {
    scanEl.textContent = 'Last scan: ' + new Date(forksData.last_scan).toLocaleTimeString();
  } else {
    scanEl.textContent = 'Not scanned yet';
  }

  if (!forks.length) { tbody.innerHTML = ''; empty.style.display = 'block'; return; }
  empty.style.display = 'none';

  let sorted = [...forks];
  if (forkSortCol) {
    sorted.sort((a, b) => {
      let va = a[forkSortCol], vb = b[forkSortCol];
      if (forkSortCol === 'changes') { va = (a.additions || 0) + (a.deletions || 0); vb = (b.additions || 0) + (b.deletions || 0); }
      if (typeof va === 'number' && typeof vb === 'number') return forkSortAsc ? va - vb : vb - va;
      return forkSortAsc ? String(va || '').localeCompare(String(vb || '')) : String(vb || '').localeCompare(String(va || ''));
    });
  }

  tbody.innerHTML = sorted.map(f => {
    const changes = f.additions !== undefined
      ? `<span class="changes-add">+${f.additions}</span> <span class="changes-del">-${f.deletions}</span> (${f.changed_files})`
      : '-';
    const bpb = f.best_val_bpb ? f.best_val_bpb.toFixed(6) : '-';
    const trainBadge = f.train_py_changed ? '<span class="badge-train">train.py</span>' : '';
    const compareUrl = `https://github.com/${GITHUB_REPO}/compare/master...${f.owner}:${f.default_branch}`;
    return `
    <tr>
      <td><a class="fork-link" href="${f.html_url}" target="_blank">${f.full_name}</a>${trainBadge}</td>
      <td>${relativeTime(f.pushed_at)}</td>
      <td>${f.stargazers_count}</td>
      <td>${changes}</td>
      <td>${bpb}</td>
      <td>
        <button class="btn-small" onclick="showForkDiff('${f.owner}', '${f.repo}')">View diff</button>
        <a class="btn-small" href="${compareUrl}" target="_blank">GitHub</a>
      </td>
    </tr>`;
  }).join('');
}

document.querySelectorAll('th[data-fcol]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.fcol;
    if (forkSortCol === col) forkSortAsc = !forkSortAsc;
    else { forkSortCol = col; forkSortAsc = true; }
    renderForks();
  });
});

async function triggerScan() {
  const btn = document.getElementById('scan-btn');
  btn.disabled = true;
  btn.textContent = 'Scanning...';
  try {
    await fetch('/api/forks/scan', { method: 'POST' });
  } catch (e) {}
  // Poll for completion
  setTimeout(async function poll() {
    try {
      const resp = await fetch('/api/forks');
      forksData = await resp.json();
      renderForks();
      if (forksData.scanning) {
        setTimeout(poll, 3000);
      } else {
        btn.disabled = false;
        btn.textContent = 'Scan now';
      }
    } catch (e) {
      btn.disabled = false;
      btn.textContent = 'Scan now';
    }
  }, 3000);
}

// --- Diff panel (experiment + fork) ---
async function showDiff(commit, row) {
  if (!commit || commit === '-') return;
  document.querySelectorAll('tr.selected').forEach(r => r.classList.remove('selected'));
  if (row) row.classList.add('selected');

  const panel = document.getElementById('diff-panel');
  const content = document.getElementById('diff-content');
  const commitEl = document.getElementById('diff-commit');
  const ghLink = document.getElementById('diff-github-link');

  commitEl.textContent = commit;
  ghLink.innerHTML = `<a class="btn-small" href="https://github.com/${GITHUB_REPO}/commit/${commit}" target="_blank">View on GitHub</a>`;
  panel.classList.add('visible');
  content.textContent = 'Loading...';

  try {
    const resp = await fetch(`/api/diff/${commit}`);
    const data = await resp.json();
    content.innerHTML = colorDiff(data.diff || data.error || 'No diff available');
  } catch (e) {
    content.textContent = 'Error loading diff';
  }
}

async function showForkDiff(owner, repo) {
  document.querySelectorAll('tr.selected').forEach(r => r.classList.remove('selected'));

  const panel = document.getElementById('diff-panel');
  const content = document.getElementById('diff-content');
  const commitEl = document.getElementById('diff-commit');
  const ghLink = document.getElementById('diff-github-link');

  commitEl.textContent = owner + '/' + repo;
  ghLink.innerHTML = '';
  panel.classList.add('visible');
  content.textContent = 'Loading...';

  try {
    const resp = await fetch(`/api/forks/${owner}/${repo}/diff`);
    const data = await resp.json();
    if (data.compare_url) {
      ghLink.innerHTML = `<a class="btn-small" href="${data.compare_url}" target="_blank">View on GitHub</a>`;
    }
    content.innerHTML = colorDiff(data.diff || data.error || 'No diff available');
  } catch (e) {
    content.textContent = 'Error loading diff';
  }
}

function colorDiff(text) {
  return text.split('\\n').map(line => {
    if (line.startsWith('@@')) return `<span class="hunk">${esc(line)}</span>`;
    if (line.startsWith('+') && !line.startsWith('+++')) return `<span class="add">${esc(line)}</span>`;
    if (line.startsWith('-') && !line.startsWith('---')) return `<span class="del">${esc(line)}</span>`;
    return esc(line);
  }).join('\\n');
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// --- Status ---
function updateStatus(status, live) {
  const indicator = document.getElementById('status-indicator');
  const running = live.running;
  const dot = running ? '<span class="status-dot running"></span>' : '<span class="status-dot idle"></span>';
  const lastStep = live.steps.length ? live.steps[live.steps.length - 1] : null;
  const statusText = running && lastStep
    ? `Training step ${lastStep.step} (${lastStep.progress.toFixed(1)}%) — ${lastStep.remaining}s left`
    : (live.summary ? `Done — val_bpb: ${live.summary.val_bpb?.toFixed(6) || '?'}` : 'Idle');
  indicator.innerHTML = dot + statusText;

  document.getElementById('branch-info').textContent = status.branch || '';
  document.getElementById('exp-count').textContent = status.experiment_count ? `${status.experiment_count} experiments` : '';

  // Fork count in header
  const forkBadge = document.getElementById('fork-count-badge');
  if (status.fork_count > 0) {
    const scanTime = status.last_scan ? 'Last scan: ' + new Date(status.last_scan).toLocaleTimeString() : '';
    forkBadge.textContent = `${status.fork_count} forks`;
    forkBadge.title = scanTime;
  } else if (status.scanning) {
    forkBadge.textContent = 'Scanning forks...';
  }

  const badges = document.getElementById('gpu-badges');
  badges.innerHTML = (status.gpus || []).map(g =>
    `<span class="gpu-badge ${g.util_pct > 10 ? 'active' : 'idle'}">GPU ${g.index}: ${g.util_pct}% · ${(g.mem_used_mb/1024).toFixed(1)}/${(g.mem_total_mb/1024).toFixed(0)}GB · ${g.temp_c}°C</span>`
  ).join(' ');
}

// --- Polling ---
async function refresh() {
  try {
    const [resResp, liveResp, statusResp] = await Promise.all([
      fetch('/api/results'), fetch('/api/live'), fetch('/api/status')
    ]);
    results = await resResp.json();
    const live = await liveResp.json();
    const status = await statusResp.json();

    updateBpbChart();
    updateLossChart(live.steps);
    renderTable();
    updateStatus(status, live);
  } catch (e) {
    console.error('Refresh error:', e);
  }
}

async function refreshForks() {
  try {
    const resp = await fetch('/api/forks');
    forksData = await resp.json();
    renderForks();
  } catch (e) {
    console.error('Fork refresh error:', e);
  }
}

initCharts();
refresh();
refreshForks();
setInterval(refresh, REFRESH_INTERVAL);
setInterval(refreshForks, FORK_REFRESH_INTERVAL);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML.replace("__GITHUB_REPO__", GITHUB_REPO)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
