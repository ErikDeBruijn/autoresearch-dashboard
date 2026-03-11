"""Autoresearch Dashboard — FastAPI server with embedded frontend."""
import csv
import io
import os
import re
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

REPO_DIR = Path(os.environ.get("REPO_DIR", "/root/github.com/karpathy/autoresearch"))
RESULTS_TSV = REPO_DIR / "results.tsv"
RUN_LOG = REPO_DIR / "run.log"

app = FastAPI()


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
    # Parse step lines: step 00042 (15.5%) | loss: 5.009873 | ...
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

    # Parse final summary if present
    summary = None
    summary_pattern = re.compile(r"^(val_bpb|training_seconds|peak_vram_mb|mfu_percent|total_tokens_M|num_steps|num_params_M|depth):\s+(.+)$", re.MULTILINE)
    matches = dict(summary_pattern.findall(text))
    if matches:
        summary = {k: float(v) for k, v in matches.items()}

    # Check if training is currently running
    try:
        result = subprocess.run(["pgrep", "-f", "uv run train.py"], capture_output=True, timeout=5)
        running = result.returncode == 0
    except Exception:
        running = False

    return {"steps": steps, "summary": summary, "running": running}


@app.get("/api/diff/{commit}")
def get_diff(commit: str):
    # Sanitize commit hash
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
    # GPU status
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

    # Current branch
    branch = ""
    try:
        result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, cwd=REPO_DIR, timeout=5)
        branch = result.stdout.strip()
    except Exception:
        pass

    # Experiment count from results.tsv
    exp_count = 0
    if RESULTS_TSV.exists():
        exp_count = max(0, sum(1 for _ in open(RESULTS_TSV)) - 1)

    return {"gpus": gpu_info, "branch": branch, "experiment_count": exp_count}


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
  .header { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 16px; font-weight: 600; }
  .header .meta { display: flex; gap: 16px; align-items: center; font-size: 13px; color: var(--text2); }
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
  .commit-hash { font-family: 'SF Mono', 'Fira Code', monospace; color: var(--blue); cursor: pointer; }
  .commit-hash:hover { text-decoration: underline; }

  .diff-panel { background: var(--bg2); border-top: 1px solid var(--border); padding: 16px 20px; display: none; }
  .diff-panel.visible { display: block; }
  .diff-panel h2 { font-size: 13px; color: var(--text2); margin-bottom: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
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

<div class="table-section">
  <h2>Experiments</h2>
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

<div class="diff-panel" id="diff-panel">
  <h2>Diff — <span id="diff-commit"></span></h2>
  <div class="diff-content" id="diff-content"></div>
</div>

<script>
const REFRESH_INTERVAL = 5000;
let bpbChart, lossChart;
let results = [];
let sortCol = null, sortAsc = true;

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
  // Downsample if too many points
  const maxPoints = 300;
  const stride = Math.max(1, Math.floor(steps.length / maxPoints));
  const sampled = steps.filter((_, i) => i % stride === 0 || i === steps.length - 1);

  lossChart.data.labels = sampled.map(s => s.step);
  lossChart.data.datasets[0].data = sampled.map(s => s.loss);
  lossChart.update();
}

// --- Table ---
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

  // Find best val_bpb among keeps
  const bestBpb = Math.min(...results.filter(r => r.status === 'keep' && r.val_bpb > 0).map(r => r.val_bpb));

  tbody.innerHTML = sorted.map(r => `
    <tr data-commit="${r.commit}" onclick="showDiff('${r.commit}', this)">
      <td>${r.idx}</td>
      <td><span class="commit-hash">${r.commit || '-'}</span></td>
      <td class="${r.val_bpb === bestBpb && r.status === 'keep' ? 'val-improved' : ''}">${r.val_bpb ? r.val_bpb.toFixed(6) : '-'}</td>
      <td>${r.memory_gb ? r.memory_gb.toFixed(1) : '-'}</td>
      <td class="status-${r.status}">${r.status}</td>
      <td>${r.description || ''}</td>
    </tr>
  `).join('');
}

document.querySelectorAll('th[data-col]').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (sortCol === col) sortAsc = !sortAsc;
    else { sortCol = col; sortAsc = true; }
    renderTable();
  });
});

// --- Diff ---
async function showDiff(commit, row) {
  if (!commit || commit === '-') return;
  document.querySelectorAll('tr.selected').forEach(r => r.classList.remove('selected'));
  if (row) row.classList.add('selected');

  const panel = document.getElementById('diff-panel');
  const content = document.getElementById('diff-content');
  document.getElementById('diff-commit').textContent = commit;
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

initCharts();
refresh();
setInterval(refresh, REFRESH_INTERVAL);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
