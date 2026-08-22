# MissionMind — Preview run doc

## Reproduce the uncommitted artifacts a fresh checkout needs

No secrets are required to run the app (watsonx API key / project ID are optional:
without them the Granite client uses its deterministic mock fallback).

- Scenario CSVs (`data/run_*.csv`) are gitignored; `missionmind/tests/conftest.py`
  regenerates them **only when missing** (full consistent set, deterministic).
- Model artifacts (`missionmind/models/*.joblib`, `features.txt`, `dataset.json`)
  are committed — no training needed. `train.py` skips rebuild when they exist.
- `.freebuff/` infra (pid files, logs, this doc) is gitignored; recreate as needed.

## Run the server

The **active preview** is the Vite web console (port 5173) with the FastAPI
backend (port 8100) — no Streamlit dashboard. To start both:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .freebuff/start_web_preview.ps1
```

This script starts FastAPI first (port 8100), waits for `/api/health`, then starts
Vite (port 5173) using the bundled node at `.freebuff\node\node-v22.14.0-win-x64`.
Each service gets its own pid file:

| Service | Port | PID file | Health |
|---|---|---|---|
| FastAPI backend | 8100 | `.freebuff/api.pid` | `http://127.0.0.1:8100/api/health` |
| Vite web console | 5173 | `.freebuff/vite.pid` | `http://localhost:5173` |
| Streamlit dashboard | 8501 | `.freebuff/dashboard.pid` | `http://127.0.0.1:8501/_stcore/health` |

Cold starts take ~2–3 min per service (health-wait loops). The script is idempotent —
healthy services are skipped.

To start **everything** (including Streamlit):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_all.ps1
```

### Re-register the Preview (required after every Freebuff restart)

1. Confirm the Vite listener:
   ```powershell
   powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5173 -State Listen | Select-Object OwningProcess"
   ```
2. Get the pid: above command returns OwningProcess — register **that** pid, not the
   one in `.freebuff/vite.pid` (the pid file records the npm launcher, but its node
   child owns the socket).
3. Call `register_preview` with `url: http://localhost:5173` and that pid.
   Note: Vite may bind IPv6-only on Windows (`::1:5173`), so use `localhost`
   rather than `127.0.0.1`.
4. Verify with `preview_snapshot`.

If either port is dead, rerun `.freebuff/start_web_preview.ps1` (idempotent, self-heals).

### Stop

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop_all.ps1
```

## Ports

- Vite web console (main preview): http://localhost:5173
- FastAPI backend: http://127.0.0.1:8100/api/health
- Streamlit dashboard: http://127.0.0.1:8501 (optional, not started by default)