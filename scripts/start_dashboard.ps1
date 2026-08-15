# MissionMind dashboard launcher (idempotent, detached, ASCII-only).
# If the dashboard is already healthy on :8501 this no-ops; otherwise it
# starts streamlit hidden, writes the PID to .freebuff/dashboard.pid and
# logs to .freebuff/dashboard.out.log / dashboard.err.log.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root ".freebuff\dashboard.pid"
$logOut = Join-Path $root ".freebuff\dashboard.out.log"
$logErr = Join-Path $root ".freebuff\dashboard.err.log"
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

# 1. Already healthy? No-op.
try {
    $h = Invoke-WebRequest -Uri "http://127.0.0.1:8501/_stcore/health" `
        -UseBasicParsing -TimeoutSec 5
    if ($h.StatusCode -eq 200) {
        Write-Output "dashboard already healthy on :8501 - nothing to do"
        exit 0
    }
} catch { }

# 2. Stale pid file? Remove it.
if (Test-Path $pidFile) {
    $old = Get-Content $pidFile -ErrorAction SilentlyContinue
    if ($old) {
        $proc = Get-Process -Id $old -ErrorAction SilentlyContinue
        if (-not $proc) { Remove-Item $pidFile -Force }
    }
}

# 3. Launch detached (no console window) and record the PID.
$p = Start-Process -FilePath $venvPy `
    -ArgumentList "-m","streamlit","run","missionmind/viz/app.py",`
        "--server.port","8501","--server.headless","true" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $logOut -RedirectStandardError $logErr `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File -FilePath $pidFile -Encoding ascii
Write-Output "started dashboard pid $($p.Id); log: $logOut"

# 4. Wait for health (up to 90 s).
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 3
    try {
        $h = Invoke-WebRequest -Uri "http://127.0.0.1:8501/_stcore/health" `
            -UseBasicParsing -TimeoutSec 3
        if ($h.StatusCode -eq 200) {
            Write-Output "dashboard healthy on :8501"
            exit 0
        }
    } catch { }
}
Write-Output "dashboard did not become healthy within 90 s - check $logErr"
exit 1
