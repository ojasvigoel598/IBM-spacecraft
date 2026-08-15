# MissionMind - start ALL services with one command (idempotent, detached).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_all.ps1
#
# Starts in dependency order and waits for each to answer its health check
# before moving on:
#   1. FastAPI  :8100  (backend the React console talks to)
#   2. Streamlit:8501  (main Mission Control dashboard)
#   3. Vite     :5173  (React web console - uses the BUNDLED node)
#
# If a service is already healthy it is skipped (no port clash). PIDs are
# written to .freebuff/*.pid so scripts/stop_all.ps1 can tear everything down.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
$nodeDir = Join-Path $root ".freebuff\node\node-v22.14.0-win-x64"

# --------------------------------------------------------------------------
function Test-Health([string]$url, [int]$timeoutSec = 5) {
    try {
        $h = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec $timeoutSec
        return ($h.StatusCode -eq 200)
    } catch { return $false }
}

function Start-Service {
    param([string]$Name, [string]$File, [string[]]$ArgList, [string]$WorkDir,
          [string]$PidFile, [string]$Url, [string]$HealthPath,
          [string]$LogOut, [string]$LogErr)
    $fullUrl = "$Url$HealthPath"
    if (Test-Health $fullUrl) {
        Write-Host "[$Name] already healthy on $Url - skipped"
        return $true
    }
    if (Test-Path $PidFile) {
        $old = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($old) {
            $proc = Get-Process -Id $old -ErrorAction SilentlyContinue
            if (-not $proc) { Remove-Item $PidFile -Force }
        }
    }
    $p = Start-Process -FilePath $File -ArgumentList $ArgList `
        -WorkingDirectory $WorkDir `
        -RedirectStandardOutput $LogOut -RedirectStandardError $LogErr `
        -WindowStyle Hidden -PassThru
    $p.Id | Out-File -FilePath $PidFile -Encoding ascii
    Write-Host "[$Name] started pid $($p.Id) - waiting for health..."
    for ($i = 0; $i -lt 30; $i++) {   # up to 90 s
        Start-Sleep -Seconds 3
        if (Test-Health $fullUrl) {
            Write-Host "[$Name] healthy on $Url"
            return $true
        }
    }
    Write-Host "[$Name] FAILED to become healthy within 90 s - see $LogErr"
    return $false
}

# --------------------------------------------------------------------------
function Status-Text([bool]$ok) { if ($ok) { "OK" } else { "FAILED" } }

Write-Output "=============================================="
Write-Output " MissionMind - starting all services"
Write-Output "=============================================="

# 1. FastAPI backend on :8100
$ok1 = Start-Service -Name "FastAPI" `
    -File $venvPy `
    -ArgList @("-m","uvicorn","missionmind.viz.api_server:app","--port","8100","--host","127.0.0.1") `
    -WorkDir $root `
    -PidFile (Join-Path $root ".freebuff\api.pid") `
    -Url "http://127.0.0.1:8100" -HealthPath "/api/health" `
    -LogOut (Join-Path $root ".freebuff\api.out.log") `
    -LogErr (Join-Path $root ".freebuff\api.err.log")

# 2. Streamlit dashboard on :8501
$ok2 = Start-Service -Name "Streamlit" `
    -File $venvPy `
    -ArgList @("-m","streamlit","run","missionmind/viz/app.py","--server.port","8501","--server.headless","true") `
    -WorkDir $root `
    -PidFile (Join-Path $root ".freebuff\dashboard.pid") `
    -Url "http://127.0.0.1:8501" -HealthPath "/_stcore/health" `
    -LogOut (Join-Path $root ".freebuff\dashboard.out.log") `
    -LogErr (Join-Path $root ".freebuff\dashboard.err.log")

# 3. Vite web console on :5173 (bundled node - fixes npx-not-on-PATH)
if (-not (Test-Path (Join-Path $nodeDir "node.exe"))) {
    Write-Output "[Vite] bundled node NOT found at $nodeDir - skipping (static web/dist build still available)"
    $ok3 = $false
} else {
    $env:PATH = "$nodeDir;$env:PATH"
    $ok3 = Start-Service -Name "Vite" `
        -File "cmd.exe" `
        -ArgList @("/c","npm run dev -- --port 5173 --strictPort") `
        -WorkDir (Join-Path $root "web") `
        -PidFile (Join-Path $root ".freebuff\vite.pid") `
        -Url "http://localhost:5173" -HealthPath "/" `
        -LogOut (Join-Path $root ".freebuff\vite.out.log") `
        -LogErr (Join-Path $root ".freebuff\vite.err.log")
}

Write-Output "=============================================="
Write-Output " SUMMARY"
Write-Output "  FastAPI  :8100  $(Status-Text $ok1)"
Write-Output "  Streamlit:8501  $(Status-Text $ok2)"
Write-Output "  Vite     :5173  $(Status-Text $ok3)"
Write-Output "  Dashboard: http://127.0.0.1:8501"
Write-Output "  API      : http://127.0.0.1:8100/api/health"
Write-Output "  Console  : http://127.0.0.1:5173"
Write-Output "  Stop all : powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop_all.ps1"
Write-Output "=============================================="
if ($ok1 -and $ok2 -and $ok3) { exit 0 } else { exit 1 }
