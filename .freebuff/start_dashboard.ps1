# Launch Streamlit dashboard detached
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

$p = Start-Process -FilePath $venvPy `
    -ArgumentList "-m","streamlit","run","missionmind/viz/app.py","--server.port=8501","--server.headless=true","--browser.gatherUsageStats=false" `
    -WorkingDirectory $root `
    -RedirectStandardOutput (Join-Path $root ".freebuff\dashboard.out.log") `
    -RedirectStandardError (Join-Path $root ".freebuff\dashboard.err.log") `
    -WindowStyle Hidden -PassThru
$p.Id | Out-File -FilePath (Join-Path $root ".freebuff\dashboard.pid") -Encoding ascii
Write-Output "Streamlit started pid $($p.Id)"