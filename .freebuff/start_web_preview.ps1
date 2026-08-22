# MissionMind - start web preview services (FastAPI + Vite) detached. No Streamlit.
#   powershell -NoProfile -ExecutionPolicy Bypass -File .freebuff/start_web_preview.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
$nodeDir = Join-Path $root ".freebuff\node\node-v22.14.0-win-x64"
$previewLog = Join-Path $root ".freebuff\preview-ba687057-5525-4a41-88c9-ee273cb04c9d.log"

function Test-Health([string]$url, [int]$timeoutSec = 5) {
    try {
        $h = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec $timeoutSec
        return ($h.StatusCode -eq 200)
    } catch { return $false }
}

function Start-Svc {
    param([string]$Name, [string]$File, [string[]]$ArgList, [string]$WorkDir,
          [string]$PidFile, [string]$Url, [string]$HealthPath,
          [string]$LogOut, [string]$LogErr)
    if (Test-Health "$Url$HealthPath") { Write-Output "[$Name] already healthy - skipped"; return }
    if (Test-Path $PidFile) {
        $old = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($old) {
            $proc = Get-Process -Id $old -ErrorAction SilentlyContinue
            if (-not $proc) { Remove-Item $PidFile -Force }
        }
    }
    $p = Start-Process -FilePath $File -ArgumentList $ArgList -WorkingDirectory $WorkDir `
        -RedirectStandardOutput $LogOut -RedirectStandardError $LogErr -WindowStyle Hidden -PassThru
    $p.Id | Out-File -FilePath $PidFile -Encoding ascii
    Write-Output "[$Name] started pid $($p.Id) - waiting for health..."
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 3
        if (Test-Health "$Url$HealthPath") { Write-Output "[$Name] healthy on $Url"; return }
    }
    Write-Output "[$Name] FAILED to become healthy - see $LogErr"
}

# 1. FastAPI backend :8100 — load .env so bootstrap_admin() creates the admin account
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Write-Output "[FastAPI] loading env vars from .env"
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and ($line -notmatch '^\s*#')) {
            $eq = $line.IndexOf('=')
            if ($eq -gt 0) {
                $key = $line.Substring(0, $eq).Trim()
                $val = $line.Substring($eq + 1).Trim() -replace '^"(.*)"$', '$1' -replace "^'(.*)'$", '$1'
                [Environment]::SetEnvironmentVariable($key, $val, 'Process')
            }
        }
    }
}
Start-Svc -Name "FastAPI" -File $venvPy `
    -ArgList @("-m","uvicorn","missionmind.viz.api_server:app","--port","8100","--host","127.0.0.1") `
    -WorkDir $root `
    -PidFile (Join-Path $root ".freebuff\api.pid") `
    -Url "http://127.0.0.1:8100" -HealthPath "/api/health" `
    -LogOut (Join-Path $root ".freebuff\api.out.log") `
    -LogErr (Join-Path $root ".freebuff\api.err.log")

# 2. Vite web console :5173 (bundled node - fixes npx-not-on-PATH)
if (-not (Test-Path (Join-Path $nodeDir "node.exe"))) {
    Write-Output "[Vite] bundled node missing at $nodeDir - cannot start"
    exit 1
}
$env:PATH = "$nodeDir;$env:PATH"
Start-Svc -Name "Vite" -File "cmd.exe" `
    -ArgList @("/c","npm run dev -- --port 5173 --strictPort") `
    -WorkDir (Join-Path $root "web") `
    -PidFile (Join-Path $root ".freebuff\vite.pid") `
    -Url "http://127.0.0.1:5173" -HealthPath "/" `
    -LogOut $previewLog `
    -LogErr "$previewLog.err"

Write-Output "web preview services started - API http://127.0.0.1:8100/api/health, Console http://127.0.0.1:5173"