# Register MissionMind dashboard to auto-start at Windows login (current user only,
# no admin needed). Writes a .cmd into the user's Startup folder that launches the
# dashboard hidden via the launcher script. Idempotent (re-running overwrites).
# Uses [System.IO.File]::WriteAllText because Set-Content can be denied where the
# raw .NET API succeeds.
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $root "scripts\start_dashboard.ps1"

$startup = [Environment]::GetFolderPath("Startup")
if (-not $startup) { $startup = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup" }
New-Item -ItemType Directory -Force -Path $startup | Out-Null

$cmdPath = Join-Path $startup "MissionMind Dashboard.cmd"
$content = "@echo off`r`nstart `"`" /b powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcher`"`r`n"
[System.IO.File]::WriteAllText($cmdPath, $content, [System.Text.Encoding]::ASCII)

Write-Output "Installed auto-start: $cmdPath"
Write-Output "Launcher: $launcher"
Write-Output "Remove anytime by deleting the .cmd file above."
