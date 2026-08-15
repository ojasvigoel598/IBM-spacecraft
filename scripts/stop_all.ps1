# MissionMind - stop all services on :8100 / :8501 / :5173.
# Kills by LISTENING PORT (robust even if pid files are stale/missing or the
# real listener is a child of a cmd.exe wrapper), then removes pid files.
$ErrorActionPreference = "SilentlyContinue"
$ports = 8100, 8501, 5173

foreach ($port in $ports) {
    $pids = @()
    $lines = netstat -ano | Select-String "LISTENING" | Select-String ":$port\s"
    foreach ($line in $lines) {
        $parts = $line.ToString().Trim() -split '\s+'
        $pidVal = $parts[-1]
        if ($pidVal -match '^\d+$') { $pids += [int]$pidVal }
    }
    foreach ($id in ($pids | Select-Object -Unique)) {
        $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
            Write-Output "stopped pid $id (port $port)"
        }
    }
}

# Clean up pid files regardless.
foreach ($f in @("api.pid", "dashboard.pid", "vite.pid", "web-api.pid", "web-vite.pid")) {
    $p = Join-Path (Split-Path -Parent $PSScriptRoot) ".freebuff\$f"
    Remove-Item $p -Force -ErrorAction SilentlyContinue
}
Write-Output "done"
