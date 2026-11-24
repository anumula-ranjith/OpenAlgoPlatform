[CmdletBinding()]
param(
    [ValidateSet('Start', 'Stop', 'Status')]
    [string]$Action = 'Status',
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$applicationPath = Join-Path $projectRoot 'app.py'
$stateDirectory = Join-Path $projectRoot 'tmp'
$statePath = Join-Path $stateDirectory 'local-openalgo-process.json'
$stdoutPath = Join-Path $projectRoot 'log\local-openalgo.stdout.log'
$stderrPath = Join-Path $projectRoot 'log\local-openalgo.stderr.log'

function Get-LocalSetting([string]$Name) {
    $pattern = '^\s*' + [regex]::Escape($Name) + '\s*=\s*[''\"]?([^''\"#\s]+)'
    $match = Select-String -LiteralPath (Join-Path $projectRoot '.env') -Pattern $pattern | Select-Object -Last 1
    if (-not $match) { throw "Missing setting: $Name" }
    return $match.Matches[0].Groups[1].Value
}

function Get-OwnedProcess {
    if (-not (Test-Path -LiteralPath $statePath)) { return $null }
    $saved = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$saved.ProcessId)" -ErrorAction SilentlyContinue
    if (-not $processInfo) { return $null }
    if ($processInfo.CommandLine -notlike ('*' + $applicationPath + '*')) { return $null }
    if ($processInfo.CreationDate.ToUniversalTime().Ticks.ToString() -ne $saved.CreationTicks) { return $null }
    return $processInfo
}

function Get-LocalHealth {
    try {
        $health = Invoke-RestMethod -Uri "$baseUrl/health/check" -TimeoutSec 3
        return ($health.serviceId -eq 'openalgo' -and $health.status -in @('pass', 'warn'))
    } catch { return $false }
}

try {
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Dependencies are missing. Run: python -m uv sync --locked' }
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) { throw 'The local .env is missing. Restore the original .env if you already have an account.' }
    $webPort = [int](Get-LocalSetting 'FLASK_PORT')
    $websocketPort = [int](Get-LocalSetting 'WEBSOCKET_PORT')
    $zmqPort = [int](Get-LocalSetting 'ZMQ_PORT')
    $baseUrl = "http://127.0.0.1:$webPort"
    $ownedProcess = Get-OwnedProcess

    if ($Action -eq 'Stop') {
        if (-not $ownedProcess) { Write-Output 'No server started by this launcher is running.'; exit 0 }
        # Limit shutdown to the saved process, its creation time and this app.py.
        # Windows virtualenv redirectors may have one child interpreter.
        $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($ownedProcess.ProcessId)")
        foreach ($child in $children) {
            if ($child.CommandLine -like ('*' + $applicationPath + '*')) {
                Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
            }
        }
        Stop-Process -Id $ownedProcess.ProcessId -ErrorAction SilentlyContinue
        Write-Output 'Stopped this local OpenAlgo server.'
        exit 0
    }

    if ($Action -eq 'Status') {
        if (Get-LocalHealth) { Write-Output "OpenAlgo is responding at $baseUrl"; exit 0 }
        throw "OpenAlgo is not ready at $baseUrl. See $stderrPath"
    }

    if ($ownedProcess -and (Get-LocalHealth)) {
        Write-Output "OpenAlgo is already running at $baseUrl"
        if (-not $NoBrowser) { Start-Process $baseUrl }
        exit 0
    }
    if ($ownedProcess) { throw 'The local server is still starting or unhealthy. Check its logs before restarting.' }
    foreach ($port in @($webPort, $websocketPort, $zmqPort)) {
        if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
            throw "Port $port is already occupied. The existing process was left running."
        }
    }
    New-Item -ItemType Directory -Force -Path $stateDirectory, (Join-Path $projectRoot 'log') | Out-Null
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $serverProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-u', ('"' + $applicationPath + '"')) -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($serverProcess.Id)"
    if (-not $processInfo) { throw "OpenAlgo exited during startup. See $stderrPath" }
    @{
        ProcessId = $serverProcess.Id
        CreationTicks = $processInfo.CreationDate.ToUniversalTime().Ticks.ToString()
        Application = $applicationPath
    } | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    Write-Output 'Starting OpenAlgo. The first startup may take a few minutes.'
    $deadline = (Get-Date).AddMinutes(4)
    do {
        if (Get-LocalHealth) {
            Write-Output "OpenAlgo is ready at $baseUrl"
            Write-Output "WebSocket: ws://127.0.0.1:$websocketPort"
            if (-not $NoBrowser) { Start-Process $baseUrl }
            exit 0
        }
        $serverProcess.Refresh()
        if ($serverProcess.HasExited) { throw "OpenAlgo exited. See $stderrPath" }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    throw "Startup is taking longer than expected. See $stderrPath. The process was left running."
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
