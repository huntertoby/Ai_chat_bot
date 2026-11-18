# Shared helpers for the autostart scripts (dot-sourced). ASCII only, Windows PowerShell 5.1 compatible.
$ErrorActionPreference = 'Continue'

$Root       = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$LogDir     = Join-Path $Root 'logs'
$PidDir     = Join-Path $LogDir 'autostart'
$VenvPython = Join-Path (Split-Path $Root -Parent) '.venv\Scripts\python.exe'
$Distro     = 'swift'          # WSL distro that holds ~/BreezyVoice
$HealthUrl  = 'http://127.0.0.1:8890/health'
$RestartDelaySec = 10
$MaxRestartsPerHour = 5

New-Item -ItemType Directory -Force -Path $LogDir, $PidDir | Out-Null

function Write-Log([string]$Tag, [string]$Msg) {
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Tag, $Msg
    # two watchers write this file at the same time; retry if the other one holds it
    for ($i = 0; $i -lt 10; $i++) {
        try { Add-Content -LiteralPath (Join-Path $LogDir 'autostart.log') -Value $line -Encoding UTF8 -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds (50 + (Get-Random -Maximum 100)) }
    }
}

function Get-PidFilePath([string]$Name) { Join-Path $PidDir ($Name + '.pid') }

function Read-PidFile([string]$Name) {
    $p = Get-PidFilePath $Name
    if (-not (Test-Path -LiteralPath $p)) { return $null }
    try { return (Get-Content -LiteralPath $p -Raw | ConvertFrom-Json) } catch { return $null }
}

function Write-PidFile([string]$Name, $Watcher, $Child, $Extra) {
    $o = [ordered]@{ watcher = $Watcher; child = $Child; extra = @($Extra | Where-Object { $_ }) }
    ($o | ConvertTo-Json -Compress) | Set-Content -LiteralPath (Get-PidFilePath $Name) -Encoding ASCII
}

function Get-ProcInfo([int]$ProcId) {
    if (-not $ProcId) { return $null }
    Get-CimInstance Win32_Process -Filter ("ProcessId=" + $ProcId) -ErrorAction SilentlyContinue
}

# Is the watcher recorded in the pid file alive AND really one of our watch scripts?
function Test-WatcherAlive([string]$Name) {
    $d = Read-PidFile $Name
    if (-not $d -or -not $d.watcher) { return $false }
    if ([int]$d.watcher -eq $PID) { return $false }
    $pi = Get-ProcInfo ([int]$d.watcher)
    return [bool]($pi -and $pi.CommandLine -like ('*watch_' + $Name + '.ps1*'))
}

function Get-Descendants([int]$ProcId) {
    $all = @(Get-CimInstance Win32_Process)
    $out = New-Object System.Collections.ArrayList
    $queue = New-Object System.Collections.Queue
    $queue.Enqueue($ProcId)
    while ($queue.Count -gt 0) {
        $cur = $queue.Dequeue()
        foreach ($p in $all) {
            if ($p.ParentProcessId -eq $cur -and $p.ProcessId -ne $cur) {
                [void]$out.Add([int]$p.ProcessId); $queue.Enqueue([int]$p.ProcessId)
            }
        }
    }
    return @($out)
}

function Test-Health {
    try {
        $r = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 3
        return ($r.StatusCode -eq 200)
    } catch { return $false }
}

# Is this pid alive and is its process name one of $Names (guards against pid reuse)?
function Test-ProcAlive([int]$ProcId, [string[]]$Names) {
    $pi = Get-ProcInfo $ProcId
    return [bool]($pi -and ($Names | Where-Object { $pi.Name -like $_ }))
}

function Stop-Tree([int]$ProcId) {
    & taskkill.exe /PID $ProcId /T /F *> $null
}

# Restart throttle: returns $true if a restart is allowed now (and records it).
$script:RestartTimes = New-Object System.Collections.ArrayList
function Request-Restart {
    $now = Get-Date
    $keep = @($script:RestartTimes | Where-Object { ($now - $_).TotalMinutes -lt 60 })
    $script:RestartTimes.Clear(); foreach ($t in $keep) { [void]$script:RestartTimes.Add($t) }
    if ($script:RestartTimes.Count -ge $MaxRestartsPerHour) { return $false }
    [void]$script:RestartTimes.Add($now)
    return $true
}
