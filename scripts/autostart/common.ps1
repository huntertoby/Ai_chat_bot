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

# ---- LM Studio / SearXNG helpers (added in A1 v2) ----
$LmsExe      = Join-Path $env:USERPROFILE '.lmstudio\bin\lms.exe'
$EnvFile     = Join-Path $Root '.env'

# Read ONE value from .env (never log the whole file). Returns $Default when missing.
function Get-EnvValue([string]$Key, [string]$Default) {
    if (Test-Path -LiteralPath $EnvFile) {
        foreach ($l in (Get-Content -LiteralPath $EnvFile -Encoding UTF8)) {
            if ($l -match ('^\s*' + [regex]::Escape($Key) + '\s*=\s*(.*)$')) {
                $v = $Matches[1].Trim().Trim('"').Trim("'")
                if ($v) { return $v }
            }
        }
    }
    return $Default
}

$LlmModel     = Get-EnvValue 'LLM_MODEL' ''
$LmsBaseUrl   = (Get-EnvValue 'LMSTUDIO_BASE_URL' 'http://127.0.0.1:8001/v1').TrimEnd('/')
$LmsModelsUrl = $LmsBaseUrl + '/models'
$LmsPort      = 8001
if ($LmsBaseUrl -match ':(\d+)(/|$)') { $LmsPort = [int]$Matches[1] }
$SearxFetch   = 'http://127.0.0.1:8888/'   # where docker publishes it (searxng/docker-compose.yml)

# Any HTTP answer (even 4xx) counts as "responding"; connection errors / timeouts do not.
function Test-Http([string]$Url, [int]$TimeoutSec = 3) {
    try { $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec; return $true }
    catch { return [bool]($_.Exception.Response) }
}

function Test-ModelsEndpoint { Test-Http $LmsModelsUrl }

# /v1/models lists every downloaded model, so "loaded" must come from `lms ps`.
function Test-ModelLoaded([string]$Model) {
    if (-not $Model) { return $false }
    if (-not (Test-Path -LiteralPath $LmsExe)) {
        # no CLI: best effort, the endpoint answering with the model name
        try { return ((Invoke-WebRequest -Uri $LmsModelsUrl -UseBasicParsing -TimeoutSec 5).Content.Contains($Model)) } catch { return $false }
    }
    try {
        $txt = (& $LmsExe ps --json 2>$null) -join "`n"
        if (-not $txt) { return $false }
        foreach ($m in @($txt | ConvertFrom-Json)) {
            if ($m.identifier -eq $Model -or $m.modelKey -eq $Model -or $m.path -eq $Model) { return $true }
        }
    } catch { }
    return $false
}

# Run a command line hidden, append its output to logs\<LogName>, kill it after TimeoutSec. Returns exit code (-1 = timeout).
function Invoke-Logged([string]$LogName, [string]$CmdLine, [int]$TimeoutSec) {
    $lf = Join-Path $LogDir $LogName
    Add-Content -LiteralPath $lf -Value ('--- {0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $CmdLine.Replace($env:USERPROFILE, '~')) -Encoding UTF8
    $p = Start-Process -FilePath 'cmd.exe' -ArgumentList ('/s /c "' + $CmdLine + ' >> "' + $lf + '" 2>&1"') -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    if (-not $p.WaitForExit($TimeoutSec * 1000)) { Stop-Tree $p.Id; return -1 }
    return $p.ExitCode
}
