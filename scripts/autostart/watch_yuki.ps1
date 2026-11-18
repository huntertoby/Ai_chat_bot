# Start Yuki and restart it when it exits. ASCII only.
. (Join-Path $PSScriptRoot 'common.ps1')
$Tag = 'yuki'

if (Test-WatcherAlive $Tag) { Write-Log $Tag 'watcher already running, exit'; exit 0 }

# Is Yuki of THIS project already running? Match the full path of this project's main.py,
# or the project venv python running a main.py. Never match a bare "main.py".
function Get-ProjectYuki {
    $mainPath = Join-Path $Root 'main.py'
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like 'python*.exe' -and $_.CommandLine -and (
            $_.CommandLine.IndexOf($mainPath, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            ($_.ExecutablePath -and ($_.ExecutablePath -ieq $VenvPython) -and $_.CommandLine -match '(^|[\s"\\])main\.py')
        )
    }
}

$existing = @(Get-ProjectYuki)
if ($existing.Count -gt 0) {
    Write-Log $Tag ('Yuki already running (pid {0}), exit' -f ($existing[0].ProcessId)); exit 0
}
if (-not (Test-Path -LiteralPath $VenvPython)) { Write-Log $Tag "python not found: $VenvPython"; exit 1 }

Write-PidFile $Tag $PID $null $null
Write-Log $Tag ('watcher started (pid {0})' -f $PID)
$logFile = Join-Path $LogDir 'yuki.log'

# Yuki needs the LM Studio model (loaded) and SearXNG. Wait with no upper limit (D-16).
function Wait-Dependencies {
    $lastNote = Get-Date
    $first = $true
    while ($true) {
        $lmOk = Test-ModelLoaded $LlmModel
        $sxOk = Test-Http $SearxFetch
        if ($lmOk -and $sxOk) {
            if (-not $first) { Write-Log $Tag 'lmstudio model and searxng are ready' }
            return
        }
        if ($first -or ((Get-Date) - $lastNote).TotalMinutes -ge 5) {
            $w = @()
            if (-not $lmOk) { $w += 'lmstudio' }
            if (-not $sxOk) { $w += 'searxng' }
            Write-Log $Tag ('still waiting for ' + ($w -join '/'))
            $lastNote = Get-Date; $first = $false
        }
        Start-Sleep -Seconds 5
    }
}

while ($true) {
    Wait-Dependencies
    $inner = '"{0}" -u main.py >> "{1}" 2>&1' -f $VenvPython, $logFile
    $p = Start-Process -FilePath 'cmd.exe' -ArgumentList ('/s /c "' + $inner + '"') `
        -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 2
    $desc = @(Get-Descendants $p.Id)
    Write-PidFile $Tag $PID $p.Id $desc
    Write-Log $Tag ('started Yuki (cmd pid {0}, python pids {1})' -f $p.Id, ($desc -join ','))
    # track python descendants while it runs (the venv launcher may spawn a child a bit later)
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 3
        $now = @(Get-Descendants $p.Id)
        if ($now.Count -gt 0 -and (($now -join ',') -ne ($desc -join ','))) {
            $desc = $now; Write-PidFile $Tag $PID $p.Id $desc
        }
    }
    Write-Log $Tag ('Yuki exited (code {0})' -f $p.ExitCode)
    foreach ($d in $desc) { if (Test-ProcAlive $d @('python*.exe')) { Stop-Tree $d } }  # orphans of our own run
    if (-not (Request-Restart)) {
        Write-Log $Tag "more than $MaxRestartsPerHour restarts in 1 hour, giving up"; exit 2
    }
    Write-Log $Tag "restarting in $RestartDelaySec s"
    Start-Sleep -Seconds $RestartDelaySec
}
