# Start BreezyVoice (in WSL) and restart it when it exits. ASCII only.
. (Join-Path $PSScriptRoot 'common.ps1')
$Tag = 'breezy'

if (Test-WatcherAlive $Tag) { Write-Log $Tag 'watcher already running, exit'; exit 0 }
if (Test-Health) { Write-Log $Tag 'BreezyVoice already responding on 8890, exit'; exit 0 }

# wait for WSL to be usable (every 5 s, up to 3 minutes)
$deadline = (Get-Date).AddMinutes(3)
$ready = $false
while ((Get-Date) -lt $deadline) {
    & wsl.exe -d $Distro -e true *> $null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 5
}
if (-not $ready) { Write-Log $Tag "WSL ($Distro) not ready after 3 minutes, exit"; exit 1 }

Write-PidFile $Tag $PID $null $null
Write-Log $Tag ('watcher started (pid {0})' -f $PID)

# project path as seen from WSL, derived from this script's location
$drive = $Root.Substring(0, 1).ToLower()
$wslRoot = '/mnt/' + $drive + $Root.Substring(2).Replace('\', '/')
$logFile = Join-Path $LogDir 'breezy.log'
$bashCmd = "cd ~/BreezyVoice && . .venv/bin/activate && PYTHONUTF8=1 exec python $wslRoot/scripts/breezy_server.py"

while ($true) {
    # a leftover server from our previous run would hold port 8890
    & wsl.exe -d $Distro -- pkill -f breezy_server.py *> $null
    $inner = 'wsl.exe -d {0} -- bash -lc "{1}" >> "{2}" 2>&1' -f $Distro, $bashCmd, $logFile
    $p = Start-Process -FilePath 'cmd.exe' -ArgumentList ('/s /c "' + $inner + '"') `
        -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 2
    $desc = @(Get-Descendants $p.Id)
    Write-PidFile $Tag $PID $p.Id $desc
    Write-Log $Tag ('started BreezyVoice (cmd pid {0}, child pids {1})' -f $p.Id, ($desc -join ','))
    $p.WaitForExit()
    Write-Log $Tag ('BreezyVoice exited (code {0})' -f $p.ExitCode)
    foreach ($d in $desc) { if (Test-ProcAlive $d @('wsl*.exe')) { Stop-Tree $d } }
    if (-not (Request-Restart)) {
        Write-Log $Tag "more than $MaxRestartsPerHour restarts in 1 hour, giving up"; exit 2
    }
    Write-Log $Tag "restarting in $RestartDelaySec s"
    Start-Sleep -Seconds $RestartDelaySec
}
