# Stop only the processes recorded in the pid files. ASCII only.
. (Join-Path $PSScriptRoot 'common.ps1')

foreach ($Tag in @('yuki', 'breezy')) {
    $d = Read-PidFile $Tag
    if (-not $d) { Write-Log $Tag 'stop: no pid file'; continue }
    # watcher first, so it cannot restart anything
    if ($d.watcher -and (Test-WatcherAlive $Tag)) { Stop-Tree ([int]$d.watcher); Write-Log $Tag ('stop: killed watcher {0}' -f $d.watcher) }
    if ($d.child -and (Test-ProcAlive ([int]$d.child) @('cmd.exe'))) { Stop-Tree ([int]$d.child); Write-Log $Tag ('stop: killed child {0}' -f $d.child) }
    foreach ($x in @($d.extra)) {
        if ($x -and (Test-ProcAlive ([int]$x) @('python*.exe', 'wsl*.exe'))) { Stop-Tree ([int]$x); Write-Log $Tag ('stop: killed {0}' -f $x) }
    }
    Remove-Item -LiteralPath (Get-PidFilePath $Tag) -ErrorAction SilentlyContinue
}

# BreezyVoice runs inside WSL; if it survived the wsl.exe kill, clean up by name inside WSL
Start-Sleep -Seconds 2
if (Test-Health) {
    & wsl.exe -d $Distro -- pkill -f breezy_server.py *> $null
    Write-Log 'breezy' 'stop: server still answering, ran pkill in WSL'
}
Write-Log 'stop' 'done'
