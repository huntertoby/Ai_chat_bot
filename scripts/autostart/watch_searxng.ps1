# Make sure Docker is up and the SearXNG container runs; keep checking. ASCII only.
# Starts Docker Desktop if needed, never changes Docker settings.
. (Join-Path $PSScriptRoot 'common.ps1')
$Tag = 'searxng'

if (Test-WatcherAlive $Tag) { Write-Log $Tag 'watcher already running, exit'; exit 0 }

Write-PidFile $Tag $PID $null $null
Write-Log $Tag ('watcher started (pid {0})' -f $PID)

function Test-Docker {
    & docker.exe info *> $null
    return ($LASTEXITCODE -eq 0)
}

function Find-DockerDesktop {
    foreach ($c in @((Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'),
                     (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'))) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    return $null
}

function Ensure-Searxng {
    if (Test-Http $SearxFetch) { Write-Log $Tag 'SearXNG already responding'; return $true }
    if (-not (Test-Docker)) {
        if (-not (Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue)) {
            $exe = Find-DockerDesktop
            if (-not $exe) { Write-Log $Tag 'Docker Desktop.exe not found'; return $false }
            Write-Log $Tag 'docker not responding, starting Docker Desktop'
            # Start it through WMI so it is NOT a child of this watcher: stop.ps1 kills the watcher's
            # process tree and must not take Docker Desktop (and the SearXNG container) down with it.
            $su = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]7 }
            $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
                -Arguments @{ CommandLine = ('"' + $exe + '"'); CurrentDirectory = (Split-Path $exe -Parent); ProcessStartupInformation = $su }
            if ($r.ReturnValue -ne 0) { Write-Log $Tag ('could not start Docker Desktop (code {0})' -f $r.ReturnValue); return $false }
        } else {
            Write-Log $Tag 'Docker Desktop is starting, waiting'
        }
        $deadline = (Get-Date).AddMinutes(5)
        $ok = $false
        while ((Get-Date) -lt $deadline) {
            if (Test-Docker) { $ok = $true; break }
            Start-Sleep -Seconds 5
        }
        if (-not $ok) { Write-Log $Tag 'docker not ready after 5 minutes'; return $false }
        Write-Log $Tag 'docker is ready'
    }
    $code = Invoke-Logged 'searxng.log' 'docker compose -f searxng\docker-compose.yml up -d' 180
    Write-Log $Tag ('docker compose up exit code {0}' -f $code)
    $deadline = (Get-Date).AddMinutes(2)
    while ((Get-Date) -lt $deadline) {
        if (Test-Http $SearxFetch) { Write-Log $Tag 'SearXNG is responding'; return $true }
        Start-Sleep -Seconds 5
    }
    Write-Log $Tag 'SearXNG did not respond within 2 minutes'
    return $false
}

[void](Ensure-Searxng)

while ($true) {
    Start-Sleep -Seconds 60
    if (-not (Test-Http $SearxFetch)) {
        Write-Log $Tag 'SearXNG not responding'
        if (-not (Request-Restart)) { Write-Log $Tag "more than $MaxRestartsPerHour redo attempts in 1 hour, giving up"; exit 2 }
        [void](Ensure-Searxng)
    }
}
