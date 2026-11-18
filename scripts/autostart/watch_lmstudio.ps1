# Make sure the LM Studio server is up and Yuki's model is loaded; keep checking. ASCII only.
# Never unloads other models, never stops the server, never touches LM Studio settings.
. (Join-Path $PSScriptRoot 'common.ps1')
$Tag = 'lmstudio'

if (Test-WatcherAlive $Tag) { Write-Log $Tag 'watcher already running, exit'; exit 0 }
if (-not (Test-Path -LiteralPath $LmsExe)) { Write-Log $Tag 'lms.exe not found, exit'; exit 1 }
if (-not $LlmModel) { Write-Log $Tag 'LLM_MODEL not set in .env, exit'; exit 1 }

Write-PidFile $Tag $PID $null $null
Write-Log $Tag ('watcher started (pid {0}), model {1}, port {2}' -f $PID, $LlmModel, $LmsPort)

function Ensure-Server {
    if (Test-ModelsEndpoint) { Write-Log $Tag 'server already answering'; return $true }
    Write-Log $Tag 'starting server'
    $code = Invoke-Logged 'lmstudio.log' ('"{0}" server start --port {1} --bind 127.0.0.1' -f $LmsExe, $LmsPort) 180
    Write-Log $Tag ('lms server start exit code {0}' -f $code)
    $deadline = (Get-Date).AddMinutes(3)
    while ((Get-Date) -lt $deadline) {
        if (Test-ModelsEndpoint) { Write-Log $Tag 'server is answering'; return $true }
        Start-Sleep -Seconds 5
    }
    Write-Log $Tag 'server did not answer within 3 minutes'
    return $false
}

function Ensure-Model {
    if (Test-ModelLoaded $LlmModel) { Write-Log $Tag 'model already loaded'; return $true }
    Write-Log $Tag 'loading model'
    $cl = '"{0}" load "{1}" --identifier "{1}" -c 16384 --parallel 1 --speculative-draft-mtp -y' -f $LmsExe, $LlmModel
    $code = Invoke-Logged 'lmstudio.log' $cl 900
    Write-Log $Tag ('lms load exit code {0}' -f $code)
    if (Test-ModelLoaded $LlmModel) { Write-Log $Tag 'model loaded'; return $true }
    Write-Log $Tag 'model is not loaded (see logs/lmstudio.log)'
    return $false
}

[void](Ensure-Server)
if (Test-ModelsEndpoint) { [void](Ensure-Model) }

while ($true) {
    Start-Sleep -Seconds 60
    if (-not (Test-ModelsEndpoint)) {
        Write-Log $Tag 'server not answering'
        if (-not (Request-Restart)) { Write-Log $Tag "more than $MaxRestartsPerHour redo attempts in 1 hour, giving up"; exit 2 }
        [void](Ensure-Server)
        if (Test-ModelsEndpoint) { [void](Ensure-Model) }
    } elseif (-not (Test-ModelLoaded $LlmModel)) {
        Write-Log $Tag 'model not loaded'
        if (-not (Request-Restart)) { Write-Log $Tag "more than $MaxRestartsPerHour redo attempts in 1 hour, giving up"; exit 2 }
        [void](Ensure-Model)
    }
}
