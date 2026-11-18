# Create the Startup-folder shortcut for the current user (re-runnable). ASCII only; touches nothing else.
$name = [regex]::Unescape('Yuki 自動啟動.lnk')   # Chinese name, kept ASCII-escaped for PS 5.1
$startup = [Environment]::GetFolderPath('Startup')
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$lnk = Join-Path $startup $name
$sh = New-Object -ComObject WScript.Shell
$s = $sh.CreateShortcut($lnk)
$s.TargetPath = Join-Path $env:WINDIR 'System32\wscript.exe'
$s.Arguments = '"' + (Join-Path $PSScriptRoot 'launch.vbs') + '"'
$s.WorkingDirectory = $root
$s.WindowStyle = 7
$s.Description = 'Start Yuki and BreezyVoice at login'
$s.Save()
Write-Host "Installed: $lnk"
