# Remove only the Startup-folder shortcut. ASCII only.
$name = [regex]::Unescape('Yuki \u81EA\u52D5\u555F\u52D5.lnk')
$lnk = Join-Path ([Environment]::GetFolderPath('Startup')) $name
if (Test-Path -LiteralPath $lnk) { Remove-Item -LiteralPath $lnk; Write-Host "Removed: $lnk" } else { Write-Host 'Shortcut not found, nothing to do.' }
