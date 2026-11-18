# Start the four watchers hidden and return immediately. ASCII only.
Start-Process -FilePath 'wscript.exe' -ArgumentList ('"' + (Join-Path $PSScriptRoot 'launch.vbs') + '"')
