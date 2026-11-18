' Starts the four watchers with no visible window (window style 0). Used by start.ps1 and the Startup shortcut.
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = fso.GetParentFolderName(fso.GetParentFolderName(dir))
For Each n In Array("watch_lmstudio.ps1", "watch_searxng.ps1", "watch_breezy.ps1", "watch_yuki.ps1")
  sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & dir & "\" & n & """", 0, False
Next
