@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'SilentlyContinue';" ^
  "$distro = if ($env:WSL_DISTRO_NAME) { $env:WSL_DISTRO_NAME } else { 'Ubuntu' };" ^
  "$timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss');" ^
  "Write-Host \"[$timestamp] Stopping Windows realtime controller...\";" ^
  "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'big_3_Windows.py' -or $_.CommandLine -match 'uv run .*big_3_Windows.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force };" ^
  "$timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss');" ^
  "Write-Host \"[$timestamp] Stopping uv/python shells...\";" ^
  "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'big_3_WSL.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force };" ^
  "$timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss');" ^
  "Write-Host \"[$timestamp] Stopping WSL helpers...\";" ^
  "& wsl.exe -d $distro -- bash -lc 'pkill -f big_3_WSL.py || true';" ^
  "$timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss');" ^
  "Write-Host \"[$timestamp] Shutdown complete.\";"
endlocal
