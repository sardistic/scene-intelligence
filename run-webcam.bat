@echo off
setlocal

if not exist ".venv\Scripts\scene-intelligence.exe" (
  call install.bat || exit /b 1
)

set SOURCE=%~1
if "%SOURCE%"=="" set SOURCE=0

".venv\Scripts\scene-intelligence.exe" --source %SOURCE%
