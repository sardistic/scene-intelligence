@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\scene-intelligence.exe" (
  call install.bat
  if errorlevel 1 goto error
)

set SOURCE=%~1
if "%SOURCE%"=="" set SOURCE=0

".venv\Scripts\scene-intelligence.exe" --source %SOURCE%
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" goto runtime_error
exit /b 0

:runtime_error
echo.
echo Scene Intelligence exited with code %EXIT_CODE%.
echo The window is staying open so you can read the error.
echo If you want more detail, try run-webcam-debug.bat.
pause
exit /b %EXIT_CODE%

:error
echo.
echo Installation failed.
pause
exit /b 1
