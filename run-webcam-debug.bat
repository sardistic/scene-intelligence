@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  call install.bat
  if errorlevel 1 goto error
)

set SOURCE=%~1
if "%SOURCE%"=="" set SOURCE=0

echo Running Scene Intelligence in debug mode...
echo Source: %SOURCE%
echo.

".venv\Scripts\python.exe" -m scene_intelligence --source %SOURCE% --verbose
set EXIT_CODE=%ERRORLEVEL%

echo.
if not "%EXIT_CODE%"=="0" (
  echo Scene Intelligence exited with code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%

:error
echo.
echo Installation failed.
pause
exit /b 1
