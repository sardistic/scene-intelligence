@echo off
setlocal

if exist ".venv\Scripts\python.exe" goto install

where py >nul 2>nul
if %errorlevel%==0 (
  py -m venv .venv || goto error
) else (
  python -m venv .venv || goto error
)

:install
".venv\Scripts\python.exe" -m pip install --upgrade pip wheel || goto error
".venv\Scripts\python.exe" -m pip install . || goto error

echo.
echo Scene Intelligence is installed.
echo Run run-webcam.bat to start the default webcam.
exit /b 0

:error
echo.
echo Installation failed.
exit /b 1
