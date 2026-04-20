$ErrorActionPreference = "Stop"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonLauncher) {
        & py -m venv (Join-Path $PSScriptRoot ".venv")
    } else {
        & python -m venv (Join-Path $PSScriptRoot ".venv")
    }
}

& $venvPython -m pip install --upgrade pip wheel
& $venvPython -m pip install $PSScriptRoot

Write-Host ""
Write-Host "Scene Intelligence is installed."
Write-Host "Run .\run-webcam.ps1 to start the default webcam."
