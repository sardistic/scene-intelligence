param(
    [string]$Source = "0"
)

$ErrorActionPreference = "Stop"

$sceneExe = Join-Path $PSScriptRoot ".venv\Scripts\scene-intelligence.exe"

if (-not (Test-Path $sceneExe)) {
    & (Join-Path $PSScriptRoot "install.ps1")
}

& $sceneExe --source $Source
