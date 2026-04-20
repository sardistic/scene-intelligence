param(
    [string]$Source = "0"
)

$ErrorActionPreference = "Stop"

$sceneExe = Join-Path $PSScriptRoot ".venv\Scripts\scene-intelligence.exe"

if (-not (Test-Path $sceneExe)) {
    & (Join-Path $PSScriptRoot "install.ps1")
}

try {
    & $sceneExe --source $Source
    if ($LASTEXITCODE -ne 0) {
        throw "Scene Intelligence exited with code $LASTEXITCODE."
    }
} catch {
    Write-Host ""
    Write-Host $_
    Read-Host "Press Enter to close"
    exit 1
}
