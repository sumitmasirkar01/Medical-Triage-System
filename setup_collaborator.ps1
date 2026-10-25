param(
    [switch]$SkipTests,
    [switch]$RunModelSmoke
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

$launcher = "python"
$launcherArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $launcher = "py"
    $launcherArgs = @("-3.12")
}

& $launcher @launcherArgs -c "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'"
Assert-NativeSuccess "Python version check"
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & $launcher @launcherArgs -m venv .venv
    Assert-NativeSuccess "Virtual environment creation"
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip setuptools wheel
Assert-NativeSuccess "Packaging-tool installation"
& $python -m pip install -e ".[dev]"
Assert-NativeSuccess "Project installation"
& $python -m scripts.check_collaborator_setup
Assert-NativeSuccess "Artifact verification"

if (-not $SkipTests) {
    & $python -m pytest -q
    Assert-NativeSuccess "Test suite"
}
if ($RunModelSmoke) {
    & $python -m scripts.multimodal_smoke --device cpu
    Assert-NativeSuccess "Three-model smoke test"
}

Write-Host "Collaborator setup passed."
Write-Host "Run the CT research interface with:"
Write-Host "  .\.venv\Scripts\python.exe -m streamlit run app.py"
