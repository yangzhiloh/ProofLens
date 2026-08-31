param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "verify", "preflight", "artifacts", "demo", "help")]
    [string]$Command = "help",

    [string]$PythonVersion = "3.11",
    [string]$Output = "artifacts/demo"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepositoryRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install uv 0.12.0, then rerun this command."
}

function Invoke-Uv {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & uv @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Install-LockedEnvironment {
    Invoke-Uv sync --locked --extra dev --python $PythonVersion
}

function Publish-DemoArtifacts {
    Install-LockedEnvironment
    Invoke-Uv run --locked --extra dev python scripts/reproduce_small.py `
        --output $Output --experiment e4 --publish-demo-artifacts
}

switch ($Command) {
    "setup" {
        Install-LockedEnvironment
    }
    "verify" {
        Install-LockedEnvironment
        Invoke-Uv run --locked --extra dev python -m ruff check src tests scripts
        Invoke-Uv run --locked --extra dev python -m pytest -q
        Invoke-Uv run --locked --extra dev python scripts/release_check.py --root .
    }
    "preflight" {
        Install-LockedEnvironment
        Invoke-Uv run --locked --extra dev python scripts/task8_preflight.py
    }
    "artifacts" {
        Publish-DemoArtifacts
    }
    "demo" {
        $Manifest = Join-Path $Output "export/artifact_manifest.json"
        if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
            Publish-DemoArtifacts
        }
        Invoke-Uv run --locked --extra dev python -m prooflens.cli app `
            --backend onnx `
            --model (Join-Path $Output "export/model.onnx") `
            --calibration (Join-Path $Output "export/calibration.json")
    }
    "help" {
        Write-Host "ProofLens one-click workflow"
        Write-Host "  .\scripts\prooflens.ps1 setup     # install the locked environment"
        Write-Host "  .\scripts\prooflens.ps1 verify    # lint, test, and run the release gate"
        Write-Host "  .\scripts\prooflens.ps1 preflight # audit task 8 readiness without downloads"
        Write-Host "  .\scripts\prooflens.ps1 artifacts # generate the fixture demo bundle"
        Write-Host "  .\scripts\prooflens.ps1 demo      # generate if needed, then launch the app"
    }
}
