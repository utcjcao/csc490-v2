$ErrorActionPreference = "Stop"

function Invoke-NativeOrThrow {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Command
  )

  Invoke-Expression $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $Command"
  }
}

Write-Host "Running Rust format and lint checks..."
Invoke-NativeOrThrow "cargo fmt --all --check"
Invoke-NativeOrThrow "cargo clippy --workspace --all-targets -- -D warnings"
Invoke-NativeOrThrow "cargo test --workspace"

Write-Host "Checking Python tooling..."
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('ruff') else 1)"
if ($LASTEXITCODE -eq 0) {
  Invoke-NativeOrThrow "python -m ruff check ."
  Invoke-NativeOrThrow "python -m ruff format --check ."
} else {
  Write-Host "Skipping Ruff; install Python dev dependencies with .\\scripts\\bootstrap.ps1"
}

python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)"
if ($LASTEXITCODE -eq 0) {
  Invoke-NativeOrThrow "python -m pytest"
} else {
  Write-Host "Skipping pytest; install Python dev dependencies with .\\scripts\\bootstrap.ps1"
}
