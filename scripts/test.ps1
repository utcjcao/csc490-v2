$ErrorActionPreference = "Stop"

cargo test --workspace

python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)"
if ($LASTEXITCODE -eq 0) {
  python -m pytest
} else {
  Write-Host "Skipping pytest; install Python dev dependencies with .\\scripts\\bootstrap.ps1"
}

