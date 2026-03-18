$ErrorActionPreference = "Stop"

python -m pip install -r requirements-dev.txt
python -m pip install -e python/adapter-common
python -m pip install -e python/alpha-beta-crown-adapter

Write-Host "Bootstrap complete."

