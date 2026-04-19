# Start FastAPI from the repo root that contains the `backend` package (inner ML_Project2 folder).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "API cwd: $root"
Write-Host "Endpoints include POST /import/csv and POST /api/import/csv"
Write-Host "Open http://127.0.0.1:8000/docs and confirm those routes exist."
# 0.0.0.0 so both http://127.0.0.1:8000 and http://localhost:8000 reach this process on Windows.
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
