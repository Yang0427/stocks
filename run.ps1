# run.ps1 — start Savings Scout (FastAPI backend + Vite frontend) for development.
#
#   powershell -ExecutionPolicy Bypass -File run.ps1
#
# Backend:  http://127.0.0.1:8000   (API)
# Frontend: http://127.0.0.1:5173   (open this in your browser)
#
# The Vite dev server proxies /api -> :8000, so just use the 5173 URL.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "Starting FastAPI backend on :8000 ..." -ForegroundColor Cyan
$backend = Start-Process -PassThru -WindowStyle Normal -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "api:app", "--port", "8000" -WorkingDirectory $root

Start-Sleep -Seconds 2

if (-not (Test-Path (Join-Path $root "frontend/node_modules"))) {
    Write-Host "Installing frontend dependencies (first run) ..." -ForegroundColor Yellow
    Push-Location (Join-Path $root "frontend"); npm install; Pop-Location
}

Write-Host "Starting Vite frontend on :5173 ..." -ForegroundColor Cyan
Write-Host "==> Open http://localhost:5173 in your browser <==" -ForegroundColor Green

Push-Location (Join-Path $root "frontend")
try {
    npm run dev
} finally {
    Pop-Location
    Write-Host "Shutting down backend ..." -ForegroundColor Cyan
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
}
