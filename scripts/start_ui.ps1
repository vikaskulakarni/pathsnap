# Laptop 2: start the evaluator UI (http://127.0.0.1:8501).
# Pass the address of Laptop 1, e.g.  .\scripts\start_ui.ps1 -Api http://10.1.6.53:8000
param([string]$Api = "http://localhost:8000")
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PATHSNAP_API_URL = $Api
python -m ui.app
