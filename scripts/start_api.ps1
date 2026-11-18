# Laptop 1: start the PathSnap API on all interfaces (port 8000).
# Until data/eval_test/ exists, rehearse with the student_test trips:  -Rehearse
param([switch]$Rehearse)
Set-Location (Split-Path -Parent $PSScriptRoot)
if ($Rehearse) { $env:PATHSNAP_EVAL_SPLIT = "student_test" }
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
