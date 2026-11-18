# Laptop 1: start the PathSnap API on all interfaces (port 8000).
# Until data/eval_test/ exists, rehearse with the student_test trips:  -Rehearse
param([switch]$Rehearse, [switch]$AllTrips)
Set-Location (Split-Path -Parent $PSScriptRoot)
$previousSplit = $env:PATHSNAP_EVAL_SPLIT
$previousRelease = $env:PATHSNAP_MODEL_RELEASE
try {
    $env:PATHSNAP_MODEL_RELEASE = if ($AllTrips) { "all80" } else { "demonstrated" }
    $env:PATHSNAP_EVAL_SPLIT = if ($Rehearse) { "student_test" } else { "eval_test" }
    python -m scripts.verify_release --runtime-only
    if ($LASTEXITCODE -ne 0) { throw "Model release verification failed." }
    # Show the LAN addresses Laptop 2 can use; a VPN may add another adapter.
    $adapters = Get-NetIPConfiguration -ErrorAction SilentlyContinue |
        Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq "Up" }
    foreach ($adapter in $adapters) {
        foreach ($address in $adapter.IPv4Address) {
            Write-Host ("Laptop 2 API address ({0}): http://{1}:8000" -f $adapter.InterfaceAlias, $address.IPAddress)
        }
    }
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
} finally {
    $env:PATHSNAP_EVAL_SPLIT = $previousSplit
    $env:PATHSNAP_MODEL_RELEASE = $previousRelease
}
