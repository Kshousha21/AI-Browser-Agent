[CmdletBinding()]
param()
. (Join-Path $PSScriptRoot 'common.ps1')
try {
    $repo = Get-ProjectRoot
    $python = Get-ProjectPython $repo
    Push-Location (Join-Path $repo 'backend')
    try {
        $raw = & $python -B -m browser.config_check --json
        if ($LASTEXITCODE -ne 0) { throw 'Configuration is not ready. Run scripts/doctor.ps1 or python -m browser.config_check. Values are never printed.' }
        $configuration = $raw | ConvertFrom-Json
        $portState = & $python -B (Join-Path $repo 'tools/doctor.py') --host $configuration.server.host --port $configuration.server.port
        if ($LASTEXITCODE -ne 0) {
            throw "Port $($configuration.server.port) is occupied or unavailable. The backend may already be running. Check the existing backend terminal or choose a free loopback port in your private local configuration. No process was stopped."
        }
        Write-Host "Starting backend on loopback port $($configuration.server.port); active provider type: $($configuration.active_provider)."
        Invoke-Checked $python @('-B', '-m', 'browser.server')
    } finally { Pop-Location }
} catch { Write-Error $_; exit 1 }
