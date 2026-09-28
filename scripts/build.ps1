[CmdletBinding()]
param()
. (Join-Path $PSScriptRoot 'common.ps1')
try {
    $repo = Get-ProjectRoot
    $npm = Get-Npm
    Push-Location (Join-Path $repo 'extension')
    try { Invoke-Checked $npm @('run', 'build') } finally { Pop-Location }
    foreach ($output in @('extension/dist', 'extension/dist/manifest.json', 'extension/dist/content/extract.js', 'extension/dist/build-info.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $repo $output))) { throw "Expected build output missing: $output" }
    }
    Write-Host 'Build complete. Reload extension/dist in Chrome and restart the matching backend.'
} catch { Write-Error $_; exit 1 }
