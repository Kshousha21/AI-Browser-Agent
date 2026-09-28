[CmdletBinding()]
param()
. (Join-Path $PSScriptRoot 'common.ps1')
try {
    $repo = Get-ProjectRoot
    $python = Join-Path $repo 'backend/.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        $available = Get-Command python -ErrorAction SilentlyContinue
        if (-not $available) { throw 'NOT READY: Python is missing. Install Python 3.11+, then run scripts/setup.ps1.' }
        $python = $available.Source
    }
    & $python -B (Join-Path $repo 'tools/doctor.py')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} catch { Write-Error $_; exit 1 }
