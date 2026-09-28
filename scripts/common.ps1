Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ProjectRoot {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    foreach ($directory in @('backend', 'extension')) {
        if (-not (Test-Path -LiteralPath (Join-Path $candidate $directory) -PathType Container)) {
            throw 'Wrong project root. Expected backend/, extension/ and shared/build-contract.json.'
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $candidate 'shared/build-contract.json') -PathType Leaf)) {
        throw 'Wrong project root. Expected backend/, extension/ and shared/build-contract.json.'
    }
    # Anchor to this script, never a parent checkout or Git environment override.
    # Preserve mapped-drive spelling: npm.cmd cannot use a UNC working directory.
    return $candidate
}

function Get-ProjectPython([string] $Root) {
    $python = Join-Path $Root 'backend/.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'Project Python is missing. Run scripts/setup.ps1 first.' }
    return $python
}

function Invoke-Checked([string] $Executable, [string[]] $Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Required command failed (exit $LASTEXITCODE). See output above." }
}

function Get-Npm {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) { throw 'Node/npm are missing. Install Node.js with npm, then rerun setup.' }
    return $npm.Source
}
