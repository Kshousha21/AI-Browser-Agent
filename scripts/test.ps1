[CmdletBinding(DefaultParameterSetName='Full')]
param(
    [Parameter(ParameterSetName='Fast')][switch] $Fast,
    [Parameter(ParameterSetName='Browser')][switch] $Browser
)
. (Join-Path $PSScriptRoot 'common.ps1')
try {
    $repo = Get-ProjectRoot
    $python = Get-ProjectPython $repo
    $mode = 'Full'
    if ($Fast) {
        $mode = 'Fast'
        Push-Location (Join-Path $repo 'extension')
        try { Invoke-Checked (Get-Npm) @('run', 'typecheck') } finally { Pop-Location }
    } else {
        if ($Browser) { $mode = 'Browser' }
        & (Join-Path $PSScriptRoot 'build.ps1')
    }
    Invoke-Checked $python @('-B', (Join-Path $repo 'tools/test_runner.py'), '--mode', $mode)
} catch { Write-Error $_; exit 1 }
