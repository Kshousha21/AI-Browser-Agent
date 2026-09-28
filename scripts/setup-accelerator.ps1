[CmdletBinding()]
param([Parameter(Mandatory=$true)][ValidateSet('cuda','directml')][string] $Device)
. (Join-Path $PSScriptRoot 'common.ps1')
try {
    $repo = Get-ProjectRoot
    $python = Get-ProjectPython $repo
    Invoke-Checked $python @('-B', (Join-Path $repo 'tools/setup_accelerator.py'), '--device', $Device)
    Write-Host 'Optional runtime installed in .local/accelerators. Model preparation and inference acceptance are separate explicit actions.'
} catch { Write-Error $_; exit 1 }
