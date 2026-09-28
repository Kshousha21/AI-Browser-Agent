[CmdletBinding()]
param()
. (Join-Path $PSScriptRoot 'common.ps1')
try {
    $repo = Get-ProjectRoot
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if (-not $systemPython) { throw 'Python 3.11+ is required. Install it, then rerun setup.' }
    Invoke-Checked $systemPython.Source @('-c', 'import sys; assert (3,11) <= sys.version_info < (4,0); print(sys.version.split()[0])')
    $npm = Get-Npm
    $nodeVersion = & node --version
    if ($LASTEXITCODE -ne 0 -or $nodeVersion -notmatch '^v(\d+)\.') { throw 'Node.js version could not be checked.' }
    if ([version]$nodeVersion.TrimStart('v') -lt [version]'22.12.0') { throw 'Node.js 22.12+ is required; Node 24 LTS is recommended.' }
    Write-Host "Node $nodeVersion"
    $environmentPath = Join-Path $repo 'backend/.venv'
    if (-not (Test-Path -LiteralPath $environmentPath)) {
        Invoke-Checked $systemPython.Source @('-m', 'venv', $environmentPath)
    }
    $python = Get-ProjectPython $repo
    Invoke-Checked $python @('-c', 'import sys; assert (3,11) <= sys.version_info < (4,0)')
    Invoke-Checked $python @('-B', (Join-Path $repo 'tools/check_base_runtime.py'))
    Invoke-Checked $python @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', (Join-Path $repo 'backend/requirements.txt'))
    Invoke-Checked $python @('-m', 'pip', 'check')
    Push-Location (Join-Path $repo 'extension')
    try {
        if (Test-Path -LiteralPath 'package-lock.json') { Invoke-Checked $npm @('ci', '--no-audit', '--no-fund') }
        else { throw 'The lockfile is missing. Restore extension/package-lock.json before setup.' }
    } finally { Pop-Location }
    & (Join-Path $PSScriptRoot 'build.ps1')
    Push-Location (Join-Path $repo 'backend')
    try { Invoke-Checked $python @('-B', '-m', 'browser.config_check', '--structure') }
    finally { Pop-Location }
    if (-not (Test-Path -LiteralPath (Join-Path $repo 'backend/.env'))) {
        Write-Host 'backend/.env is missing. Run: .\backend\.venv\Scripts\python.exe -B tools/init_local_config.py'
    }
    Write-Host 'Setup complete. Follow README.md to configure local_agent, pull both Ollama models, and connect Chrome.'
    Write-Host 'Next: scripts/doctor.ps1, then scripts/start-backend.ps1. In Chrome load unpacked extension/dist; do not load extension/.'
} catch { Write-Error $_; exit 1 }
