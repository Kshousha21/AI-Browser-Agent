# Testing and development checks

Run commands from the project root. Install dependencies as described in the
[README](../README.md). The structural root markers are `backend/`, `extension/`
and `shared/build-contract.json`; a Git checkout is not required. Git is still used
by optional publication checks and may be needed to install source dependencies.

```powershell
.\scripts\build.ps1
.\scripts\test.ps1 -Fast
.\scripts\test.ps1 -Browser
.\scripts\doctor.ps1
```

`test.ps1` without a switch runs the full suite. Fast runs TypeScript and excludes
the two real-browser suites; tests that consume built assets require a current build.
Browser and Full build first. In `extension/`, `npm.cmd run typecheck` checks only
TypeScript and `npm.cmd run build` runs TypeScript, both Vite builds and build metadata.

For targeted operational checks, use a fresh local temporary directory:

```powershell
$testRoot = Join-Path $env:TEMP ('agent-tests-' + [guid]::NewGuid().ToString('N'))
.\backend\.venv\Scripts\python.exe -B -m pytest backend/tests/test_repository_root.py backend/tests/test_operations.py -q -p no:cacheprovider --basetemp=$testRoot
```

Real Chrome tests use synthetic pages and isolated test profiles. They must not use
personal profiles. They require installed Chrome and an environment that permits
local browser automation; some checks need an interactive Windows desktop. Skipped
checks are incomplete validation, even when the underlying runner exits successfully.

Provider wire tests use deterministic transports. They verify serialization and
routing, not the quality or availability of a live local/cloud model. PDF tests use
synthetic files; live MiniLM/Nomic inference requires installed artifacts and suitable
runtime dependencies. `tools/local_routing_check.py --run` performs a separate local
router/main-model check. Doctor checks configuration and prerequisites without making
live provider requests; readiness is not proof of successful inference.

Build fingerprints cover extension sources, shared contracts and build inputs.
Optional commit metadata can be unavailable in source snapshots. The authenticated
runtime handshake checks the version/revision in `shared/build-contract.json`;
rebuild/reload the extension and restart the backend together after protocol changes.
