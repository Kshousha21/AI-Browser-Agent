"""Check installed distributions against project requirements without network access."""
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def check(requirements):
    from packaging.requirements import Requirement
    missing = incompatible = 0
    for line in requirements.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            missing += 1
            continue
        if installed not in requirement.specifier:
            incompatible += 1
    return dict(ready=not missing and not incompatible, missing=missing, incompatible=incompatible)


if __name__ == "__main__":
    try:
        result = check((Path(__file__).resolve().parents[1] / "backend/requirements.txt").read_text())
    except Exception:
        result = dict(ready=False)
    print(json.dumps(result))
    raise SystemExit(0 if result["ready"] else 2)
