"""Run existing pytest selections once, reporting all outcome counts explicitly."""
import argparse
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).absolute().parents[1]
BROWSER = ["test_generic_browser_transitions.py", "test_browser_control_real_chrome.py"]


def selection(mode):
    tests = ROOT / "backend/tests"
    if mode == "Browser":
        return [str(tests / name) for name in BROWSER]
    args = [str(tests)]
    if mode == "Fast":
        args += ["--ignore=" + str(tests / name) for name in BROWSER]
    return args


class Summary:
    def __init__(self):
        self.counts = dict(passed=0, failed=0, skipped=0, errors=0)

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.counts["skipped"] += 1
        elif report.failed:
            self.counts["failed" if report.when == "call" else "errors"] += 1
        elif report.when == "call":
            self.counts["passed"] += 1

    def pytest_collectreport(self, report):
        if report.failed:
            self.counts["errors"] += 1
        elif report.skipped:
            self.counts["skipped"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["Full", "Fast", "Browser"], default="Full")
    mode = parser.parse_args().mode
    sys.path.insert(0, str(ROOT / "backend"))
    import pytest
    summary = Summary()
    result = pytest.main(["-q", "-ra", "--tb=short", *selection(mode)], plugins=[summary])
    print("Local test results: " + ", ".join(f"{key}={value}" for key, value in summary.counts.items()))
    if summary.counts["skipped"]:
        print("INCOMPLETE: skipped checks are not verified. Install required local prerequisites and rerun.")
    raise SystemExit(int(result) or (1 if summary.counts["skipped"] else 0))


if __name__ == "__main__":
    main()
