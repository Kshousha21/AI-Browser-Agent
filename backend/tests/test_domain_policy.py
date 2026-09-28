from __future__ import annotations

import pytest

from browser.domain_policy import navigation_allowed, pattern_matches


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://127.0.0.1:8080/results.html", True),
        ("http://localhost:8080/results.html?q=x", True),
        ("https://other.example/", True),
        ("file:///tmp/a.html", False),
        ("javascript:alert(1)", False),
    ],
)
def test_navigation_policy(url: str, expected: bool) -> None:
    allowlist = ["http://127.0.0.1:8080/*", "http://localhost:8080/*"]
    assert navigation_allowed(url, allowlist) is expected


def test_wildcard_scheme_pattern() -> None:
    assert pattern_matches("https://docs.example.test/a", "*://docs.example.test/*")
    assert pattern_matches("http://docs.example.test/a", "*://docs.example.test/*")
