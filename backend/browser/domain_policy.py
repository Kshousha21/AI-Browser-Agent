"""Navigation allowlist enforcement independent of Chrome permissions."""

from __future__ import annotations

import re
from urllib.parse import urlparse

_PRIVILEGED = re.compile(
    r"^(?:chrome|chrome-extension|edge|about|devtools|view-source|file|data|blob):",
    re.IGNORECASE,
)
_REGEX_META = re.compile(r"([.+?^$(){}|\[\]\\])")


def pattern_matches(url: str, pattern: str) -> bool:
    trimmed = pattern.strip()
    if not trimmed:
        return False
    escaped = _REGEX_META.sub(r"\\\1", trimmed)
    if escaped.startswith("*:"):
        escaped = "https?:" + escaped[2:]
    escaped = escaped.replace("*", ".*")
    try:
        return re.fullmatch(escaped, url, re.IGNORECASE) is not None
    except re.error:
        return False


def navigation_allowed(url: str, allowlist: list[str] | None = None) -> bool:
    """Allow normal HTTP(S); legacy allowlist input carries no authority."""
    from .product_scope import product_url_allowed
    return product_url_allowed(url)
