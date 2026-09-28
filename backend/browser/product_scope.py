"""HTTP(S) URL validation. Site authorization belongs to Chrome."""
import re
from urllib.parse import urlsplit

def product_url_allowed(value: str) -> bool:
    try:
        if not isinstance(value, str) or re.search(r"[\\\s]", value):
            return False
        url = urlsplit(value)
        return bool(url.scheme in {"http", "https"} and url.hostname and not url.username
                    and not url.password and (url.port is None or 0 < url.port <= 65535))
    except (ValueError, UnicodeError):
        return False

def product_pattern_allowed(pattern: str) -> bool:
    # Compatibility with older protocol clients; never a navigation gate.
    return isinstance(pattern, str)
