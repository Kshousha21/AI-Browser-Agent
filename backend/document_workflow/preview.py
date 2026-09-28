"""Private, user-only preview identity; never derive a download URL."""
import re
from urllib.parse import parse_qsl, unquote, urlsplit
from pydantic import Field
from local_pdf.models import LocalModel
from browser.domain_policy import navigation_allowed
from browser.product_scope import product_url_allowed
from .models import DocumentError


class PreviewIdentity(LocalModel):
    parent_id: str = Field(pattern=r"^[0-9]+$", max_length=64)
    attachment_id: str = Field(pattern=r"^[0-9]+$", max_length=64)
    filename: str = Field(min_length=5, max_length=512)
    version: str | None = Field(default=None, pattern=r"^[1-9][0-9]*$", max_length=32)


def decoded(value):
    if re.search(r"%(?![0-9a-fA-F]{2})", value):
        raise ValueError("encoding")
    text = unquote(value, errors="strict")
    if "%" in text or "\\" in text or any(ord(c) < 32 for c in text):
        raise ValueError("encoding")
    return text


def safe_parts(url, source, allowlist):
    if not isinstance(url, str) or len(url) > 2048 or any(ord(c) <= 32 for c in url):
        raise ValueError("url")
    parsed, origin = urlsplit(url), urlsplit(source)
    if (parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc) or parsed.username or parsed.password or "#" in url:
        raise ValueError("origin")
    if not product_url_allowed(url) or not navigation_allowed(url, allowlist):
        raise ValueError("scope")
    path = decoded(parsed.path)
    if any(part in {".", ".."} for part in path.split("/")):
        raise ValueError("path")
    # Decode once strictly before parse_qsl; parse_qsl then performs the sole decode.
    decoded(parsed.query)
    pairs = parse_qsl(parsed.query, keep_blank_values=True, errors="strict")
    query = dict(pairs)
    if len(query) != len(pairs):
        raise ValueError("duplicate query")
    return path, query


def file_name(value, *, allow_non_pdf=False):
    if (not isinstance(value, str) or not 5 <= len(value) <= 512
            or re.search(r'[\\/:*?"<>|\x00-\x1f]', value)
            or re.fullmatch(r".+\.[A-Za-z0-9]{1,16}", value) is None):
        raise DocumentError("unsupported_attachment_type")
    if not allow_non_pdf and not value.lower().endswith(".pdf"):
        raise DocumentError("unsupported_attachment_type")
    return value


def download_query(query):
    if set(query) - {"version", "modificationDate", "api", "download"}:
        raise ValueError("query")
    for key, pattern in {"version": r"[1-9][0-9]*", "modificationDate": r"[0-9]+", "api": r"v[12]", "download": r"true|false"}.items():
        if key in query and not re.fullmatch(pattern, query[key]):
            raise ValueError("query")


def direct_file(url, source, metadata, allowlist, *, allow_non_pdf=False):
    """Explicit file-card validation; preserve the observed URL, never build one."""
    try:
        path, query = safe_parts(url, source, allowlist)
        match = re.fullmatch(r"(.+)/([^/]+)", path)
        if not match or match[2] != metadata.get("attachment_filename"):
            raise ValueError("identity")
        file_name(match[2], allow_non_pdf=allow_non_pdf)
        return url
    except DocumentError:
        raise
    except (ValueError, TypeError, UnicodeError):
        raise DocumentError("preview_identity_mismatch") from None


def preview_identity(url, source, metadata, allowlist, *, allow_non_pdf=False):
    try:
        path, query = safe_parts(url, source, allowlist)
        evidence = metadata.get("preview") or {}
        if metadata.get("preview_url") != url:
            raise ValueError("preview")
        wanted = PreviewIdentity.model_validate(evidence)
        filename = file_name(wanted.filename, allow_non_pdf=allow_non_pdf)
        if filename != metadata.get("attachment_filename"):
            raise DocumentError("preview_identity_mismatch")
        return PreviewIdentity(parent_id=wanted.parent_id, attachment_id=wanted.attachment_id, filename=filename,
                               version=wanted.version).model_dump()
    except DocumentError:
        raise
    except (ValueError, TypeError, UnicodeError):
        raise DocumentError("attachment_preview_only") from None


def validate_download(url, source, expected, evidence, allowlist, *, allow_non_pdf=False):
    """Only a separately observed, fully matching link can become an acquisition."""
    try:
        wanted, observed = PreviewIdentity.model_validate(expected), PreviewIdentity.model_validate(evidence)
        if (wanted.parent_id, wanted.attachment_id, wanted.filename) != (observed.parent_id, observed.attachment_id, observed.filename):
            raise ValueError("identity")
        if wanted.version is not None and wanted.version != observed.version:
            raise ValueError("version")
        path, query = safe_parts(url, source, allowlist)
        match = re.fullmatch(r"(.+)/([^/]+)", path)
        if not match or match[2] != wanted.filename:
            raise ValueError("target")
        if not allow_non_pdf and not match[2].lower().endswith(".pdf"):
            raise ValueError("query")
        file_name(match[2], allow_non_pdf=allow_non_pdf)
        if observed.version is not None and query.get("version") != observed.version:
            raise ValueError("version")
        return url  # Preserve the exact observed query, including its ordering.
    except (ValueError, TypeError, UnicodeError):
        raise DocumentError("preview_identity_mismatch") from None
