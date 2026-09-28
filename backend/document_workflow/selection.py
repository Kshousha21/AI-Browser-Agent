"""One current bounded result window per socket, not a global result-ref cache."""
from dataclasses import dataclass, field
import hashlib
import json
import re
import secrets
import time
from urllib.parse import parse_qs, unquote, urlsplit
from browser.protocol import PageState
from browser.state_reads import RefinedPageState, context_digest, state_identity
from browser.product_scope import product_url_allowed
from browser.domain_policy import navigation_allowed
from browser.providers import ToolCall
from app.site_adapters.page_files import retrieve, TOOL_NAME
from .models import DocumentError, PrepareRequest, FileActionRequest
from .preview import preview_identity, direct_file, file_name


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


@dataclass(frozen=True, repr=False)
class SelectedAttachment:
    result_ref: str
    index: int
    url: str = field(repr=False)
    metadata: dict = field(repr=False)
    source_fingerprint: str
    preview: dict | None = field(default=None, repr=False)


def resolve_attachment(page: RefinedPageState, result_ref: str, allowed_refs: set[str], allowlist: list[str], *, file_action=False, allow_non_pdf=False) -> SelectedAttachment:
    if not isinstance(page, RefinedPageState) or not page.context_is_current() or result_ref not in allowed_refs:
        raise DocumentError("stale_result_ref")
    matches = []
    for offset in range(0, len(page.site_context.get("files", [])), 10):
        call = ToolCall(tool=TOOL_NAME, arguments={"limit": 10, "offset": offset},
                        snapshot_id=page.snapshot_id, reason="Resolve explicit selection")
        projected = retrieve(call, page)
        if not projected.success:
            raise DocumentError("stale_result_ref")
        matches.extend((offset + i, row) for i, row in enumerate(projected.data["results"])
                       if row["result_ref"] == result_ref)
    if len(matches) != 1:
        raise DocumentError("stale_result_ref")
    index, metadata = matches[0]
    record = page.site_context["files"][index]
    url = record.get("url")
    if not allow_non_pdf and record.get("file_type") not in {None, "pdf"}:
        raise DocumentError("unsupported_attachment_type")
    if file_action:
        filename = file_name(metadata.get("attachment_filename"), allow_non_pdf=allow_non_pdf)
        if (record.get("content_type") != "attachment"
                or record.get("file_type") not in {None, filename.rsplit(".", 1)[1].lower()}):
            raise DocumentError("unsupported_attachment_type")
    preview = None
    try:
        parsed, current = urlsplit(url), urlsplit(page.url)
        if metadata.get("preview") is not None:
            if record.get("content_type") != "attachment":
                raise DocumentError("attachment_unavailable")
            preview = preview_identity(url, page.url, metadata, allowlist, allow_non_pdf=allow_non_pdf)
        decoded = unquote(parsed.path)
        safe = (isinstance(url, str) and len(url) <= 2048
            and (parsed.scheme, parsed.netloc) == (current.scheme, current.netloc)
            and not parsed.username and not parsed.password and not parsed.fragment
            and re.fullmatch(r"/.*[^/]+\.[pP][dD][fF]", decoded) is not None
            and "\\" not in decoded and not any(part in {".", ".."} for part in decoded.split("/"))
            and not any(ord(c) < 32 for c in url)
            and decoded.lower().endswith(".pdf") and record.get("content_type") == "attachment"
            and product_url_allowed(url) and navigation_allowed(url, allowlist))
    except (ValueError, TypeError, AttributeError):
        safe = False
    if file_action and preview is None:
        direct_file(url, page.url, metadata, allowlist, allow_non_pdf=allow_non_pdf)
        safe = True
    if not safe and preview is None:
        raise DocumentError("attachment_unavailable")
    # URL is used only for current private target binding. Persist its digest,
    # never raw query parameters, filenames/titles or a permanent result_ref.
    fingerprint = digest({"attachment": url, "metadata": {k: v for k, v in metadata.items() if k != "result_ref"}})
    return SelectedAttachment(result_ref, index, url, metadata, fingerprint, preview)


class CurrentSelection:
    def __init__(self, clock=None):
        # Diagnostic creation time only. Result authority is invalidated by
        # observation/session state and one-shot consumption, never elapsed time.
        self.clock = clock or time.monotonic
        self.clear()

    def clear(self):
        self.page = None
        self.offer_id = None
        self.refs = set()
        self.created = 0.0
        self.file_actions = set()

    def offer(self, page: RefinedPageState, result):
        self.clear()
        if not isinstance(page, RefinedPageState) or not page.context_is_current() or not result.success:
            return None
        if result.action != TOOL_NAME or result.snapshot_id != page.snapshot_id:
            return None
        # Verify the exact bounded projection, not an arbitrary provider result.
        data = result.data
        call = ToolCall(tool=TOOL_NAME, arguments={"limit": data.get("limit"), "offset": data.get("offset")},
                        snapshot_id=page.snapshot_id, reason="Retain current result window")
        projected = retrieve(call, page)
        if not projected.success or projected.data["results"] != data.get("results"):
            return None
        self.page = page.model_copy(deep=True)
        self.offer_id = secrets.token_hex(16)
        self.refs = {row["result_ref"] for row in data["results"]}
        self.created = self.clock()
        return {"offer_id": self.offer_id, "results": data["results"], "trust": "untrusted_webpage_data",
                "freshness": "state_bound", "expires_in_ms": 0, "expires_at_ms": 0}

    def consume(self, request: PrepareRequest, allowlist: list[str]):
        page = self.page
        try:
            if page is None or request.offer_id != self.offer_id or not self.refs:
                raise DocumentError("stale_result_ref")
            selected = [resolve_attachment(page, ref, self.refs, allowlist) for ref in request.result_refs]
            if any(item.preview is not None for item in selected):
                if len(selected) != 1:
                    raise DocumentError("preview_selection_limit")
            return page, selected
        finally:
            self.clear()  # Every selection attempt consumes its offer, including cancel/failure.

    def consume_file(self, request: FileActionRequest, allowlist: list[str]):
        """No bulk refs or silent renewal; the clicked offer is always retired."""
        try:
            if self.page is None or request.offer_id != self.offer_id or not self.refs:
                raise DocumentError("stale_result_ref")
            item = resolve_attachment(self.page, request.result_ref, self.refs, allowlist,
                file_action=True, allow_non_pdf=request.action != "prepare")
            return self.page, item
        finally:
            self.clear()

    def admit_file(self, request: FileActionRequest, allowlist: list[str], action_id: str):
        """Displayed choices survive clicks; each action ID is one-use authority.

        A fresh owner observation is still required before dispatch. Clear,
        external navigation, Stop and replacement offers retire this catalogue.
        """
        if (self.page is None or request.offer_id != self.offer_id or not self.refs
                or action_id in self.file_actions or len(self.file_actions) >= 256):
            raise DocumentError("stale_result_ref")
        self.file_actions.add(action_id)
        return self.page, resolve_attachment(self.page, request.result_ref, self.refs, allowlist,
            file_action=True, allow_non_pdf=request.action != "prepare")


def verify_refresh(original: RefinedPageState, current: PageState) -> None:
    """The workflow's own fresh observation may replace only the snapshot ID.

    Previously consumed refs never resolve against this new state. Confirmation
    and dispatch bind to the newly observed page after equality of all source
    evidence and tab/document/navigation identity has been checked.
    """
    if (not isinstance(current, RefinedPageState) or not current.context_is_current()
            or not original.context_is_current()
            or state_identity(original)[:3] != state_identity(current)[:3]
            or original.url != current.url or context_digest(original) != context_digest(current)):
        raise DocumentError("stale_result_ref")
