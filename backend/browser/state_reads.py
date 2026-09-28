"""Backend-only reads of current, provenance-bound adapter context; no I/O."""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import PrivateAttr

from .actions import ActionResult
from .domain_policy import navigation_allowed
from .product_scope import product_url_allowed
from .protocol import PageState
from .providers import ToolCall
from .session_control import RunControl

if TYPE_CHECKING:
    from app.site_adapters import SiteAdapterRegistry


def state_identity(page: PageState) -> tuple:
    return (page.tab_id, page.document_id, page.navigation_id, page.snapshot_id, page.url)


def context_digest(page: PageState) -> str:
    data = json.dumps(page.site_context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


class RefinedPageState(PageState):
    """A normal PageState with non-serialized provenance, not a second cache."""

    _context_identity: tuple = PrivateAttr(default=())
    _context_digest: str = PrivateAttr(default="")

    @classmethod
    def bind(cls, page: PageState) -> RefinedPageState:
        refined = cls.model_validate(page.model_dump())
        refined._context_identity = state_identity(refined)
        refined._context_digest = context_digest(refined)
        return refined

    def context_is_current(self) -> bool:
        return (self._context_identity == state_identity(self)
                and self._context_digest == context_digest(self))


def unavailable(call: ToolCall, page: PageState, *, stale: bool = False) -> ActionResult:
    return ActionResult(
        success=False, action=call.tool, snapshot_id=page.snapshot_id,
        error="stale_snapshot" if stale else "no_active_search_results",
        message=("The decision or adapter records do not belong to the current page state."
                 if stale else "No validated classic-search results are active on the current page."),
    )


def execute_state_read(
    call: ToolCall, page: PageState, *, target: PageState | None,
    decision_identity: tuple, decision_epoch: int | None,
    control: RunControl | None, adapters: SiteAdapterRegistry,
    allowlist: list[str],
) -> ActionResult:
    """Check the existing target/control boundary synchronously, then read it.

    There is no await between checking the epoch and reading the records. The
    session's target comes from the existing bridge; missing identity fails
    closed. A fresh observation is never performed by this read path.
    """
    if control is not None:
        control.check()
        if control.needs_refresh or control.epoch != decision_epoch:
            return unavailable(call, page, stale=True)
    if (target is None or page.tab_id is None or not page.document_id
            or not page.snapshot_id or state_identity(target) != state_identity(page)
            or decision_identity != state_identity(page)
            or call.snapshot_id != page.snapshot_id):
        return unavailable(call, page, stale=True)
    if not product_url_allowed(page.url) or not navigation_allowed(page.url, allowlist):
        return unavailable(call, page)
    if not isinstance(page, RefinedPageState) or not page.context_is_current():
        return unavailable(call, page, stale=True)
    return adapters.read_state(call, page)
