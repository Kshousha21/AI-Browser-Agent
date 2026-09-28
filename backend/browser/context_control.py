"""Explicit full/diff model observations with measured fallback thresholds."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .actions import ActionResult
from .context_diff import diff_page_payload
from .context_payload import full_page_payload, serialized_size
from .protocol import PageState

_DIFF_RATIO = 0.50

__all__ = (
    "compact_result",
    "diff_page_payload",
    "full_page_payload",
    "page_observation",
    "serialized_size",
)


def compact_result(
    result: ActionResult,
    previous: PageState,
    *,
    force_full: bool = False,
) -> ActionResult:
    """Replace an embedded raw state with an explicit full/diff observation."""
    raw = result.data.get("page_state")
    if not isinstance(raw, dict):
        return result
    try:
        current = PageState.model_validate(raw)
    except ValidationError:
        return result
    observation = page_observation(
        previous,
        current,
        force_full=force_full or result.page_changed,
    )
    return result.model_copy(
        update={"data": {**result.data, "page_state": observation}}
    )


def page_observation(
    previous: PageState | None,
    current: PageState,
    *,
    force_full: bool = False,
) -> dict[str, Any]:
    """Choose an explicit full snapshot or safe structured delta."""
    reason = _full_snapshot_reason(previous, current, force_full)
    full = full_page_payload(current, reason=reason or "diff_too_large")
    if reason or previous is None:
        return full
    try:
        diff = diff_page_payload(previous, current)
    except (KeyError, TypeError, ValueError):
        return full_page_payload(current, reason="diff_unsafe")
    if serialized_size(diff) > serialized_size(full) * _DIFF_RATIO:
        return full
    return diff


def _full_snapshot_reason(
    previous: PageState | None,
    current: PageState,
    force_full: bool,
) -> str | None:
    if force_full:
        return "navigation_or_explicit_request"
    if previous is None:
        return "initial"
    if not previous.snapshot_id or not current.snapshot_id:
        return "missing_snapshot_lineage"
    if not previous.document_id or not current.document_id:
        return "missing_page_identity"
    if previous.tab_id != current.tab_id:
        return "tab_changed"
    if previous.document_id != current.document_id:
        return "document_changed"
    if previous.navigation_id != current.navigation_id:
        return "history_navigation"
    old_frames = [(item.frame_id, item.origin) for item in previous.frames]
    new_frames = [(item.frame_id, item.origin) for item in current.frames]
    if old_frames != new_frames:
        return "frame_identity_changed"
    return None
