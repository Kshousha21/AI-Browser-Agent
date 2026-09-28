"""Privacy-safe tab metadata and exact, one-shot close authorization."""
from __future__ import annotations

from typing import Any
from pydantic import Field

from .actions import ActionResult
from .product_scope import product_url_allowed
from .providers import ToolCall
from .security import RiskAssessment, permission_denied


class ApprovedTabClose(ToolCall):
    """Backend-created proof of the exact browser revision approved by the user."""
    tab_revision: str = Field(min_length=1, max_length=200)


def safe_tabs(value: object, *, internal: bool = False) -> list[dict[str, Any]]:
    output = []
    if not isinstance(value, list):
        return output
    seen = set()
    for raw in value[:100]:
        if not isinstance(raw, dict):
            continue
        tab_id = raw.get("tab_id")
        if not isinstance(tab_id, int) or isinstance(tab_id, bool) or tab_id < 0 or tab_id in seen:
            continue
        seen.add(tab_id)
        row: dict[str, Any] = {"tab_id": tab_id, "active": raw.get("active") is True, "allowed": False}
        location = raw.get("location")
        if raw.get("allowed") is True and isinstance(location, str) and product_url_allowed(location):
            # The transport exposes no query/fragment or arbitrary extra browser fields.
            location = location.split("?", 1)[0].split("#", 1)[0]
            row.update(allowed=True, opened_by_agent=raw.get("opened_by_agent") is True,
                title=str(raw.get("title", ""))[:160], location=location[:240])
            if internal:
                revision = raw.get("revision")
                if isinstance(revision, str) and 0 < len(revision) <= 200:
                    row["revision"] = revision
        else:
            row["details"] = "redacted"
        output.append(row)
    return output


async def authorize_close(call, page, permissions, bridge, settings, step):
    """Closing needs one-shot authorization and exact tab revalidation."""
    if permissions is None or permissions.read_only:
        return call, permission_denied(call, page, "Closing a tab requires read-only mode off and explicit confirmation."), "read_only_blocked"

    async def observe():
        read = ToolCall(tool="get_tabs", arguments={}, snapshot_id=page.snapshot_id,
            reason="Inspect the exact tab before closing it.")
        result = await bridge.execute(read, step, settings.max_steps)
        await bridge.timeline(read, result, step)
        await bridge.audit(read, result, "close_tab_revalidation")
        return next((row for row in safe_tabs(result.data.get("tabs"), internal=True)
            if row["tab_id"] == call.arguments["tab_id"] and row["allowed"] and row.get("revision")), None) if result.success else None

    tab = await observe()
    if tab is None:
        return call, permission_denied(call, page, "The requested tab is missing or outside product scope."), "scope_blocked"
    target = f"Tab {tab['tab_id']}: {tab.get('title', '')} | {tab.get('location', '')} | opened_by_agent={tab.get('opened_by_agent', False)}"
    assessment = RiskAssessment("high", ("close_tab",), f"close_tab:{tab['tab_id']}", target)
    await bridge.status("Authorizing action...", step, settings.max_steps, 0, {})
    choice = await bridge.confirm(call, assessment, step)
    if choice not in {"allow_once", "allow_run", "policy_allow_once"}:
        return call, permission_denied(call, page, "The user cancelled closing the tab."), "cancel"
    current = await observe()
    if current != tab:
        return call, ActionResult(success=False, action="close_tab", snapshot_id=page.snapshot_id,
            error="stale_snapshot", message="The approved tab changed. Observe and confirm again."), "approval_invalidated"
    return ApprovedTabClose(**call.model_dump(), tab_revision=tab["revision"]), None, "policy_allow_once" if choice == "policy_allow_once" else "allow_once"
