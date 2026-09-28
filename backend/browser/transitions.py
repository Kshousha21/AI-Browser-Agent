"""Observe meaningful same-document changes after successful clicks."""

from __future__ import annotations

import asyncio

from .actions import ActionResult
from .agent_support import AgentBridge, page_signature
from .config import AgentSettings
from .protocol import PageState
from .providers import ToolCall


async def observe_same_page_click(
    call: ToolCall,
    result: ActionResult,
    page: PageState,
    bridge: AgentBridge,
    settings: AgentSettings,
    step: int,
    *,
    audit: bool,
) -> ActionResult:
    """Wait, re-read, and attach state when a click mutates the same document."""
    if call.tool != "click" or not result.success or result.page_changed:
        return result
    wait = ToolCall(
        tool="wait_for_stable",
        arguments={"timeout_ms": 10_000},
        snapshot_id=page.snapshot_id,
        reason="Wait for a successful same-page click to settle.",
    )
    stable = await _execute(wait, bridge, settings, step, page.snapshot_id)
    await bridge.timeline(wait, stable, step)
    if audit:
        await bridge.audit(wait, stable, "not_required")
    if not stable.success:
        return result.model_copy(
            update={
                "data": {**result.data, "dom_observation": "stability_timeout"}
            }
        )

    read = ToolCall(
        tool="get_page_state",
        arguments={},
        snapshot_id=page.snapshot_id,
        reason="Observe DOM changes produced by the successful same-page click.",
    )
    refreshed = await _execute(read, bridge, settings, step, page.snapshot_id)
    await bridge.timeline(read, refreshed, step)
    if audit:
        await bridge.audit(read, refreshed, "not_required")
    raw = refreshed.data.get("page_state")
    if not refreshed.success or not isinstance(raw, dict):
        return result
    try:
        current = PageState.model_validate(raw)
    except Exception:
        return result
    changed = page_signature(current) != page_signature(page)
    document_changed = bool(
        page.document_id
        and current.document_id
        and page.document_id != current.document_id
    )
    return result.model_copy(
        update={
            "snapshot_id": current.snapshot_id,
            "page_changed": document_changed,
            "data": {
                **result.data,
                "dom_changed": changed,
                "page_state": current.model_dump(),
            },
        }
    )


async def observe_navigation(
    call: ToolCall,
    result: ActionResult,
    page: PageState,
    bridge: AgentBridge,
    settings: AgentSettings,
    step: int,
    *,
    audit: bool,
) -> ActionResult:
    """Stabilize and read a successful adapter-owned browser navigation."""
    if call.tool != "navigate" or not result.success or not result.page_changed:
        return result
    wait = ToolCall(
        tool="wait_for_stable",
        arguments={"timeout_ms": 10_000},
        snapshot_id=result.snapshot_id or page.snapshot_id,
        reason="Page changed after navigation; wait before reading state.",
    )
    stable = await _execute(
        wait, bridge, settings, step, result.snapshot_id or page.snapshot_id
    )
    await bridge.timeline(wait, stable, step)
    if audit:
        await bridge.audit(wait, stable, "not_required")
    if not stable.success:
        return result.model_copy(
            update={
                "data": {
                    **result.data,
                    "navigation_observation": "stability_timeout",
                }
            }
        )

    read = ToolCall(
        tool="get_page_state",
        arguments={},
        snapshot_id=stable.snapshot_id or result.snapshot_id or page.snapshot_id,
        reason="Read the fresh PageState after successful navigation.",
    )
    refreshed = await _execute(
        read, bridge, settings, step, read.snapshot_id
    )
    await bridge.timeline(read, refreshed, step)
    if audit:
        await bridge.audit(read, refreshed, "not_required")
    raw = refreshed.data.get("page_state")
    if not refreshed.success or not isinstance(raw, dict):
        return result.model_copy(
            update={
                "data": {
                    **result.data,
                    "navigation_observation": "page_state_unavailable",
                }
            }
        )
    return result.model_copy(
        update={
            "snapshot_id": refreshed.snapshot_id,
            "data": {
                **result.data,
                "navigation_observation": "complete",
                "page_state": raw,
            },
        }
    )


async def _execute(
    call: ToolCall,
    bridge: AgentBridge,
    settings: AgentSettings,
    step: int,
    snapshot_id: str,
) -> ActionResult:
    try:
        async with asyncio.timeout(settings.step_timeout_seconds):
            return await bridge.execute(call, step, settings.max_steps)
    except TimeoutError:
        return ActionResult(
            success=False,
            action=call.tool,
            snapshot_id=snapshot_id,
            error="timeout",
            message="Transition observation exceeded the per-step timeout.",
            duration_ms=int(settings.step_timeout_seconds * 1000),
        )


async def observe_stale_snapshot(result, page, bridge, settings, step, *, audit):
    """Read only; never replay an action rejected against an expired snapshot."""
    read = ToolCall(tool="get_page_state", arguments={"full": True},
        snapshot_id=page.snapshot_id, reason="Refresh the expired snapshot before choosing the next action.")
    refreshed = await _execute(read, bridge, settings, step, page.snapshot_id)
    await bridge.timeline(read, refreshed, step)
    if audit:
        await bridge.audit(read, refreshed, "stale_snapshot_observation")
    raw = refreshed.data.get("page_state")
    if not refreshed.success or not isinstance(raw, dict):
        return result
    try:
        current = PageState.model_validate(raw)
    except ValueError:
        return result
    if (current.tab_id, current.document_id, current.navigation_id, current.url) != (
            page.tab_id, page.document_id, page.navigation_id, page.url):
        return result  # A refresh does not authorize moving to a different owner.
    return result.model_copy(update={"data":{**result.data, "page_state": current.model_dump(),
        "stale_snapshot_refreshed": True}, "message":
        "The action was not executed. A fresh full snapshot is attached. Reassess the goal and current controls; use its snapshot_id for the next decision."})
