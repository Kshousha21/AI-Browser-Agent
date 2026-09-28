"""Fresh observation after take-over without retaining actionable old page state."""
import asyncio
from .agent_runtime import refine_page
from .protocol import PageState
from .providers import ToolCall
from .session_control import ControlInterrupted


async def recover_page(control, bridge, settings, page, adapters, limits, step):
    while True:
        await control.wait_ready()
        call = ToolCall(tool="get_page_state", arguments={"full": True},
            snapshot_id=page.snapshot_id, reason="Resume requires a fresh full observation of the current tab.")
        try:
            async with asyncio.timeout(settings.step_timeout_seconds):
                result = await bridge.execute(call, step, settings.max_steps)
        except ControlInterrupted:
            continue
        await bridge.timeline(call, result, step)
        await bridge.audit(call, result, "resume_observation")
        raw = result.data.get("page_state")
        if result.success and isinstance(raw, dict):
            current = refine_page(PageState.model_validate(raw), adapters, limits)
            if not current.looks_like_login:
                await control.complete_refresh()
                return current
            reason = "Sign in manually, then Resume"
        else:
            reason = "Current tab is outside supported scope or unavailable. Return to an allowed tab and Resume."
        control.invalidate(reason, state="waiting")
        await control.notify()
