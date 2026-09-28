"""WebSocket implementation of the agent loop's browser bridge."""

from __future__ import annotations

import asyncio
import time

from fastapi import WebSocket

from .actions import ActionResult
from .audit import AuditLogger
from .config import SecuritySettings
from .protocol import (
    ConfirmationRequestPayload,
    ConfirmationResponsePayload,
    RunStatusPayload,
    TimelineStepPayload,
    ToolRequestPayload,
    ToolResultPayload,
    make_envelope,
)
from .providers import ToolCall
from .product_scope import product_url_allowed
from .protocol import PageState, ControlStatePayload
from .session_control import RunControl, ControlInterrupted
from .tab_control import ApprovedTabClose, safe_tabs
from .security import ConfirmationChoice, RiskAssessment


class WebSocketBridge:
    """Send tool requests and correlate asynchronous client results."""

    tab_bound = True  # The installed side panel owns one Chrome tab.

    def __init__(
        self,
        websocket: WebSocket,
        run_id: str,
        turn_id: str,
        session_id: str,
        audit_logger: AuditLogger,
        *,
        security: SecuritySettings | None = None,
    ) -> None:
        self.websocket = websocket
        self.run_id = run_id
        self.turn_id = turn_id
        self.session_id = session_id
        self.audit_logger = audit_logger
        self.confirmation_policy = (security or SecuritySettings()).confirmation_policy
        self.control = RunControl()
        self.control.on_state = self.control_status
        self.target: PageState | None = None
        self._retired: set[str] = set()
        self._started = time.monotonic()
        self._stopped = False
        self._pending: dict[str, asyncio.Future[ActionResult]] = {}
        self._confirmations: dict[str, asyncio.Future[ConfirmationChoice]] = {}
        self._send_lock = asyncio.Lock()
        self.document_session = None
        self.user_file_action = False
        self.research = False
        self.research_specific = False  # Trusted single-card admission, never a wire option.
        self.research_capture = False
        self.research_source_url = None
        self.preview_snapshot_failure: tuple[str, int] | None = None
        self.document_diagnostic = {}

    async def offer_document_results(self, page, result, allowlist):
        if self.document_session is not None:
            await self.document_session.offer(self, page, result, allowlist)
            if self.research_capture and self.document_session.selection.offer_id:
                from document_workflow.research import ResearchResultsReady
                raise ResearchResultsReady()

    @property
    def stopped(self) -> bool:
        return self._stopped

    def stop(self) -> None:
        self._stopped = True
        self.control.invalidate("Run stopped", state="stopped")
        for future in self._confirmations.values():
            if not future.done():
                future.set_result("cancel")

    async def send(self, type_: str, payload: object, corr_id: str | None = None) -> str:
        if self.research and type_ in {"document_choices", "document_progress", "document_ready", "document_done", "research_plan", "research_file"} and isinstance(payload, dict):
            payload = {**payload, "research_id": self.turn_id}
        envelope = make_envelope(type_, payload, corr_id=corr_id)  # type: ignore[arg-type]
        async with self._send_lock:
            await self.websocket.send_text(envelope.model_dump_json())
        return envelope.id

    async def control_status(self) -> None:
        await self.send("control_state", ControlStatePayload(
            run_id=self.run_id, state=self.control.state,
            epoch=self.control.epoch, reason=self.control.reason,
        ), self.turn_id)

    async def session_event(self, event: str) -> None:
        await asyncio.to_thread(self.audit_logger.record_event,
            session_id=self.session_id, run_id=self.run_id, event=event,
            tab_id=self.target.tab_id if self.target else None,
            epoch=self.control.epoch)

    async def execute(
        self, call: ToolCall, step: int, max_steps: int
    ) -> ActionResult:
        refreshing = self.control.state == "resuming" and call.tool == "get_page_state" and call.arguments.get("full") is True
        self.control.check(refreshing=refreshing)
        if call.tool == "return_research_results":
            from document_workflow.models import ApprovedResearchReturn
            if (not isinstance(call, ApprovedResearchReturn) or not self.research
                    or call.arguments.get("research_id") != self.turn_id
                    or not self.research_source_url or call.arguments.get("url") != self.research_source_url):
                return ActionResult(success=False, action=call.tool, snapshot_id=call.snapshot_id,
                    error="permission_denied", message="No current research return authority.")
        if "user_action_id" in call.arguments:
            if (not self.user_file_action or call.arguments["user_action_id"] != self.turn_id
                    or call.tool not in {"open_attachment_preview", "inspect_attachment_preview", "acquire_attachment", "download_attachment"}):
                return ActionResult(success=False, action=call.tool, snapshot_id=call.snapshot_id,
                    error="permission_denied", message="No matching explicit file action.")
        if call.tool == "download_attachment":
            from document_workflow.models import ApprovedUserDownload
            if (not isinstance(call, ApprovedUserDownload) or not self.user_file_action
                    or call.arguments.get("user_action_id") != self.turn_id):
                return ActionResult(success=False, action=call.tool, snapshot_id=call.snapshot_id,
                    error="permission_denied", message="A current explicit user download is required.")
        if call.tool == "acquire_attachment":
            from document_workflow.models import ApprovedAcquisition
            if not isinstance(call, ApprovedAcquisition):
                return ActionResult(success=False, action=call.tool, snapshot_id=call.snapshot_id,
                    error="permission_denied", message="Explicit selection and one-shot approval are required.")
        if call.tool in {"open_attachment_preview", "inspect_attachment_preview"}:
            from document_workflow.models import ApprovedPreview
            if not isinstance(call, ApprovedPreview):
                return ActionResult(success=False, action=call.tool, snapshot_id=call.snapshot_id,
                    error="permission_denied", message="Explicit selected-preview authority is required.")
        if (self.document_session is not None and not (self.user_file_action and (not self.research or self.research_specific))
                and call.tool not in {"get_tabs", "read_page", "find_text", "get_element_details"}):
            await self.document_session.invalidate(self)
        if call.tool in {"navigate", "open_tab"} and not product_url_allowed(str(call.arguments.get("url", ""))):
            return ActionResult(success=False, action=call.tool, snapshot_id=call.snapshot_id,
                error="navigation_blocked", message="Destination is outside the supported product scope.")
        if call.tool == "close_tab" and not isinstance(call, ApprovedTabClose):
            return ActionResult(success=False, action=call.tool, snapshot_id=call.snapshot_id,
                error="permission_denied", message="An exact one-shot tab approval is required.")
        request = ToolRequestPayload(
            run_id=self.run_id, step=step, max_steps=max_steps, tool=call.tool,
            arguments=call.arguments, snapshot_id=call.snapshot_id, reason=call.reason,
            tab_id=self.target.tab_id if self.target else None,
            document_id=self.target.document_id if self.target else "",
            control_epoch=self.control.epoch,
            reset_target=refreshing or self.target is None,
            tab_revision=call.tab_revision if isinstance(call, ApprovedTabClose) else None,
        )
        envelope = make_envelope("tool_request", request, corr_id=self.turn_id)
        future = asyncio.get_running_loop().create_future()
        self._pending[envelope.id] = future
        try:
            async with self._send_lock:
                self.control.check(refreshing=refreshing)
                await self.websocket.send_text(envelope.model_dump_json())
            result = await self.control.interruptible(future, refreshing=refreshing)
            if not result.success and call.tool in {"get_page_state", "acquire_attachment", "download_attachment", "open_attachment_preview", "inspect_attachment_preview"}:
                from document_workflow.diagnostics import safe_diagnostic
                self.document_diagnostic = safe_diagnostic(result.data)
            if "tabs" in result.data:
                result = result.model_copy(update={"data": {"tabs": safe_tabs(result.data["tabs"], internal=True)}})
            raw = result.data.get("page_state")
            if result.success and isinstance(raw, dict):
                page = PageState.model_validate(raw)
                if not product_url_allowed(page.url):
                    return ActionResult(success=False, action=call.tool, snapshot_id=call.snapshot_id,
                        error="permission_denied", message="Current tab is outside supported scope.")
                # Only a refresh/tab operation may establish a new target.
                expected = request.tab_id
                if not refreshing and call.tool not in {"switch_tab", "open_tab", "close_tab"} and expected is not None and page.tab_id != expected:
                    result = ActionResult(success=False, action=call.tool, snapshot_id=call.snapshot_id,
                        error="target_tab_changed", message="Observation belongs to another tab.")
                else:
                    self.target = page
            # An expired observation is not a user Pause or a changed owner. Return
            # ordinary browser failures to the agent for a fresh read and replan.
            # Document operations retain their one-shot interruption semantics.
            recoverable_stale = (result.error == "stale_snapshot" and not self.research
                and not self.user_file_action and call.tool in {
                    "navigate", "click", "type_text", "clear_input", "press_key", "scroll",
                    "read_page", "find_text", "get_element_details"})
            if result.error in {"stale_snapshot", "target_tab_changed", "target_document_changed", "target_tab_closed", "run_paused"} and not recoverable_stale:
                await self.timeline(call, result, step)
                await self.audit(call, result, "target_invalidated")
                await self.session_event("stale_action_rejected")
                self.control.invalidate("Target changed - resume to read fresh state")
                if call.tool in {"open_attachment_preview", "inspect_attachment_preview"} and result.error == "stale_snapshot":
                    # Safe terminal diagnosis only; invalidation/cancellation still wins.
                    self.preview_snapshot_failure = (call.snapshot_id, self.control.epoch)
                await self.control.notify()
                raise ControlInterrupted()
            if result.success and result.data.get("page_access") is False:
                self.control.invalidate("Outside supported scope - select an allowed tab and Resume", state="waiting")
                await self.control.notify()
            return result
        finally:
            self._pending.pop(envelope.id, None)
            self._retired.add(envelope.id)
            if len(self._retired) > 200:
                self._retired = {envelope.id}

    def resolve(self, corr_id: str | None, payload: ToolResultPayload) -> bool:
        if corr_id in self._retired:
            return True
        if not corr_id:
            return False
        future = self._pending.get(corr_id)
        if future is None or future.done():
            return False
        future.set_result(payload.result)
        return True

    async def confirm(
        self,
        call: ToolCall,
        assessment: RiskAssessment,
        step: int,
    ) -> ConfirmationChoice:
        self.control.check()
        if "delete" not in assessment.reasons:
            # Trusted local configuration; callers still revalidate the exact target.
            return "policy_allow_once"
        payload = ConfirmationRequestPayload(
            run_id=self.run_id,
            step=step,
            tool=call.tool,
            target=assessment.target,
            arguments=call.arguments,
            reason=call.reason,
            risk="high",
            risk_reasons=list(assessment.reasons),
            scope=assessment.scope,
        )
        envelope = make_envelope(
            "confirmation_request", payload, corr_id=self.turn_id
        )
        future: asyncio.Future[ConfirmationChoice] = (
            asyncio.get_running_loop().create_future()
        )
        self._confirmations[envelope.id] = future
        try:
            async with self._send_lock:
                self.control.check()
                await self.websocket.send_text(envelope.model_dump_json())
            choice = await self.control.interruptible(future)
            if "delete" in assessment.reasons and choice == "allow_run":
                return "allow_once"  # Every deletion needs its own decision.
            return choice
        finally:
            self._confirmations.pop(envelope.id, None)
            self._retired.add(envelope.id)

    def resolve_confirmation(
        self,
        corr_id: str | None,
        payload: ConfirmationResponsePayload,
    ) -> bool:
        if corr_id in self._retired:
            return True
        if not corr_id:
            return False
        future = self._confirmations.get(corr_id)
        if future is None or future.done():
            return False
        future.set_result(payload.decision if self.control.state == "running" else "cancel")
        return True

    async def audit(
        self,
        call: ToolCall,
        result: ActionResult,
        confirmation: str,
        password_target: bool = False,
    ) -> None:
        if call.tool in {"acquire_attachment", "download_attachment", "open_attachment_preview", "inspect_attachment_preview", "return_research_results"}:
            call = call.model_copy(update={"arguments": {"acquisition_id": call.arguments.get("acquisition_id")}})
            result = result.model_copy(update={"data": {}, "element_name": None, "element_id": None,
                                              "message": "Local attachment acquisition status."})
        await asyncio.to_thread(
            self.audit_logger.record,
            session_id=self.session_id,
            run_id=self.run_id,
            call=call,
            result=result,
            confirmation=confirmation,
            password_target=password_target,
        )

    async def status(
        self,
        status: str,
        step: int,
        max_steps: int,
        elapsed_ms: int,
        usage: dict[str, int],
    ) -> None:
        await self.send(
            "run_status",
            RunStatusPayload(
                run_id=self.run_id,
                status=status,
                step=step,
                max_steps=max_steps,
                elapsed_ms=elapsed_ms,
                usage=usage,
            ),
            self.turn_id,
        )

    async def timeline(
        self, call: ToolCall, result: ActionResult, step: int
    ) -> None:
        if call.tool in {"acquire_attachment", "download_attachment", "open_attachment_preview", "inspect_attachment_preview", "return_research_results"}:
            from document_workflow.diagnostics import safe_diagnostic
            diagnostic = safe_diagnostic(result.data)
            label = "Local attachment acquisition status."
            if diagnostic.get("stage"):
                label += " Stage: " + diagnostic["stage"] + "."
            result = result.model_copy(update={"data": diagnostic, "element_name": None, "element_id": None,
                                              "message": label})
        await self.send(
            "timeline_step",
            TimelineStepPayload(
                run_id=self.run_id,
                step=step,
                tool=call.tool,
                target=result.element_name or result.element_id or "",
                reason=call.reason,
                result=result,
            ),
            self.turn_id,
        )

    async def close(self) -> None:
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        for future in self._confirmations.values():
            if not future.done():
                future.cancel()
        self._confirmations.clear()
