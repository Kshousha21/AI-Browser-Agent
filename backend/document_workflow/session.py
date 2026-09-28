"""User-originated preparation/answer runs sharing browser confirmation/control."""
import asyncio
from collections import OrderedDict
import secrets
from dataclasses import replace
from pydantic import ValidationError
from app.site_adapters import SiteAdapterRegistry
from browser.actions import ActionResult
from browser.prompting import clamp_page_state
from browser.protocol import PageState
from browser.providers import ToolCall
from browser.security import PermissionManager, RiskAssessment
from browser.session_control import ControlInterrupted
from browser.state_reads import execute_state_read, state_identity
from app.site_adapters.page_files import TOOL_NAME
from local_rag import RagError
from .models import ApprovedAcquisition, ApprovedPreview, DocumentError, PrepareRequest, QuestionRequest, RefreshRequest
from .selection import CurrentSelection, verify_refresh, digest
from .preview import validate_download
from .services import DocumentServices


class DocumentSession:
    def __init__(self, settings, services_factory=None):
        self.settings = settings
        self.selection = CurrentSelection()
        self._services_factory = services_factory or (lambda: DocumentServices(settings.provider, visual_settings=settings.visual_retrieval, multimodal_settings=settings.multimodal_retrieval))
        self._services = None
        self.ready = OrderedDict()
        self.allowlist = []
        self.target = None  # Owner anchor only; never authorizes reuse of an expired offer.
        self.active_document_ids = ()
        self.active_document_owner = None
        self.last_document_question = None

    def clear_document_context(self):
        self.active_document_ids = ()
        self.active_document_owner = None
        self.last_document_question = None

    def document_context(self, page):
        owner = (*state_identity(page)[:3], page.url)
        if (owner != self.active_document_owner or
                any(identifier not in self.ready for identifier in self.active_document_ids)):
            self.clear_document_context()
        return self.active_document_ids

    def remember_document_question(self, question, identifiers, page):
        if not identifiers or any(identifier not in self.ready for identifier in identifiers):
            raise DocumentError("document_not_ready")
        self.active_document_owner = (*state_identity(page)[:3], page.url)
        self.active_document_ids = tuple(identifiers)
        self.last_document_question = question

    def document_question(self, original):
        # Exact retry controls are resolved only AFTER model routing to DOCUMENT.
        # They never classify intent, select sources, or preserve remote consent.
        if original.strip().casefold().rstrip(".!?") in {"try again", "retry", "again", "please try again"}:
            if self.last_document_question is None:
                raise DocumentError("document_retry_unavailable")
            return self.last_document_question
        return original

    @property
    def services(self):
        if self._services is None:
            self._services = self._services_factory()
        return self._services

    async def offer(self, bridge, page, result, allowlist):
        self.allowlist = list(allowlist)
        payload = self.selection.offer(page, result)
        if payload:
            self.target = page
            await bridge.send("document_choices", payload)

    async def invalidate(self, bridge):
        had_selection = self.selection.offer_id is not None
        self.selection.clear()
        if had_selection:
            await bridge.send("document_choices", {"offer_id": None, "results": [], "state": "invalidated", "trust": "untrusted_webpage_data"})

    async def refresh_choices(self, raw, bridge):
        refresh_id = bridge.turn_id
        try:
            RefreshRequest.model_validate(raw)
            await self.invalidate(bridge)
            if self.target is None or self.target.tab_id is None:
                raise DocumentError("no_current_results")
            owner = self.target.tab_id
            bridge.target = self.target
            call = ToolCall(tool="get_page_state", arguments={"full": True},
                snapshot_id=self.target.snapshot_id, reason="Refresh current document choices without searching.")
            async with asyncio.timeout(self.settings.agent.step_timeout_seconds):
                result = await bridge.control.interruptible(bridge.execute(call, 1, 1))
            bridge.control.check()
            if not result.success or not isinstance(result.data.get("page_state"), dict):
                raise DocumentError("stale_result_ref")
            page = PageState.model_validate(result.data["page_state"])
            if page.tab_id != owner or not page.document_id or not page.snapshot_id:
                raise DocumentError("stale_result_ref")
            adapters = SiteAdapterRegistry(self.settings.site_adapters)
            page = adapters.refine_page_state(clamp_page_state(page, self.settings.page))
            bridge.target = page
            read = ToolCall(tool=TOOL_NAME, arguments={"limit": 10}, snapshot_id=page.snapshot_id,
                           reason="Publish freshly observed current-page choices.")
            read = adapters.prepare_state_read(page.url, read)
            if read is None:
                raise DocumentError("no_current_results")
            projection = execute_state_read(read, page, target=bridge.target,
                decision_identity=state_identity(page), decision_epoch=bridge.control.epoch,
                control=bridge.control, adapters=adapters, allowlist=self.allowlist)
            if not projection.success:
                raise DocumentError("no_current_results")
            payload = self.selection.offer(page, projection)
            bridge.control.check()
            if not payload:
                raise DocumentError("no_current_results")
            self.target = page
            await bridge.send("document_choices", {**payload, "refresh_id": refresh_id})
            bridge.control.check()
            await bridge.send("document_done", {"status": "refreshed", "refresh_id": refresh_id})
        except (DocumentError, ValidationError, ControlInterrupted, TimeoutError) as error:
            await self.invalidate(bridge)
            code = error.code if isinstance(error, DocumentError) else (
                "invalid_document_request" if isinstance(error, ValidationError) else "operation_interrupted")
            await bridge.send("document_done", {"status": "failed", "error": code, "refresh_id": refresh_id})
        except Exception:
            await self.invalidate(bridge)
            await bridge.send("document_done", {"status": "failed", "error": "document_operation_failed", "refresh_id": refresh_id})
        finally:
            await bridge.close()

    async def refresh(self, bridge, original):
        bridge.control.check()
        if bridge.control.needs_refresh:
            raise DocumentError("stale_result_ref")
        call = ToolCall(tool="get_page_state", arguments={"full": True, "selection_snapshot": bridge.target.snapshot_id}, snapshot_id=bridge.target.snapshot_id,
                        reason="Verify the selected attachment before local analysis.")
        result = await bridge.control.interruptible(bridge.execute(call, 1, 1))
        if not result.success or not isinstance(result.data.get("page_state"), dict):
            raise DocumentError("stale_result_ref")
        raw = PageState.model_validate(result.data["page_state"])
        current = SiteAdapterRegistry(self.settings.site_adapters).refine_page_state(clamp_page_state(raw, self.settings.page))
        verify_refresh(original, current)
        bridge.target = current
        return current

    async def observe_file_click(self, bridge, original):
        """A new explicit click observes anew; it never renews an in-flight grant.

        Only this session's last verified owner document may be used. After our
        own preview navigation it is the verified preview, not a fabricated
        search observation. Any external navigation invalidates the catalogue.
        """
        anchor = self.target
        if anchor is None or anchor.tab_id != original.tab_id:
            raise DocumentError("stale_result_ref")
        bridge.target = anchor
        call = ToolCall(tool="get_page_state", arguments={"full": True},
            snapshot_id=anchor.snapshot_id, reason="Observe the owning document for this explicit file click.")
        result = await bridge.control.interruptible(bridge.execute(call, 1, 1))
        bridge.control.check()
        if not result.success or not isinstance(result.data.get("page_state"), dict):
            raise DocumentError("stale_result_ref")
        raw = PageState.model_validate(result.data["page_state"])
        current = SiteAdapterRegistry(self.settings.site_adapters).refine_page_state(clamp_page_state(raw, self.settings.page))
        if state_identity(anchor)[:3] != state_identity(current)[:3] or anchor.url != current.url:
            raise DocumentError("stale_result_ref")
        if anchor.url == original.url:
            verify_refresh(original, current)
        bridge.target = current
        self.target = current
        return current

    async def preview_observation(self, bridge, selection, nonce, opening=False, expected_url=None, *, user_action_id=None, allow_non_pdf=False):
        before = bridge.target
        call = ApprovedPreview(tool="open_attachment_preview" if opening else "inspect_attachment_preview",
            arguments={"preview_id": nonce, "url": selection.url, "attachment": selection.preview,
                **({"user_action_id": user_action_id, "user_result_ref": selection.result_ref} if user_action_id is not None else {})},
            snapshot_id=before.snapshot_id, reason="Locate the selected preview's Download control only.")
        try:
            result = await bridge.control.interruptible(bridge.execute(call, 1, 1))
        except ControlInterrupted:
            if getattr(bridge, "preview_snapshot_failure", None) == (call.snapshot_id, bridge.control.epoch):
                raise DocumentError("stale_snapshot") from None
            raise
        bridge.control.check()
        if user_action_id is not None:
            await self.audit(bridge, call, result.success,
                "preview_observed" if result.success else "operation_interrupted", "user_click")
        if not result.success:
            # Preserve the original safe typed snapshot failure for the card;
            # elapsed expiry, cancellation and target changes are not synonyms.
            raise DocumentError(result.data.get("document_error") or
                ("stale_snapshot" if result.error == "stale_snapshot" else "operation_interrupted"))
        page = PageState.model_validate(result.data.get("page_state"))
        if (page.tab_id != before.tab_id or page.url != selection.url or not page.document_id
                or not page.snapshot_id or page.looks_like_login
                or (not opening and state_identity(page)[:3] != state_identity(before)[:3])):
            raise DocumentError("preview_identity_mismatch")
        observed = result.data.get("preview", {})
        url = validate_download(observed.get("url"), selection.url, selection.preview,
                                observed.get("attachment"), self.allowlist, allow_non_pdf=allow_non_pdf)
        if expected_url is not None and url != expected_url:
            raise DocumentError("preview_identity_mismatch")
        bridge.target = page
        self.target = page  # A later explicit Refresh observes this page, never old search refs.
        return page, url

    async def run(self, kind, raw, bridge):
        if kind == "document_file_action":
            from .file_actions import run_file_action
            await run_file_action(self, raw, bridge)
            return
        if kind == "document_refresh":
            await self.refresh_choices(raw, bridge)
            return
        prepared = []
        active_call = None
        try:
            if kind == "document_prepare":
                request = PrepareRequest.model_validate(raw)
                try:
                    original, selected = self.selection.consume(request, self.allowlist)
                finally:
                    # consume() retires the offer on failure too. Retire its UI rows
                    # before returning any error, not only after successful resolution.
                    await bridge.send("document_choices", {"offer_id": None, "results": [], "state": "consumed", "trust": "untrusted_webpage_data"})
                bridge.target = original
                permissions = PermissionManager(self.settings.security, read_only=request.read_only)
                for selection in selected:
                    preview_selection = selection if selection.preview is not None else None
                    preview_id = secrets.token_hex(16)
                    if preview_selection is not None:
                        active_call = ToolCall(tool="open_attachment_preview", arguments={"preview_id": preview_id},
                            snapshot_id=original.snapshot_id,
                            reason="Open the selected preview in this tab. Chrome may load/render the attachment; this is not network-free. Download requires separate backend authorization.")
                        preview_risk = RiskAssessment("high", ("preview_navigation",), "preview:" + preview_id,
                                                      selection.metadata.get("attachment_filename", "Selected PDF"))
                        if permissions.check(preview_risk) == "block_read_only":
                            raise DocumentError("read_only_blocked")
                        current = await self.refresh(bridge, original)
                        active_call.snapshot_id = current.snapshot_id
                        await self.progress(bridge, "preview_confirmation")
                        choice = await bridge.confirm(active_call, preview_risk, 1)
                        if choice not in {"allow_once", "policy_allow_once"}:
                            raise DocumentError("confirmation_cancelled")
                        await self.refresh(bridge, original)
                        await self.progress(bridge, "locating_download")
                        current, url = await self.preview_observation(bridge, selection, preview_id, opening=True)
                        await self.audit(bridge, active_call, True, "preview_observed", choice)
                        active_call = None
                        selection = replace(selection, url=url,
                            source_fingerprint=digest({"selection": selection.source_fingerprint, "observed_download": url}))
                    acquisition_id = secrets.token_hex(16)
                    active_call = ToolCall(tool="acquire_attachment", arguments={"acquisition_id": acquisition_id},
                                          snapshot_id=original.snapshot_id, reason="Download locally for analysis")
                    assessment = RiskAssessment("high", ("download",), "acquisition:" + acquisition_id,
                        " | ".join(str(selection.metadata.get(k) or "") for k in
                                   ("title", "attachment_filename", "parent_page_title"))[:1000])
                    if permissions.check(assessment) == "block_read_only":
                        raise DocumentError("read_only_blocked")
                    if preview_selection is not None:
                        current, _ = await self.preview_observation(bridge, preview_selection, preview_id, expected_url=selection.url)
                    else:
                        current = await self.refresh(bridge, original)
                    active_call.snapshot_id = current.snapshot_id
                    await self.progress(bridge, "confirmation_required")
                    # Always obtain backend authorization; the trusted local policy
                    # decides whether a human decision is required.
                    choice = await bridge.confirm(active_call, assessment, 1)
                    if choice not in {"allow_once", "policy_allow_once"} or not permissions.apply(assessment, choice):
                        await self.audit(bridge, active_call, False, "confirmation_cancelled", choice)
                        active_call = None
                        raise DocumentError("confirmation_cancelled")
                    if preview_selection is not None:
                        current, _ = await self.preview_observation(bridge, preview_selection, preview_id, expected_url=selection.url)
                    else:
                        current = await self.refresh(bridge, original)
                    bridge.control.check()
                    approved = ApprovedAcquisition(tool="acquire_attachment", arguments={
                        "url": selection.url, "acquisition_id": acquisition_id,
                        **({"preview_id": preview_id} if preview_selection is not None else {})},
                        snapshot_id=current.snapshot_id, reason="Download the explicitly selected attachment locally")
                    await self.progress(bridge, "acquiring")
                    async with asyncio.timeout(65):
                        download = await bridge.execute(approved, 1, 1)
                    if not download.success:
                        raise DocumentError("download_failed")
                    # Audit only safe identity/status; neither the private URL
                    # nor completed browser filename is a timeline/audit payload.
                    await self.audit(bridge, active_call, True, "download_completed", choice)
                    active_call = None
                    loop = asyncio.get_running_loop()
                    def status(stage):
                        bridge.control.check()
                        # Serialize progress before advancing the worker, so no
                        # queued stage can arrive after the completed response.
                        asyncio.run_coroutine_threadsafe(self.progress(bridge, stage), loop).result(timeout=5)
                    worker = asyncio.create_task(asyncio.to_thread(self.services.prepare,
                        download.data, acquisition_id, selection, status))
                    try:
                        row = await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        bridge.stop()
                        await asyncio.gather(worker, return_exceptions=True)
                        raise
                    prepared.append(row)
                    self.ready[row["document_id"]] = {"document_id": row["document_id"], "index_status": "ready",
                        "title": row.get("title") or selection.metadata.get("attachment_filename") or "Selected PDF"}
                    while len(self.ready) > 10:
                        self.ready.popitem(last=False)
                    await bridge.send("document_ready", {"documents": list(self.ready.values())})
                question, identifiers = request.question, [row["document_id"] for row in prepared]
                allow_remote = request.allow_remote_context
            else:
                request = QuestionRequest.model_validate(raw)
                identifiers, question, allow_remote = request.document_ids, request.question, request.allow_remote_context
                if any(identifier not in self.ready for identifier in identifiers):
                    raise DocumentError("document_not_ready")
            answer = None
            if question is not None:
                await self.progress(bridge, "answering")
                answer = await bridge.control.interruptible(self.services.answer(question, identifiers, allow_remote))
            bridge.control.check()
            answer_payload = answer.model_dump() if answer is not None else None
            if answer_payload is not None:
                for citation in answer_payload["citations"]:
                    identifier = citation["document_id"]
                    if identifier not in identifiers or identifier not in self.ready:
                        raise DocumentError("document_not_ready")
                    # Only the authorized prepared record supplies a display
                    # filename. Never accept model-supplied naming/provenance.
                    citation["filename"] = self.ready[identifier].get("title") or "Selected PDF"
            await bridge.send("document_done", {"status": "ready" if answer is None else answer.status,
                "prepared": prepared, "answer": answer_payload})
        except (DocumentError, RagError, ValidationError, ControlInterrupted, TimeoutError) as error:
            code = error.code if isinstance(error, (DocumentError, RagError)) else (
                "operation_interrupted" if isinstance(error, (ControlInterrupted, TimeoutError)) else "invalid_document_request")
            if active_call is not None:
                await self.audit(bridge, active_call, False, code, "not_executed")
            await bridge.send("document_done", {"status": "failed", "error": code, "prepared": prepared, "answer": None})
        except asyncio.CancelledError:
            raise
        except Exception:
            await bridge.send("document_done", {"status": "failed", "error": "document_operation_failed", "prepared": prepared, "answer": None})
        finally:
            await bridge.close()

    @staticmethod
    async def progress(bridge, stage):
        await bridge.send("document_progress", {"stage": stage, "run_id": bridge.run_id})

    @staticmethod
    async def audit(bridge, call, success, code, confirmation):
        result = ActionResult(success=success, action=call.tool, snapshot_id=call.snapshot_id,
                              error=None if success else "permission_denied", message=code)
        await bridge.audit(call, result, confirmation)
