"""One explicit UI file action. No provider, arbitrary URL, or model authority."""
import asyncio
import json
import secrets
from dataclasses import dataclass, field, replace
from pydantic import ValidationError
from browser.session_control import ControlInterrupted
from browser.state_reads import state_identity
from .models import (ApprovedAcquisition, ApprovedUserDownload, DocumentError,
                     FileActionRequest, UserDownloadReceipt)
from .selection import SelectedAttachment, digest, resolve_attachment


@dataclass(frozen=True, repr=False)
class DocumentOperation:
    """Admitted task/document authority, not an expiring result-ref cache.

    Created only by trusted application code after fresh selection admission.
    The existing user_action_id nonce scopes the operation; the source binding
    scopes one document in a bounded cohort. JSON makes nested evidence immutable.
    A result_ref carried on the wire is only exact observation lineage thereafter.
    """
    operation_id: str
    action: str
    epoch: int
    source_identity: tuple
    source_url: str = field(repr=False)
    binding: str = field(repr=False)

    @staticmethod
    def _binding(selected):
        return json.dumps({"url": selected.url, "preview": selected.preview,
            "source_fingerprint": selected.source_fingerprint,
            "metadata": {key: value for key, value in selected.metadata.items() if key != "result_ref"}},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def admitted(cls, operation_id, action, control, page, selected):
        control.check()
        if control.needs_refresh or not page.context_is_current():
            raise DocumentError("stale_result_ref")
        return cls(operation_id, action, control.epoch, state_identity(page)[:3],
                   page.url, cls._binding(selected))

    def check(self, bridge, action):
        bridge.control.check()
        if (self.operation_id != bridge.turn_id or self.action != action or
                bridge.control.needs_refresh or self.epoch != bridge.control.epoch):
            raise DocumentError("stale_result_ref")

    def bind_observation(self, page, selected):
        # A controlled research return may create a new document/snapshot; it
        # cannot change the owning tab, observed source, or admitted attachment.
        if (not page.context_is_current() or page.tab_id != self.source_identity[0]
                or page.url != self.source_url or self._binding(selected) != self.binding):
            raise DocumentError("stale_result_ref")
        bound = json.loads(self.binding)
        bound["metadata"]["result_ref"] = selected.result_ref
        return SelectedAttachment(result_ref=selected.result_ref, index=selected.index, **bound)


async def run_file_action(session, raw, bridge, *, authority=None):
    prepared = []
    active = None
    action_id = bridge.turn_id
    reusable = authority is None

    async def send(kind, payload):
        await bridge.send(kind, {**payload, "user_action_id": action_id})

    async def progress(stage):
        await send("document_progress", {"stage": stage, "run_id": bridge.run_id})

    try:
        request = FileActionRequest.model_validate(raw)
        if not isinstance(action_id, str) or not action_id or len(action_id) > 128:
            raise DocumentError("invalid_document_request")
        try:
            if authority is None:
                original, selected = session.selection.admit_file(request, session.allowlist, action_id)
                authority = DocumentOperation.admitted(action_id, request.action, bridge.control, original, selected)
            else:
                # Only the internal, already-admitted research cohort may enter
                # here. External requests cannot supply an authority object.
                if not isinstance(authority, DocumentOperation):
                    raise DocumentError("invalid_document_request")
                authority.check(bridge, request.action)
                original = session.selection.page
                if original is None or request.offer_id != session.selection.offer_id:
                    raise DocumentError("stale_result_ref")
                selected = resolve_attachment(original, request.result_ref, session.selection.refs,
                    session.allowlist, file_action=True, allow_non_pdf=request.action != "prepare")
            selected = authority.bind_observation(original, selected)
        finally:
            if not reusable:
                session.selection.clear()
                await send("document_choices", {"offer_id": None, "results": [],
                    "state": "consumed", "trust": "untrusted_webpage_data"})
        bridge.target = original
        def current():
            authority.check(bridge, request.action)

        current()
        if request.action == "preview" and selected.preview is None:
            raise DocumentError("attachment_preview_only")
        page = (await session.observe_file_click(bridge, original) if reusable
                else await session.refresh(bridge, original))
        current()
        preview = selected if selected.preview is not None else None
        preview_id = secrets.token_hex(16)
        if preview is not None:
            await progress("locating_download")
            page, url = await session.preview_observation(bridge, preview, preview_id, opening=True,
                user_action_id=action_id, allow_non_pdf=request.action != "prepare")
            current()
            selected = replace(selected, url=url,
                source_fingerprint=digest({"selection": selected.source_fingerprint, "observed_download": url}))
        info = {"action": request.action,
                "title": selected.metadata["attachment_filename"],
                "file_type": selected.metadata.get("file_type")}
        if request.action == "preview":
            await send("document_done", {"status": "preview_opened", "file": info,
                "prepared": [], "answer": None})
            return
        if preview is not None:
            page, _ = await session.preview_observation(bridge, preview, preview_id, expected_url=selected.url,
                user_action_id=action_id, allow_non_pdf=request.action != "prepare")
        current()
        nonce = secrets.token_hex(16)
        common = {"url": selected.url, "user_action_id": action_id, "user_result_ref": selected.result_ref,
            **({"preview_id": preview_id} if preview is not None else {})}
        if request.action == "download":
            active = ApprovedUserDownload(tool="download_attachment", arguments={**common,
                "download_action_id": nonce, "filename": selected.metadata["attachment_filename"]},
                snapshot_id=page.snapshot_id, reason="Download only the attachment explicitly clicked by the user.")
        else:
            active = ApprovedAcquisition(tool="acquire_attachment", arguments={**common, "acquisition_id": nonce},
                snapshot_id=page.snapshot_id, reason="Prepare only the PDF explicitly clicked by the user.")
        await progress("acquiring")
        current()
        result = await bridge.control.interruptible(bridge.execute(active, 1, 1))
        current()
        if not result.success:
            raise DocumentError(result.data.get("document_error", "download_failed"))
        receipt = UserDownloadReceipt.model_validate(result.data) if request.action == "download" else None
        await session.audit(bridge, active, True, "download_completed", "user_click")
        active = None
        if request.action == "download":
            info["download_id"] = receipt.download_id
            await send("document_done", {"status": "downloaded", "file": info,
                "prepared": [], "answer": None})
            return
        loop = asyncio.get_running_loop()

        def status(stage):
            current()
            asyncio.run_coroutine_threadsafe(progress(stage), loop).result()

        worker = asyncio.create_task(asyncio.to_thread(session.services.prepare,
            result.data, nonce, selected, status, check=current))
        try:
            row = await bridge.control.interruptible(asyncio.shield(worker))
        except (ControlInterrupted, asyncio.CancelledError) as error:
            if isinstance(error, asyncio.CancelledError):
                bridge.stop()
            # A native parser/SQLite call can finish safely in its owned thread.
            # Do not delay Stop/Clear acknowledgment or publish its late result.
            worker.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            raise
        current()
        prepared.append(row)
        session.ready[row["document_id"]] = {"document_id": row["document_id"], "index_status": "ready", "title": row.get("title", "Selected PDF")}
        while len(session.ready) > 10:
            session.ready.popitem(last=False)
        info["document_id"] = row["document_id"]
        await send("document_ready", {"documents": list(session.ready.values())})
        await send("document_done", {"status": "ready", "file": info,
            "prepared": prepared, "answer": None})
    except (DocumentError, ValidationError, ControlInterrupted, TimeoutError) as error:
        code = error.code if isinstance(error, DocumentError) else (
            "invalid_document_request" if isinstance(error, ValidationError) else "operation_interrupted")
        if code in {"stale_result_ref", "stale_snapshot", "preview_identity_mismatch", "attachment_preview_only",
                    "unsupported_attachment_type", "read_only_blocked", "attachment_unavailable"} or isinstance(error, ControlInterrupted):
            await session.invalidate(bridge)
        if active is not None:
            await session.audit(bridge, active, False, code, "not_executed")
        await send("document_done", {"status": "failed", "error": code,
            "diagnostic": getattr(bridge, "document_diagnostic", {}), "prepared": prepared, "answer": None})
    except asyncio.CancelledError:
        raise
    except Exception:
        await send("document_done", {"status": "failed", "error": "document_operation_failed",
            "prepared": prepared, "answer": None})
    finally:
        await bridge.close()
