"""One explicit question, a bounded current cohort, existing acquisition and RAG."""
import asyncio
import re
import unicodedata
from pydantic import ValidationError
from app.site_adapters import SiteAdapterRegistry
from app.site_adapters.page_files import retrieve, TOOL_NAME
from browser.agent import run_agent
from browser.protocol import PageState
from browser.providers import ToolCall
from browser.prompting import clamp_page_state
from browser.product_scope import product_pattern_allowed, product_url_allowed
from browser.session_control import ControlInterrupted
from browser.state_reads import state_identity
from local_rag import RagError
from .file_actions import DocumentOperation, run_file_action
from .models import DocumentError, ResearchRequest, ApprovedResearchReturn
from .selection import resolve_attachment


class ResearchResultsReady(Exception):
    """Stop metadata-agent generation before entering event-driven document work."""


def words(value):
    return set(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold()))


STOP_WORDS = words("find information about research look through relevant pdf pdfs report reports study studies "
                   "what does this that these say in on the a an of for and or to latest please compare explain")


def rank_candidates(selection, question, allowlist, limit):
    """Local metadata selection, does not change website ranking."""
    if type(limit) is not int or not 1 <= limit <= 5:
        raise DocumentError("invalid_document_request")
    if selection.page is None or selection.offer_id is None:
        raise DocumentError("stale_result_ref")
    terms = words(question) - STOP_WORDS
    scored = []
    for ref in sorted(selection.refs):
        try:
            item = resolve_attachment(selection.page, ref, selection.refs, allowlist, file_action=True)
        except DocumentError:
            continue  # Invalid/ambiguous/non-PDF records never become research authority.
        name = words(str(item.metadata.get("attachment_filename") or ""))
        metadata = words(" ".join(str(item.metadata.get(key) or "") for key in
            ("title", "attachment_filename", "parent_page_title", "space_name")))
        score = 2 * len(terms & name) + len(terms & metadata)
        if score:
            scored.append((-score, item.index, item))
    scored.sort(key=lambda value: (value[0], value[1]))
    result, seen = [], set()
    for _, _, item in scored:
        if item.url not in seen:
            result.append(item)
            seen.add(item.url)
        if len(result) == limit:
            break
    return result


class FileBridge:
    """Reuse the one-file pipeline without ending the enclosing research task."""
    def __init__(self, bridge, progress):
        object.__setattr__(self, "base", bridge)
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "done", None)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __setattr__(self, name, value):
        if name == "done":
            object.__setattr__(self, name, value)
        else:
            setattr(self.base, name, value)

    async def send(self, kind, payload, *args):
        if kind == "document_done":
            self.done = payload
        elif kind == "document_progress":
            await self.progress(payload["stage"])
        else:
            await self.base.send(kind, {**payload, "research_id": self.base.turn_id})

    async def close(self):
        pass  # The outer operation alone owns close/cancellation.

    async def execute(self, call, *args):
        call = call.model_copy(update={"reason": "Prepare only this PDF from the explicitly authorized research cohort."})
        return await self.base.execute(call, *args)

    async def audit(self, call, result, confirmation):
        return await self.base.audit(call, result, "user_research" if confirmation == "user_click" else confirmation)


async def run_research(session, raw, bridge, provider, *, prepared_ids=None):
    prepared, unavailable = [], 0
    identifier = bridge.turn_id
    epoch = bridge.control.epoch
    bridge.research_specific = False

    def current():
        bridge.control.check()
        if bridge.control.needs_refresh or bridge.control.epoch != epoch:
            raise DocumentError("research_scope_changed")

    async def send(kind, payload):
        current()
        return await bridge.send(kind, {**payload, "research_id": identifier})

    async def progress(stage, **details):
        await send("document_progress", {"stage": stage, "run_id": bridge.run_id, **details})

    def refined(raw_page):
        return SiteAdapterRegistry(session.settings.site_adapters).refine_page_state(
            clamp_page_state(raw_page, session.settings.page))

    async def publish(page):
        result = retrieve(ToolCall(tool=TOOL_NAME, arguments={"limit": 10},
            snapshot_id=page.snapshot_id, reason="Read only this current bounded research window."), page)
        if result.success:
            await session.offer(bridge, page, result, request.allowlist)
        return result.success and session.selection.offer_id is not None

    try:
        request = ResearchRequest.model_validate(raw)
        if (not product_url_allowed(request.page_state.url) or
                not all(product_pattern_allowed(value) for value in request.allowlist)):
            raise DocumentError("research_scope_changed")
        session.allowlist = list(request.allowlist)
        bridge.target = request.page_state
        current()
        if prepared_ids is not None:
            if not prepared_ids or len(prepared_ids) > session.settings.research.max_pdfs or any(i not in session.ready for i in prepared_ids):
                raise DocumentError("document_not_ready")
            prepared = [dict(session.ready[i]) for i in prepared_ids]
        elif request.document_id is not None:
            if request.document_id not in session.ready:
                raise DocumentError("document_not_ready")
            prepared = [dict(session.ready[request.document_id])]
        else:
            if request.mode == "specific":
                selection = session.selection
                anchor = session.target
                if (request.offer_id != selection.offer_id or selection.page is None or
                        anchor is None or request.page_state.url != anchor.url or
                        state_identity(request.page_state)[:3] != state_identity(anchor)[:3]):
                    raise DocumentError("stale_result_ref")
                chosen = [resolve_attachment(selection.page, request.result_ref, selection.refs,
                    request.allowlist, file_action=True)]
                # A previous explicit Download/Preview moved our owner to its
                # verified preview. Keep the original catalogue, but authorize
                # this new question only against the last verified owner anchor.
                # run_file_action performs the same fresh check as a card click.
                bridge.target = anchor
                bridge.research_specific = True
            else:
                await progress("searching_reports")
                await session.invalidate(bridge)
                observation = await bridge.control.interruptible(bridge.execute(ToolCall(tool="get_page_state",
                    arguments={"full": True}, snapshot_id=bridge.target.snapshot_id,
                    reason="Observe current research metadata."), 1, 1))
                current()
                if not observation.success:
                    raise DocumentError("research_search_incomplete")
                page = refined(PageState.model_validate(observation.data["page_state"]))
                bridge.target = page
                if not await publish(page) or not rank_candidates(session.selection, request.question,
                        request.allowlist, session.settings.research.max_pdfs):
                    bridge.research_capture = True
                    try:
                        from browser.tool_routing import RoutingDecision
                        from browser.tool_profiles import Route
                        from browser.routing_settings import RoutingSettings
                        await run_agent("Find current bounded PDF attachment results for this research topic: "
                            + request.question + ". Use get_page_files. Do not download or prepare files.",
                            page, request.allowlist, provider, session.settings.agent, bridge,
                            security_settings=session.settings.security, read_only=True,
                            site_adapters=SiteAdapterRegistry(session.settings.site_adapters),
                            page_settings=session.settings.page, control=bridge.control,
                            routing_settings=RoutingSettings(enabled=True),
                            initial_routing=(RoutingDecision(Route.SEARCH), {}, "orchestrator"))
                    except ResearchResultsReady:
                        pass
                    finally:
                        bridge.research_capture = False
                current()
                if session.selection.page is None:
                    raise DocumentError("research_search_incomplete")
                chosen = rank_candidates(session.selection, request.question, request.allowlist,
                    session.settings.research.max_pdfs)
            if not chosen:
                raise DocumentError("research_no_relevant_documents")
            original = session.selection.page
            # Admit only the current authoritative bounded cohort. Recheck its
            # source/owner/attachment identity on each subsequent observation;
            # neither the result cards nor admitted operations have a clock TTL.
            authorities = [DocumentOperation.admitted(identifier, "prepare", bridge.control,
                original, item) for item in chosen]
            source_url = bridge.target.url if bridge.research_specific else original.url
            bridge.research_source_url = source_url
            await send("research_plan", {"source_url": source_url, "files": [
                {"source_key": item.source_fingerprint, "result_ref": item.result_ref,
                 "filename": item.metadata["attachment_filename"], "file_type": "pdf"} for item in chosen]})
            for number, (wanted, authority) in enumerate(zip(chosen, authorities), 1):
                current()
                if number > 1:
                    call = ApprovedResearchReturn(tool="return_research_results",
                        arguments={"research_id": identifier, "url": original.url},
                        snapshot_id=bridge.target.snapshot_id, reason="Return to this task's observed bounded results.")
                    result = await bridge.control.interruptible(bridge.execute(call, number, len(chosen)))
                    current()
                    await session.audit(bridge, call, result.success, "research_return", "user_research")
                    if not result.success:
                        raise DocumentError("research_scope_changed")
                    page = refined(PageState.model_validate(result.data["page_state"]))
                    if page.url != original.url or page.tab_id != original.tab_id:
                        raise DocumentError("research_scope_changed")
                    bridge.target = page
                    if not await publish(page):
                        raise DocumentError("stale_result_ref")
                selection = session.selection
                matches = []
                for ref in selection.refs:
                    try:
                        item = resolve_attachment(selection.page, ref, selection.refs, request.allowlist, file_action=True)
                    except DocumentError:
                        continue
                    if item.source_fingerprint == wanted.source_fingerprint:
                        matches.append(item)
                if len(matches) != 1:
                    raise DocumentError("stale_result_ref")
                item = matches[0]
                # Deprecated numeric deadline kept for wire shape only. Zero
                # and state_bound explicitly mean there is no elapsed-time TTL.
                await send("research_file", {"source_key": wanted.source_fingerprint,
                    "result_ref": item.result_ref, "filename": item.metadata["attachment_filename"],
                    "file_type": "pdf", "freshness": "state_bound", "expires_at_ms": 0})

                async def file_progress(stage, current_number=number):
                    await progress(stage, current=current_number, total=len(chosen))

                await file_progress("preparing")
                file_bridge = FileBridge(bridge, file_progress)
                await run_file_action(session, {"offer_id": selection.offer_id,
                    "result_ref": item.result_ref, "action": "prepare"}, file_bridge,
                    authority=None if bridge.research_specific else authority)
                current()
                done = file_bridge.done or {}
                if done.get("status") != "ready":
                    if bridge.research_specific:
                        # One selected PDF has no alternate cohort to fall back
                        # to. Preserve its safe typed preparation failure.
                        raise DocumentError(done.get("error") or "document_operation_failed")
                    if done.get("error") in {"stale_result_ref", "operation_interrupted", "preview_identity_mismatch"}:
                        raise DocumentError(done["error"])
                    unavailable += 1
                    await progress("report_unavailable", current=number, total=len(chosen))
                    continue
                for row in done["prepared"]:
                    row = {**row, "title": item.metadata["attachment_filename"]}
                    prepared.append(row)
                    session.ready[row["document_id"]] = {"document_id": row["document_id"],
                        "index_status": "ready", "title": row["title"]}
                await send("document_ready", {"documents": list(session.ready.values())})
        if not prepared:
            raise DocumentError("document_not_ready")
        current()
        titles = {row["document_id"]: row.get("title", "Selected PDF") for row in prepared}
        session.remember_document_question(request.question, sorted(titles), bridge.target)
        await progress("searching_document", prepared_count=len(titles), unavailable=unavailable)
        answer = await bridge.control.interruptible(session.services.research_answer(request.question,
            sorted(titles), request.allow_remote_context, session.settings.research.max_visual_pages, progress))
        current()
        payload = answer.model_dump()
        for citation in payload["citations"]:
            if citation["document_id"] not in titles:
                raise DocumentError("research_scope_changed")
            citation["filename"] = titles[citation["document_id"]]
        await send("document_done", {"status": answer.status, "prepared": prepared,
            "unavailable": unavailable, "answer": payload})
    except (DocumentError, RagError, ValidationError, ControlInterrupted) as error:
        code = error.code if isinstance(error, (DocumentError, RagError)) else (
            "invalid_document_request" if isinstance(error, ValidationError) else "operation_interrupted")
        await bridge.send("document_done", {"research_id": identifier, "status": "failed", "error": code,
            "prepared": prepared, "unavailable": unavailable, "answer": None})
    except asyncio.CancelledError:
        raise
    except Exception:
        await bridge.send("document_done", {"research_id": identifier, "status": "failed",
            "error": "document_operation_failed", "prepared": prepared, "answer": None})
    finally:
        bridge.research_specific = False
        bridge.research_capture = False
        await bridge.close()
