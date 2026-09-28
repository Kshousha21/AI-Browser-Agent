"""Bounded, provenance-bound metadata for observed file links; no network I/O."""
import hashlib
import json
from pydantic import BaseModel, ConfigDict, Field
from browser.actions import ActionResult
from browser.security import scan_injection
from browser.state_reads import RefinedPageState, state_identity, context_digest, unavailable

TOOL_NAME = "get_page_files"

class PageFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    limit: int = Field(default=5, ge=1, le=10)
    offset: int = Field(default=0, ge=0, le=299)

def retrieval_tool():
    return {"name": TOOL_NAME, "description": "List metadata for observed file links on this page. Up to 10 per window; use offset for more. Does not open or download files.",
            "input_schema": PageFilesArgs.model_json_schema()}

def retrieve(call, state):
    args = PageFilesArgs.model_validate(call.arguments)
    if (not isinstance(state, RefinedPageState) or not state.context_is_current()
            or call.snapshot_id != state.snapshot_id):
        return unavailable(call, state, stale=True)
    context = state.site_context or {}
    if state.looks_like_login or context.get("adapter") != "GenericSiteAdapter":
        return unavailable(call, state)
    records = context.get("files", [])
    if args.offset and args.offset >= len(records):
        return ActionResult(success=False, action=call.tool, snapshot_id=state.snapshot_id,
                            error="invalid_arguments", message="Offset is outside the current file links.")
    identity = json.dumps([state_identity(state), context_digest(state)], separators=(",", ":"))
    results = []
    for index, record in enumerate(records[args.offset:args.offset + args.limit], args.offset):
        ref = hashlib.sha256(f"{identity}\0{index}".encode()).hexdigest()[:24]
        fields = ("rank", "title", "content_type", "file_type", "attachment_filename")
        results.append({"result_ref": "result-" + ref, **{k: record.get(k) for k in fields}})
    return ActionResult(success=True, action=call.tool, snapshot_id=state.snapshot_id,
        message="Observed file metadata only; no file content read.", data={
            "source": "observed_page_files", "trust": "untrusted_webpage_data",
            "offset": args.offset, "limit": args.limit, "returned_count": len(results),
            "available_on_current_page": len(records), "total_results": len(records),
            "results": results, "warnings": sorted(set(state.warnings) | set(scan_injection(json.dumps(results))))})
