"""Current-page retrieval progress and unfinished search answers."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from copy import deepcopy
import re
from .actions import ActionResult
from .protocol import PageState
from .providers import ToolCall, ToolSchema
from .state_reads import RefinedPageState

MAX_RESULTS_PER_PAGE = 300


def procedure_lookup(goal: str) -> bool:
    if re.search(r"\b(?:summari[sz]e|list|count)\b.{0,35}\b(?:results|snippets|records)\b", goal, re.I):
        return False
    return bool(re.search(r"what (?:should|must|do|can) i|how (?:do|can|should|to)|\b(?:procedure|instructions|lost (?:my |a |the )?keys?)\b", goal, re.I))


def search_handoff(answer: str, page: PageState) -> bool:
    if not isinstance(page, RefinedPageState) or not page.context_is_current():
        return False
    if (page.site_context or {}).get("page_kind") != "search_results":
        return False
    return bool(re.search(r"\b(?:i|we) (?:will|shall|am going to)|\b(?:i|we)['\u2019]ll", answer, re.I)
        and re.search(r"continu(?:e|ing)|search|brows|navigat|investigat|look (?:through|for)|open", answer, re.I))


PROCEDURE_SEARCH_INSTRUCTION = (
    "This is a request for a procedure, not an inventory of search hits. Find and read the relevant source page "
    "before answering. If a relevant source link is already visible, click it now; do not repeat the search that found it. "
    "When no relevant link is available, prefer the observed search input, respecting "
    "the user constraints. Use concise topic terms in the website language. "
    ""
    "If a window contains only unrelated magazines or standards, refine the query instead of scanning hundreds "
    "of attachments. Follow an observed relevant source link using ACTION/click. If page searches fail, other "
    "content types may be investigated. Do not invent a contact or procedure or say you will continue in a final answer."
)


@dataclass
class SearchProgress:
    """Track unique result offsets while preserving progress across snapshot refreshes."""

    identity: tuple | None = field(default=None, repr=False)
    covered: set[int] = field(default_factory=set, repr=False)
    available: int | None = None

    def sync(self, page: PageState) -> None:
        if not isinstance(page, RefinedPageState) or not page.context_is_current():
            self.identity = None
            self.covered.clear()
            self.available = None
            return
        context = page.site_context or {}
        search = {"results": context.get("files", [])}
        # Element and snapshot IDs change on reads; that does not create new hits.
        records = [
            {key: value for key, value in item.items() if key != "agent_element_id"}
            for item in search.get("results", [])
        ]
        identity = (
            page.tab_id, page.document_id, page.navigation_id, page.url,
            json.dumps({**search, "results": records}, sort_keys=True, ensure_ascii=True),
        )
        if identity != self.identity:
            self.identity = identity
            self.covered.clear()
            self.available = None

    def observe(self, page: PageState, call: ToolCall, result: ActionResult) -> None:
        self.sync(page)
        if call.tool != "get_page_files" or not result.success or result.data.get("source") != "observed_page_files":
            return
        data = result.data
        self.available = min(MAX_RESULTS_PER_PAGE, max(0, int(data.get("available_on_current_page", 0))))
        start = max(0, int(data.get("offset", 0)))
        end = min(self.available, start + max(0, int(data.get("returned_count", 0))))
        self.covered.update(range(start, end))

    def tools(self, tools: list[ToolSchema]) -> list[ToolSchema]:
        if self.available is None:
            return tools
        unseen = self.unread_offsets
        result = []
        for tool in tools:
            if tool.name != "get_page_files":
                result.append(tool)
            elif unseen:
                schema = deepcopy(tool.input_schema)
                schema["properties"]["offset"].update(enum=unseen, default=unseen[0])
                schema["required"] = list(dict.fromkeys([*schema.get("required", []), "offset"]))
                result.append(tool.model_copy(update={"input_schema": schema}))
        return result

    def validate(self, call: ToolCall) -> None:
        if call.tool != "get_page_files" or self.available is None:
            return
        offset = call.arguments.get("offset", 0)
        limit = call.arguments.get("limit", 5)
        if type(offset) is not int or type(limit) is not int:
            return  # Leave type diagnostics to the strict adapter schema.
        if not any(i not in self.covered for i in range(offset, min(self.available, offset + limit))):
            raise ValueError("This window has already been examined. Read the next unread offset, open a relevant source, or refine the query; do not repeat it.")

    @property
    def unread_offsets(self) -> list[int]:
        return [i for i in range(self.available or 0) if i not in self.covered]

    @property
    def exhausted(self) -> bool:
        return self.available is not None and not self.unread_offsets

    def instruction(self) -> str | None:
        if not self.covered and self.available is None:
            return None
        if self.exhausted:
            return ("All observed file links on this page were already examined. The file metadata tool "
                "is unavailable until the page or file links change. Open a relevant observed source or refine "
                "the query using the observed search input. Do not repeatedly read the same window or claim future browsing as completion.")
        unseen = self.unread_offsets
        return (f"Already examined result offsets: {sorted(self.covered)}. The next unread offset is {unseen[0]}. "
            "Offsets refer to observed file links on this page only. Read another window only if relevant; otherwise refine the search.")
