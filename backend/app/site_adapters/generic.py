"""Default real-DOM adapter for every normal HTTP(S) website."""

import re
from urllib.parse import urljoin, urlsplit, unquote
from browser.domain_policy import navigation_allowed
from browser.state_reads import RefinedPageState
from .base import BaseSiteAdapter
from .page_files import PageFilesArgs, TOOL_NAME, retrieve, retrieval_tool


class GenericSiteAdapter(BaseSiteAdapter):
    site_name = "generic"

    def matches(self, url):
        return navigation_allowed(url)

    def refine_page_state(self, state):
        files, seen = [], set()
        for element in state.elements:
            if not element.visible or not element.enabled or element.tag != "a" or not element.href:
                continue
            url = urljoin(state.url, element.href)
            if not navigation_allowed(url) or url in seen:
                continue
            try:
                filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1], errors="strict")
            except (ValueError, UnicodeError):
                continue
            if not re.fullmatch(r'[^\\/:*?"<>|\x00-\x1f]+\.(?:pdf|docx?|xlsx?|pptx?|txt|csv|zip)', filename, re.I):
                continue
            seen.add(url)
            files.append({"rank": len(files) + 1, "title": element.name or filename,
                "url": url, "content_type": "attachment", "file_type": filename.rsplit('.', 1)[1].lower(),
                "attachment_filename": filename, "agent_element_id": element.agent_id})
        # Never accept page-supplied adapter metadata as backend provenance.
        clean = state.model_copy(update={"site_context": {"adapter": "GenericSiteAdapter",
            "site": "generic", "page_kind": "page", "files": files}, "adapter_payload": None})
        return RefinedPageState.bind(clean)

    def extra_tools(self):
        return [retrieval_tool()]

    def validate_state_read(self, name, arguments):
        return PageFilesArgs.model_validate(arguments).model_dump() if name == TOOL_NAME else None

    def read_state(self, call, state):
        return retrieve(call, state) if call.tool == TOOL_NAME else None

    def tools_for_route(self, route, page_kind="unknown"):
        return [TOOL_NAME] if route in {"READ", "SEARCH"} else []

    def system_prompt_addendum(self):
        return ("Use the current DOM: search by typing into an observed search input and submitting it. "
                "Open relevant observed results, then answer from the source. Never invent site-specific search URLs. "
                "Use get_page_files when file cards are needed. Ordinary browser actions need no confirmation; "
                "clear deletion or destructive actions require one confirmation.")
