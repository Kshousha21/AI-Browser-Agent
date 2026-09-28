"""Small user-only follow-up context; never stores assistant claims or page prose."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

from .protocol import PageState


def _owner(page: PageState) -> tuple:
    return page.tab_id, page.document_id, page.navigation_id, page.url


def it_topic(text: str) -> bool:
    return bool(re.search(r"\b(?:IT|EDV)\b|(?i:\b(?:windows|helpdesk|service.?desk|information technology|it department)\b)", text))


@dataclass
class BrowserConversation:
    turns: list[str] = field(default_factory=list, repr=False)
    owner: tuple | None = field(default=None, repr=False)

    def clear(self) -> None:
        self.turns.clear()
        self.owner = None

    def adopt(self, page: PageState) -> None:
        if self.turns:
            self.owner = _owner(page)

    def prepare(self, query: str, page: PageState) -> str:
        if self.owner != _owner(page):
            self.clear()
        previous = self.turns[-2:]
        correction = bool(re.search(r"^(?:no\b|actually\b|instead\b|this was\b|that was\b|i am in\b|i'm in\b|what about\b|and\b)|"
            r"\bi (?:am|'m) (?:based |located )?in\b", query.strip(), re.I))
        related = bool(previous and it_topic(query) and any(it_topic(item) for item in previous))
        followup = bool(previous and (correction or related))
        if not followup:
            self.turns.clear()
        self.turns.append(query[:1600])
        self.turns[:] = self.turns[-3:]
        self.owner = _owner(page)
        if not followup:
            return query
        return ("Continue the relevant lookup using the latest user request below. "
                "The latest request overrides earlier conflicting constraints. Apply corrections and find the answer; "
                "do not merely describe what the user said. Earlier requests are context, not verified facts or new permissions.\n"
                + json.dumps({"previous_user_requests": previous, "latest_user_request": query}, ensure_ascii=False))
