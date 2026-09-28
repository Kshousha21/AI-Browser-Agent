"""Bounded source checks for contact recommendations, separate from model instructions."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from urllib.parse import urlparse

from .protocol import PageState

_EMAIL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def contact_lookup_requested(goal: str) -> bool:
    if re.search(r"(?:summari[sz]e|only list|just list).{0,35}(?:search results|snippets)", goal, re.I):
        return False
    return bool(re.search(r"\b(?:contacts?|e-?mails?|phone|helpdesk|ansprechpersonen?|kontakt)\b", goal, re.I))


def requested_location(goal: str) -> str | None:
    matches = list(re.finditer(r"(?i:\b(?:i am|i'm|ich bin) (?:based |located )?in)\s+([^\W\d_][\w-]*(?:[ \t]+[A-Z][\w-]*){0,2})", goal))
    return matches[-1].group(1).casefold() if matches else None


def search_page(page: PageState) -> bool:
    return ((page.site_context or {}).get("page_kind") == "search_results"
            or urlparse(page.url).path.endswith("/dosearchsite.action"))


@dataclass
class ContactEvidence:
    pages: list[tuple[str, dict[str, list[str]]]] = field(default_factory=list, repr=False)

    def observe(self, page: PageState) -> None:
        if search_page(page):
            return
        text = page.visible_text[:20000]
        for element in page.elements:
            if element.visible and element.href.startswith("mailto:"):
                text += "\n" + element.name + " " + element.href[7:].split("?", 1)[0]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        contacts = {}
        for i, line in enumerate(lines):
            context = line
            if i and not _EMAIL.search(lines[i-1]):
                context = lines[i-1] + "\n" + context
            for email in _EMAIL.findall(line):
                contacts.setdefault(email.casefold(), []).append(context[:1200])
        self.pages = [(url, items) for url, items in self.pages if url != page.url]
        if contacts:
            self.pages.append((page.url, contacts))
        self.pages = self.pages[-6:]

    def issue(self, goal: str, answer: str, page: PageState) -> str | None:
        if not contact_lookup_requested(goal):
            return None
        emails = _EMAIL.findall(answer.replace("\\@", "@"))
        if search_page(page) and not emails:
            return "search_candidates_only"
        location = requested_location(goal)
        for email in emails:
            contexts = [context for _, items in self.pages for context in items.get(email.casefold(), [])]
            if not contexts:
                return "search_candidates_only" if search_page(page) else "contact_not_observed"
            if location and not any(re.search(r"(?<!\w)" + re.escape(location) + r"(?!\w)", context, re.I)
                or re.search(r"all (?:sites|locations)|alle Standorte|institutsweit", context, re.I) for context in contexts):
                return "location_not_verified"
        if not emails and re.search(r"you (?:may|might|can|should|need).{0,60}(?:search|navigate|look)|the user (?:mentioned|said)|not relevant", answer, re.I):
            return "lookup_unfinished"
        return None
