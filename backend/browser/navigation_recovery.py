"""Bounded recovery from invented destinations using current observed controls."""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from .context_payload import element_payload
from .browser_conversation import it_topic
from .domain_policy import navigation_allowed
from .protocol import PageState
from .providers import ToolCall
from .security import ActionClassifier


def navigation_owner(page: PageState) -> tuple:
    # Re-reading an unchanged document must not re-enable failed URL guessing.
    return page.tab_id, page.document_id, page.navigation_id, page.url


_LAB_TERMS = re.compile(r"\b(?:labs?|laboratory|laboratories|labor|labore)\b", re.I)


def resource_controls(goal: str, page: PageState, allowlist: list[str]) -> list[dict]:
    words = set(re.findall(r"[^\W_]{4,}", goal.casefold())) - {
        "want", "please", "find", "show", "open", "looking", "would", "like", "give", "with", "this", "that"}
    it_lookup = it_topic(goal)
    labs = bool(_LAB_TERMS.search(goal))
    lost_key = bool(re.search(r"\blost\b.{0,25}\bkeys?\b|schl\u00fcsselverlust|schluesselverlust", goal, re.I))
    templates = bool(re.search(r"present(?:ation)?|templat|templet|vorlag|powerpoint", goal, re.I))
    classifier = ActionClassifier(goal, page, allowlist)
    ranked = []
    for item in page.elements:
        if not item.visible or not item.enabled or item.occluded:
            continue
        if item.tag not in {"a", "button"} and item.role not in {"link", "button", "menuitem", "treeitem"}:
            continue
        label = (item.name or item.text).casefold()
        score = sum(word in label for word in words)
        if it_lookup and re.search(r"\b(?:it|edv|helpdesk|service.?desk|information technology)\b", label):
            score += 4
        if lost_key and re.search(r"schl\u00fcssel|schluessel|lost.?key|fundb\u00fcro", label):
            score += 4
        if labs and _LAB_TERMS.search(label):
            score += 2
        if templates:
            if re.search(r"templat|templet|vorlag|powerpoint", label):
                score += 4
            elif "marketing" in label:
                score += 2
        if not score or (item.href and not navigation_allowed(urljoin(page.url, item.href), allowlist)):
            continue
        call = ToolCall(tool="click", arguments={"element_id":item.agent_id},
                        snapshot_id=page.snapshot_id, reason="Observed resource control")
        if classifier.classify(call, page).risk == "high":
            continue
        ranked.append((score, item))
    ranked.sort(key=lambda row: -row[0])
    return [element_payload(item) for _, item in ranked[:6]]


def resource_pointer_handoff(goal: str, answer: str, controls: list[dict]) -> bool:
    """Catch a section pointer that leaves a requested resource lookup unfinished.

    This is bounded recovery guidance, not a general factual-answer validator.
    Explicit requests for UI directions or just a link remain valid final answers.
    """
    if not controls or not re.search(r"\b(?:where|find|locate|search|looking|want|wo|suche)\b", goal, re.I):
        return False
    if re.search(r"\bhow\b.{0,60}\b(?:open|click|navigate)\b|"
                 r"\bwhere\b.{0,60}\b(?:link|button)\b|"
                 r"\b(?:which|what)\s+link\b|\b(?:just|only)\b.{0,20}\blink\b", goal, re.I):
        return False
    labels = {word for control in controls
              for word in re.findall(r"[^\W_]{3,}", str(control.get("name") or control.get("text") or "").casefold())}
    if any(_LAB_TERMS.fullmatch(word) for word in labels):
        labels.update({"lab", "labor", "laboratory"})
    if not labels.intersection(re.findall(r"[^\W_]{3,}", answer.casefold())):
        return False
    return bool(re.search(
        r"\b(?:located|found|available|accessed)\s+(?:at|in|via|through|under)\s+(?:the\s+)?(?:link|sidebar|navigation|section)\b|"
        r"\b(?:you\s+(?:can|should|need to)|please|to find)\b.{0,80}\b(?:click|follow|open|visit|check|select)\b|"
        r"^\s*(?:click|follow|open|visit|check|select)\b|"
        r"\b(?:under|via|through)\b.{0,60}\b(?:sidebar|navigation menu)\b",
        answer, re.I))
