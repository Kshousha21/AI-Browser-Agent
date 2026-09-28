"""Provider-neutral action risk classification and run-scoped permissions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qsl, unquote, urlparse

from .actions import ActionResult
from .config import SecuritySettings
from .domain_policy import navigation_allowed
from .protocol import PageElement, PageState
from .providers import ToolCall
from .tool_execution import BuiltNavigationCall, BuiltStateReadCall

Risk = Literal["low", "medium", "high"]
PermissionState = Literal["allow", "block_read_only", "confirm"]
ConfirmationChoice = Literal["allow_once", "allow_run", "cancel", "policy_allow_once"]

LOW_RISK_TOOLS = {
    "go_back", "go_forward", "refresh",
    "get_tabs",
    "switch_tab",
    "get_page_state",
    "read_page",
    "get_element_details",
    "scroll",
    "find_text",
    "wait_for_stable",
}
MEDIUM_RISK_TOOLS = {"click", "type_text", "clear_input", "press_key", "open_tab"}

INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "security:prompt_injection_pattern",
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget)\b.{0,50}\b"
            r"(?:previous|prior|system|developer|instructions?|prompt)\b"
        ),
    ),
    (
        "security:instruction_impersonation",
        re.compile(r"(?i)\b(?:system|developer)\s*(?:message|instruction|prompt)\s*:"),
    ),
    (
        "security:suspicious_navigation_instruction",
        re.compile(r"(?i)\b(?:navigate|browse|redirect|go)\s+to\s+https?://"),
    ),
    (
        "security:credential_exfiltration_instruction",
        re.compile(
            r"(?i)\b(?:send|post|upload|reveal|exfiltrate)\b.{0,80}\b"
            r"(?:api[_ -]?key|token|cookie|authorization|password|secret)\b"
        ),
    ),
    (
        "security:prompt_delimiter_text",
        re.compile(r"(?i)</?(?:untrusted_)?(?:webpage|page)[_ -]?(?:content|snapshot)>"),
    ),
)

_SEMANTIC_RISKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("delete", re.compile(r"(?i)\b(?:delete|remove|erase|revoke|wipe|destroy|purge|l\u00f6schen|loeschen|entfernen|widerrufen)\b")),
    ("send", re.compile(r"(?i)\b(?:send|publish|post|share)\b")),
    ("upload", re.compile(r"(?i)\b(?:upload|attach)\b")),
    ("download", re.compile(r"(?i)\b(?:download|export)\b")),
    ("submit", re.compile(r"(?i)\b(?:submit|confirm|approve|purchase|book|save)\b")),
)


def _deletion_url(url: str) -> bool:
    """Recognize deletion endpoints, including explicit removal links."""
    parsed = urlparse(unquote(url))
    verbs = r"(?:delete|remove|erase|revoke|trash|purge|l\u00f6schen|loeschen|entfernen)"
    endpoint = rf"{verbs}(?:page|attachment|comment|file|content|item|space|user)?(?:\.action)?"
    if any(re.fullmatch(endpoint, part, re.IGNORECASE) for part in parsed.path.split("/")):
        return True
    return any(key.casefold() in {"action", "operation", "do"}
               and re.fullmatch(endpoint, value, re.IGNORECASE)
               for key, value in parse_qsl(parsed.query))


def scan_injection(text: str) -> list[str]:
    """Return stable warning codes without copying suspicious text."""
    return [code for code, pattern in INJECTION_PATTERNS if pattern.search(text)]


def origin(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


@dataclass(frozen=True)
class RiskAssessment:
    risk: Risk
    reasons: tuple[str, ...]
    scope: str
    target: str


@dataclass(frozen=True)
class ActionIdentity:
    """Security-relevant identity captured when an action is approved."""

    tool: str
    tab_id: int | None = None
    document_id: str = ""
    agent_id: str = ""
    tag: str = ""
    name: str = ""
    role: str = ""
    href: str = ""
    form_action: str = ""
    form_method: str = ""
    form_target: str = ""
    destination: str = ""


class ActionClassifier:
    """Classify actions using tool semantics and the small taint heuristic."""

    def __init__(self, goal: str, page: PageState, allowlist: list[str]) -> None:
        self.user_text = goal
        self.allowlist = allowlist
        self.observed_by_origin: dict[str, str] = {}
        self.observe(page)

    def observe(self, page: PageState) -> None:
        current = origin(page.url)
        if current:
            existing = self.observed_by_origin.get(current, "")
            self.observed_by_origin[current] = (
                existing + "\n" + page.visible_text
            )[-100_000:]

    def classify(self, call: ToolCall, page: PageState) -> RiskAssessment:
        element = _target_element(call, page)
        target = element.name if element and element.name else str(
            call.arguments.get("element_id", call.arguments.get("url", ""))
        )
        reasons: list[str] = []

        if call.tool == "close_tab":
            reasons.append("close_tab")

        if call.tool in {"navigate", "open_tab"}:
            destination = str(call.arguments.get("url", ""))
            if not navigation_allowed(destination, self.allowlist):
                reasons.append("navigate_external")
            if _is_file_download(destination):
                reasons.append("download")
            if (
                not isinstance(call, BuiltNavigationCall)
                and not _appears_in_user_text(destination, self.user_text)
                and not _observed_page_navigation(destination, page, self.allowlist)
            ):
                reasons.append("untrusted_argument")

        if call.tool == "type_text" and call.arguments.get("submit") is True:
            reasons.append("submit")
            value = str(call.arguments.get("text", ""))
            if value and not _appears_in_user_text(value, self.user_text):
                reasons.append("untrusted_argument")

        if call.tool == "click" and element is not None:
            if element.input_type in {"submit", "image"}:
                reasons.append("submit")
            if _is_file_download(element.href):
                reasons.append("download")
            semantic_text = f"{element.role} {element.name}"
            for code, pattern in _SEMANTIC_RISKS:
                if pattern.search(semantic_text):
                    reasons.append(code)

        for key in ("recipient", "to", "email"):
            value = call.arguments.get(key)
            if isinstance(value, str) and value and not _appears_in_user_text(
                value, self.user_text
            ):
                reasons.append("untrusted_argument")

        if call.tool == "type_text":
            value = str(call.arguments.get("text", ""))
            current_origin = origin(page.url)
            if value and current_origin:
                for source_origin, observed in self.observed_by_origin.items():
                    if source_origin != current_origin and value in observed:
                        reasons.extend(("cross_origin_transfer", "untrusted_argument"))
                        break

        # Activation can happen through a link, a button, or the keyboard.
        activates = call.tool == "click" or (
            call.tool == "press_key" and call.arguments.get("key") in {"Enter", "Space", " "}
        ) or (call.tool == "type_text" and call.arguments.get("submit") is True)
        if activates and element is not None:
            if (_SEMANTIC_RISKS[0][1].search(element.name)
                    or _deletion_url(element.href) or _deletion_url(element.form_action)):
                reasons.append("delete")
        if call.tool in {"navigate", "open_tab"}:
            destination = str(call.arguments.get("url", ""))
            if _deletion_url(destination) or any(
                item.href == destination and _SEMANTIC_RISKS[0][1].search(item.name)
                for item in page.elements
            ):
                reasons.append("delete")

        reasons = list(dict.fromkeys(reasons))
        if reasons:
            risk: Risk = "high"
        elif isinstance(call, BuiltStateReadCall) or call.tool in LOW_RISK_TOOLS or (
            call.tool == "navigate"
            and navigation_allowed(str(call.arguments.get("url", "")), self.allowlist)
        ):
            risk = "low"
        elif call.tool in MEDIUM_RISK_TOOLS:
            risk = "medium"
        else:
            risk = "high"
            reasons = ["unknown_high_risk_action"]
        return RiskAssessment(risk, tuple(reasons), _scope(call), target)


class PermissionManager:
    """Apply read-only mode and confirmation grants for one run."""

    def __init__(self, settings: SecuritySettings, read_only: bool) -> None:
        self.read_only = False  # Legacy argument ignored; there is no read-only mode.
        self.confirmation_policy = settings.confirmation_policy
        self.required = set(settings.require_confirmation_for)
        self._run_grants: dict[str, ActionIdentity | None] = {}

    def check(self, assessment: RiskAssessment) -> PermissionState:
        return "confirm" if "delete" in assessment.reasons else "allow"

    def has_run_grant(self, assessment: RiskAssessment) -> bool:
        return assessment.scope in self._run_grants

    def grant_identity(self, assessment: RiskAssessment) -> ActionIdentity | None:
        return self._run_grants.get(assessment.scope)

    def invalidate_all(self) -> None:
        self._run_grants.clear()

    def invalidate(self, assessment: RiskAssessment) -> None:
        self._run_grants.pop(assessment.scope, None)

    def apply(
        self,
        assessment: RiskAssessment,
        choice: ConfirmationChoice,
        identity: ActionIdentity | None = None,
    ) -> bool:
        if "delete" in assessment.reasons:
            return choice in {"allow_once", "allow_run"}
        if choice == "allow_run":
            self._run_grants[assessment.scope] = identity
            return True
        return choice in {"allow_once", "policy_allow_once"}


def permission_denied(
    call: ToolCall,
    page: PageState,
    message: str,
) -> ActionResult:
    return ActionResult(
        success=False,
        action=call.tool,
        element_id=(
            str(call.arguments["element_id"])
            if isinstance(call.arguments.get("element_id"), str)
            else None
        ),
        snapshot_id=page.snapshot_id,
        error="permission_denied",
        message=message,
    )


def target_element(call: ToolCall, page: PageState) -> PageElement | None:
    return _target_element(call, page)


def action_identity(call: ToolCall, page: PageState) -> ActionIdentity | None:
    """Fingerprint the approved target without trusting a later DOM lookup."""
    if call.tool in {"navigate", "open_tab"}:
        return ActionIdentity(
            tool=call.tool, tab_id=page.tab_id, document_id=page.document_id,
            destination=str(call.arguments.get("url", "")).strip(),
        )
    element_id = call.arguments.get("element_id")
    if not isinstance(element_id, str):
        return None
    element = _target_element(call, page)
    if element is None:
        return None
    return ActionIdentity(
        tool=call.tool, tab_id=page.tab_id, document_id=page.document_id,
        agent_id=element.agent_id,
        tag=element.tag,
        name=element.name,
        role=element.role,
        href=element.href,
        form_action=element.form_action,
        form_method=element.form_method,
        form_target=element.form_target,
    )


def _target_element(call: ToolCall, page: PageState) -> PageElement | None:
    element_id = call.arguments.get("element_id")
    if not isinstance(element_id, str):
        return None
    return next((item for item in page.elements if item.agent_id == element_id), None)


def _scope(call: ToolCall) -> str:
    element_id = call.arguments.get("element_id")
    if isinstance(element_id, str):
        return f"{call.tool}:element:{element_id}"
    url = call.arguments.get("url")
    if isinstance(url, str):
        return f"{call.tool}:url:{url.strip()}"
    return call.tool


def _appears_in_user_text(value: str, user_text: str) -> bool:
    return bool(value.strip()) and value.casefold() in user_text.casefold()


def _is_file_download(value: str) -> bool:
    """Recognize the observed file-transfer route from the exact raw href."""
    if not value:
        return False
    try:
        path = unquote(urlparse(value).path).casefold()
    except (TypeError, ValueError):
        return False
    return bool(re.search(r"\.(?:pdf|docx?|xlsx?|zip)$", path))


def _observed_page_navigation(destination: str, page: PageState, allowlist: list[str]) -> bool:
    """Ordinary visible links within the current origin need no repeated approval."""
    from urllib.parse import urljoin
    if (not navigation_allowed(destination, allowlist)
            or _is_file_download(destination)):
        return False
    return any(item.visible and item.enabled and item.tag == "a" and item.href
               and not any(pattern.search(f"{item.role} {item.name}") for _, pattern in _SEMANTIC_RISKS)
               and urljoin(page.url, item.href) == destination for item in page.elements)
