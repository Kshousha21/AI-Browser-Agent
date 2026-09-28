"""Small shared runtime helpers for the bounded agent loop."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.site_adapters import SiteAdapterRegistry

from .actions import tool_schemas, validate_tool_call
from .agent_support import TokenBudget
from .config import PageSettings
from .prompting import clamp_page_state
from .protocol import PageState
from .providers import ToolCall, ToolSchema


@dataclass(frozen=True)
class AgentOutcome:
    text: str
    finish_reason: str
    usage: dict[str, int]
    elapsed_ms: int


def elapsed(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def refine_page(
    page: PageState,
    adapters: SiteAdapterRegistry | None,
    limits: PageSettings | None,
) -> PageState:
    if limits is not None:
        page = clamp_page_state(page, limits)
    return adapters.refine_page_state(page) if adapters is not None else page


def available_tools(
    page_url: str, adapters: SiteAdapterRegistry | None, *, tab_bound: bool = False
) -> list[ToolSchema]:
    """Combine native tools with only the active adapter's strict tools."""
    schemas = [tool for tool in tool_schemas() if not tab_bound or tool.name not in {"open_tab", "switch_tab"}]
    if adapters is not None:
        schemas.extend(adapters.tool_schemas(page_url))
    return schemas


def prepare_tool_call(
    page_url: str,
    requested: ToolCall,
    adapters: SiteAdapterRegistry | None,
) -> tuple[ToolCall, bool]:
    """Validate a native call or lower an adapter-owned virtual call."""
    if adapters is not None:
        state_read = adapters.prepare_state_read(page_url, requested)
        if state_read is not None:
            return state_read, True
    translated = (
        adapters.translate_tool_call(page_url, requested)
        if adapters is not None
        else None
    )
    if translated is not None:
        return translated, True
    validate_tool_call(requested)
    return requested, False


def outcome(
    text: str, reason: str, budget: TokenBudget, started: float
) -> AgentOutcome:
    return AgentOutcome(text, reason, dict(budget.usage), elapsed(started))


def login_outcome(budget: TokenBudget, started: float) -> AgentOutcome:
    return outcome(
        "This page looks like a sign-in screen. Please sign in, then ask again.",
        "login_required",
        budget,
        started,
    )


def token_budget_outcome(
    budget: TokenBudget, started: float
) -> AgentOutcome:
    return outcome(
        "Token budget for this run was reached.",
        "token_budget",
        budget,
        started,
    )
