"""Public seam for host-scoped PageState normalization."""

from __future__ import annotations

from abc import ABC
from typing import Any
from urllib.parse import urlparse

from browser.protocol import PageState
from browser.providers import ToolSchema, ToolCall
from browser.actions import ActionResult


class BaseSiteAdapter(ABC):
    """Refine generic state for an explicitly configured set of hosts."""

    site_name = "generic"

    def __init__(self, domains: list[str] | None = None) -> None:
        self.domain_patterns = tuple(
            domain.casefold().rstrip(".") for domain in (domains or [])
        )

    def matches(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").casefold().rstrip(".")
        return bool(host and host in self.domain_patterns)

    def refine_page_state(self, state: PageState) -> PageState:
        return state

    def extra_tools(self) -> list[ToolSchema | dict[str, Any]]:
        return []

    def validate_state_read(self, name: str, arguments: dict[str, object]) -> dict | None:
        """Validate an owned backend-only read, or decline ownership."""
        return None

    def read_state(self, call: ToolCall, state: PageState) -> ActionResult | None:
        """Read already-normalized current state without browser or network I/O."""
        return None

    def translate_tool_call(
        self, name: str, arguments: dict[str, object]
    ) -> tuple[str, dict[str, object]] | None:
        """Translate one adapter tool to an existing browser tool, if owned."""
        del name, arguments
        return None

    def tool_call_satisfied(
        self,
        name: str,
        arguments: dict[str, object],
        state: PageState,
    ) -> bool:
        """Return whether current normalized state already reflects the call."""
        del name, arguments, state
        return False

    def system_prompt_addendum(self) -> str:
        return ""

    def tools_for_route(self, route: str, page_kind: str = "unknown") -> list[str]:
        """Names of existing owned semantic tools appropriate to this capability."""
        return []
