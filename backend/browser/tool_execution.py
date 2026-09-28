"""Internal provenance for browser calls produced by trusted code."""

from __future__ import annotations

from typing import Literal

from .providers import ToolCall


class BuiltNavigationCall(ToolCall):
    """A navigate call whose URL was built by validated, in-process code.

    This type is never parsed from provider output. Its marker only suppresses
    the model/page-derived argument check; normal allowlist enforcement still
    applies in both the backend and extension.
    """

    internally_built_navigation: Literal[True] = True


class BuiltStateReadCall(ToolCall):
    """Validated adapter read constructed in-process, never from provider JSON.

    Executes only against current backend state, never through a browser bridge.
    """
