"""Capability names reference existing tool definitions, never implementations."""
from enum import StrEnum
from types import MappingProxyType

from .providers import ToolSchema


class Route(StrEnum):
    READ = "READ"
    SEARCH = "SEARCH"
    ACTION = "ACTION"
    NAVIGATION = "NAVIGATION"
    DOCUMENT = "DOCUMENT"
    IMAGE = "IMAGE"
    GENERAL = "GENERAL"
    UNSUPPORTED = "UNSUPPORTED"


PROFILES = MappingProxyType({
    Route.READ: frozenset({"get_page_state", "read_page", "get_element_details", "find_text", "scroll"}),
    Route.SEARCH: frozenset({"get_page_state", "read_page", "get_element_details", "click", "type_text", "press_key", "wait_for_stable"}),
    Route.ACTION: frozenset({"get_page_state", "get_element_details", "click", "type_text", "clear_input", "press_key", "scroll", "wait_for_stable"}),
    Route.NAVIGATION: frozenset({"get_page_state", "click", "navigate", "go_back", "go_forward", "refresh", "get_tabs", "switch_tab", "open_tab", "close_tab", "wait_for_stable"}),
    # These operations remain behind explicit document messages/consent, or are
    # not implemented. A route is never a new source of acquisition authority.
    Route.DOCUMENT: frozenset(),
    Route.IMAGE: frozenset(),
    Route.GENERAL: frozenset(),
    Route.UNSUPPORTED: frozenset(),
})

ALLOWED_TRANSITIONS = MappingProxyType({
    Route.SEARCH: frozenset({Route.SEARCH, Route.ACTION, Route.READ, Route.NAVIGATION, Route.DOCUMENT}),
    Route.READ: frozenset({Route.READ, Route.SEARCH, Route.ACTION, Route.NAVIGATION, Route.DOCUMENT}),
    Route.ACTION: frozenset({Route.ACTION, Route.READ, Route.SEARCH, Route.NAVIGATION, Route.DOCUMENT}),
    Route.NAVIGATION: frozenset({Route.NAVIGATION, Route.READ, Route.SEARCH, Route.ACTION}),
    Route.DOCUMENT: frozenset({Route.DOCUMENT, Route.READ, Route.IMAGE}),
    Route.IMAGE: frozenset({Route.IMAGE, Route.READ}),
    Route.GENERAL: frozenset({Route.GENERAL, Route.READ, Route.SEARCH}),
    Route.UNSUPPORTED: frozenset(),
})


def profile_tools(route: Route, available: list[ToolSchema], adapter_names=()) -> list[ToolSchema]:
    """Intersect with the current site's registry; never manufacture a schema."""
    registered = {tool.name for tool in available}
    additions = set(adapter_names) & registered
    names = set(PROFILES[route]) | additions
    return [tool for tool in available if tool.name in names]
