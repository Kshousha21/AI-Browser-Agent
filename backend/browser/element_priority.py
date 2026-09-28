"""Semantic ordering and conservative deduplication for PageState elements."""

from __future__ import annotations

import re

from .protocol import PageElement

_FORM_TAGS = {"input", "textarea", "select"}
_BUTTON_ROLES = {"button"}
_LINK_ROLES = {"link"}
_ARIA_CONTROLS = {
    "checkbox",
    "combobox",
    "listbox",
    "menuitem",
    "option",
    "radio",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "textbox",
    "treeitem",
}
_STRUCTURAL_TAGS = {"form", "table"}
_LOW_VALUE_TAGS = {"path"}


def prioritize_elements(
    elements: list[PageElement], limit: int
) -> tuple[list[PageElement], bool]:
    """Return high-value controls first and omit only safe semantic duplicates."""
    candidates = [item for item in elements if not _drop_decorative(item)]
    candidates.sort(key=lambda item: (_priority(item), item.document_order))
    deduplicated: list[PageElement] = []
    exact_seen: set[tuple[str, ...]] = set()
    visible_destinations: set[tuple[str, ...]] = set()
    for item in candidates:
        exact = _semantic_key(item, include_region=True)
        if exact in exact_seen:
            continue
        destination = _destination_key(item)
        if not item.visible and destination and destination in visible_destinations:
            continue
        exact_seen.add(exact)
        if item.visible and destination:
            visible_destinations.add(destination)
        deduplicated.append(item)
    overflow = len(deduplicated) > limit
    return deduplicated[:limit], overflow


def generic_element_count(elements: list[PageElement]) -> int:
    return sum(
        1
        for item in elements
        if item.role in {"", "generic", "none", "presentation"}
    )


def _priority(item: PageElement) -> int:
    named = bool(item.name.strip())
    visible = item.visible
    if visible and named and item.tag in _FORM_TAGS:
        return 0
    if visible and named and item.role in _BUTTON_ROLES:
        return 1
    if visible and named and item.role in _LINK_ROLES and item.href:
        return 2
    if visible and named and item.role in _ARIA_CONTROLS:
        return 3
    if visible and named and _looks_like_result_control(item):
        return 4
    if visible and item.tag in _STRUCTURAL_TAGS:
        return 5
    if visible and _actionable(item):
        return 6
    if not visible and _actionable(item):
        return 7
    if item.role in {"", "generic", "none", "presentation"}:
        return 8
    return 9


def _drop_decorative(item: PageElement) -> bool:
    tag = item.tag.casefold()
    role = item.role.casefold()
    if tag in _LOW_VALUE_TAGS:
        return True
    if tag == "svg" and role in {"", "generic", "img"} and not item.name:
        return True
    return bool(
        role in {"", "generic", "none", "presentation"}
        and not item.name
        and not item.href
        and not item.form_action
        and tag not in _STRUCTURAL_TAGS
        and tag not in _FORM_TAGS
    )


def _actionable(item: PageElement) -> bool:
    return bool(
        item.tag in _FORM_TAGS
        or item.role in _BUTTON_ROLES | _LINK_ROLES | _ARIA_CONTROLS
        or item.href
        or item.form_action
    )


def _looks_like_result_control(item: PageElement) -> bool:
    marker = " ".join(
        (item.test_id, item.semantic_region, item.data_key)
    ).casefold()
    return bool(re.search(r"result|card|item|job", marker))


def _semantic_key(
    item: PageElement, *, include_region: bool
) -> tuple[str, ...]:
    values = [
        item.role.casefold(),
        _normalized(item.name),
        item.tag.casefold(),
        item.href,
        item.form_action,
        item.form_method.casefold(),
        item.form_target,
        item.data_key,
    ]
    if include_region:
        values.append(item.semantic_region)
    return tuple(values)


def _destination_key(item: PageElement) -> tuple[str, ...] | None:
    destination = item.href or item.form_action
    if not destination or not item.name:
        return None
    return (
        item.role.casefold(),
        _normalized(item.name),
        destination,
        item.form_method.casefold(),
        item.form_target,
    )


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())
