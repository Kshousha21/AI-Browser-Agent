"""Semantic element, block-text, and adapter deltas for PageState."""

from __future__ import annotations

import hashlib
import json
from difflib import SequenceMatcher
from typing import Any

from .context_payload import (
    context_elements,
    context_text,
    display_element_value,
    display_url,
    element_payload,
    model_site_context,
    normalized,
    text_regions,
)
from .protocol import PageElement, PageHeading, PageState


def diff_page_payload(previous: PageState, current: PageState) -> dict[str, Any]:
    """Return a semantic element/list/text delta against one known base."""
    old_elements = context_elements(previous)
    new_elements = context_elements(current)
    old_text, _ = context_text(previous)
    new_text, _ = context_text(current)
    return {
        "type": "page_state_diff",
        "base_snapshot_id": previous.snapshot_id,
        "snapshot_id": current.snapshot_id,
        "document_id": current.document_id,
        "navigation_id": current.navigation_id,
        "url_changed": previous.url != current.url,
        "url": current.url if previous.url != current.url else None,
        "title_changed": previous.title != current.title,
        "title": current.title if previous.title != current.title else None,
        "elements": _element_diff(old_elements, new_elements),
        "text": _text_diff(old_text, new_text),
        "headings": _heading_diff(previous.headings, current.headings),
        "warnings": {
            "added": sorted(set(current.warnings) - set(previous.warnings)),
            "removed": sorted(set(previous.warnings) - set(current.warnings)),
        },
        "site_context": _site_context_diff(
            model_site_context(previous.site_context),
            model_site_context(current.site_context),
        ),
        "metadata": _metadata_diff(previous, current),
    }


def _element_diff(
    previous: list[PageElement], current: list[PageElement]
) -> dict[str, list[dict[str, Any]]]:
    old_by_id = {item.agent_id: item for item in previous}
    new_by_id = {item.agent_id: item for item in current}
    matched: list[tuple[PageElement, PageElement]] = []
    for agent_id in old_by_id.keys() & new_by_id.keys():
        matched.append((old_by_id.pop(agent_id), new_by_id.pop(agent_id)))

    old_semantic = _unique_semantic(old_by_id.values())
    new_semantic = _unique_semantic(new_by_id.values())
    for identity in old_semantic.keys() & new_semantic.keys():
        old = old_semantic[identity]
        new = new_semantic[identity]
        matched.append((old, new))
        old_by_id.pop(old.agent_id, None)
        new_by_id.pop(new.agent_id, None)

    changed: list[dict[str, Any]] = []
    for old, new in matched:
        fields = _changed_element_fields(old, new)
        if fields:
            changed.append(
                {
                    "agent_id": new.agent_id,
                    "previous_agent_id": (
                        old.agent_id if old.agent_id != new.agent_id else None
                    ),
                    "changes": fields,
                }
            )
    return {
        "added": [element_payload(item) for item in new_by_id.values()],
        "removed": [
            {
                "agent_id": item.agent_id,
                "role": item.role,
                "name": item.name,
                "tag": item.tag,
                "href": display_url(item.href) if item.href else "",
            }
            for item in old_by_id.values()
        ],
        "changed": changed,
    }


def _unique_semantic(elements: Any) -> dict[tuple[str, ...], PageElement]:
    groups: dict[tuple[str, ...], list[PageElement]] = {}
    for item in elements:
        groups.setdefault(_element_identity(item), []).append(item)
    return {
        key: values[0]
        for key, values in groups.items()
        if len(values) == 1
    }


def _element_identity(item: PageElement) -> tuple[str, ...]:
    return (
        item.role.casefold(),
        normalized(item.name),
        item.tag.casefold(),
        item.href,
        item.form_action,
        item.form_method.casefold(),
        item.form_target,
        item.data_key,
        item.semantic_region,
    )


def _changed_element_fields(
    previous: PageElement, current: PageElement
) -> dict[str, dict[str, Any]]:
    fields = (
        "agent_id",
        "role",
        "name",
        "tag",
        "text",
        "placeholder",
        "value",
        "input_type",
        "href",
        "form_action",
        "form_method",
        "form_target",
        "data_key",
        "selected",
        "enabled",
        "visible",
        "occluded",
        "in_viewport",
    )
    return {
        field: {
            "before": display_element_value(field, getattr(previous, field)),
            "after": display_element_value(field, getattr(current, field)),
        }
        for field in fields
        if getattr(previous, field) != getattr(current, field)
    }


def _text_diff(previous: str, current: str) -> dict[str, list[dict[str, Any]]]:
    old = text_regions(previous)
    new = text_regions(current)
    matcher = SequenceMatcher(a=old, b=new, autojunk=False)
    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation == "equal":
            continue
        if operation == "insert":
            added.extend(_indexed_regions(new, j1, j2))
        elif operation == "delete":
            removed.extend(_indexed_regions(old, i1, i2))
        else:
            paired = min(i2 - i1, j2 - j1)
            for offset in range(paired):
                changed.append(
                    {
                        "previous_index": i1 + offset,
                        "index": j1 + offset,
                        "before": old[i1 + offset],
                        "after": new[j1 + offset],
                    }
                )
            removed.extend(_indexed_regions(old, i1 + paired, i2))
            added.extend(_indexed_regions(new, j1 + paired, j2))
    return {
        "added_regions": added,
        "removed_regions": removed,
        "changed_regions": changed,
    }


def _indexed_regions(
    regions: list[str], start: int, end: int
) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "hash": hashlib.sha256(regions[index].encode()).hexdigest()[:12],
            "text": regions[index],
        }
        for index in range(start, end)
    ]


def _heading_diff(
    previous: list[PageHeading], current: list[PageHeading]
) -> dict[str, list[dict[str, Any]]]:
    old = [(item.level, item.text) for item in previous]
    new = [(item.level, item.text) for item in current]
    return {
        "added": [
            {"level": level, "text": text}
            for level, text in new
            if (level, text) not in old
        ],
        "removed": [
            {"level": level, "text": text}
            for level, text in old
            if (level, text) not in new
        ],
    }


def _metadata_diff(
    previous: PageState, current: PageState
) -> dict[str, dict[str, Any]]:
    values = {
        "looks_like_login": (
            previous.looks_like_login,
            current.looks_like_login,
        ),
        "truncated": (
            previous.truncated.model_dump(),
            current.truncated.model_dump(),
        ),
        "dialogs": (
            [item.model_dump() for item in previous.dialogs],
            [item.model_dump() for item in current.dialogs],
        ),
        "opaque_regions": (
            previous.opaque_regions,
            current.opaque_regions,
        ),
    }
    return {
        key: {"before": before, "after": after}
        for key, (before, after) in values.items()
        if before != after
    }


def _site_context_diff(
    previous: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if previous == current:
        return None
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return {"before": previous, "after": current}
    return _mapping_diff(previous, current, depth=0) or None


def _mapping_diff(
    previous: dict[str, Any], current: dict[str, Any], *, depth: int
) -> dict[str, Any]:
    if depth >= 8:
        return {"before": previous, "after": current}
    output: dict[str, Any] = {}
    for key in sorted(previous.keys() | current.keys()):
        before = previous.get(key)
        after = current.get(key)
        if before == after:
            continue
        if isinstance(before, dict) and isinstance(after, dict):
            output[key] = _mapping_diff(before, after, depth=depth + 1)
        elif _is_record_list(before) and _is_record_list(after):
            output[key] = _dict_list_diff(before, after)
        else:
            output[key] = {"before": before, "after": after}
    return output


def _is_record_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _dict_list_diff(previous: Any, current: Any) -> dict[str, list[Any]]:
    old = previous if isinstance(previous, list) else []
    new = current if isinstance(current, list) else []
    old_map = {_record_identity(item): item for item in old if isinstance(item, dict)}
    new_map = {_record_identity(item): item for item in new if isinstance(item, dict)}
    return {
        "added": [new_map[key] for key in new_map.keys() - old_map.keys()],
        "removed": [old_map[key] for key in old_map.keys() - new_map.keys()],
        "changed": [
            {"before": old_map[key], "after": new_map[key]}
            for key in old_map.keys() & new_map.keys()
            if old_map[key] != new_map[key]
        ],
    }


def _record_identity(item: dict[str, Any]) -> str:
    for key in (
        "job_id",
        "attachment_id",
        "download_url",
        "url",
        "href",
        "filename",
        "title",
        "agent_element_id",
    ):
        value = item.get(key)
        if value not in (None, ""):
            suffix = ""
            if key in {"filename", "title"}:
                suffix = f":{item.get('publisher', '')}:{item.get('year', '')}"
            return f"{key}:{value}{suffix}"
    return json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
