"""Bounded, model-facing PageState payload construction."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

from .protocol import PageElement, PageHeading, PageState

_GENERIC_TEXT_LIMIT = 12_000
_ADAPTER_TEXT_LIMIT = 5_000
_GENERIC_ELEMENT_LIMIT = 220
_ADAPTER_ELEMENT_LIMIT = 32
_ADAPTER_SITE_CONTEXT_CHAR_LIMIT = 32_000


def full_page_payload(page: PageState, *, reason: str = "initial") -> dict[str, Any]:
    """Build the ordered, model-facing full PageState representation."""
    elements = context_elements(page)
    text, context_truncated = context_text(page)
    headings = context_headings(page)
    return {
        "type": "page_state_full",
        "tab_id": page.tab_id,
        "snapshot_id": page.snapshot_id,
        "document_id": page.document_id,
        "navigation_id": page.navigation_id,
        "full_snapshot_reason": reason,
        "url": page.url,
        "title": page.title,
        "looks_like_login": page.looks_like_login,
        "warnings": page.warnings,
        "site_context": model_site_context(page.site_context),
        "elements": [element_payload(item) for item in elements],
        "headings": [item.model_dump() for item in headings],
        "visible_text": text,
        "frames": [item.model_dump() for item in page.frames],
        "dialogs": [item.model_dump() for item in page.dialogs],
        "opaque_regions": page.opaque_regions,
        "truncated": {
            "text": page.truncated.text or context_truncated,
            "elements": (
                page.truncated.elements or len(elements) < len(page.elements)
            ),
        },
    }


def serialized_size(payload: dict[str, Any]) -> int:
    return len(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    )


def context_elements(page: PageState) -> list[PageElement]:
    represented = adapter_agent_ids(page.site_context)
    maximum = _ADAPTER_ELEMENT_LIMIT if page.site_context else _GENERIC_ELEMENT_LIMIT
    controls = [item for item in page.elements if item.agent_id not in represented]
    visible = [item for item in controls if item.visible]
    hidden = [item for item in controls if not item.visible]
    hidden_limit = 8 if page.site_context else 20
    hidden_budget = min(hidden_limit, max(0, maximum - len(visible)))
    return [*visible[:maximum], *hidden[:hidden_budget]][:maximum]


def context_headings(page: PageState) -> list[PageHeading]:
    represented = site_values(page.site_context)
    output: list[PageHeading] = []
    seen: set[tuple[int, str]] = set()
    for item in page.headings:
        normalized_text = normalized(item.text)
        key = (item.level, normalized_text)
        if not normalized_text or normalized_text in represented or key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= 80:
            break
    return output


def context_text(page: PageState) -> tuple[str, bool]:
    limit = _ADAPTER_TEXT_LIMIT if page.site_context else _GENERIC_TEXT_LIMIT
    represented = site_values(page.site_context)
    blocks: list[str] = []
    for block in text_regions(page.visible_text):
        normalized_block = normalized(block)
        if normalized_block in represented:
            continue
        blocks.append(block)
    text = "\n".join(blocks)
    return (text[:limit], len(text) > limit)


def element_payload(item: PageElement) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "agent_id": item.agent_id,
        "tag": item.tag,
        "role": item.role,
    }
    if item.visible:
        for key, value, limit in (
            ("name", item.name, 300),
            ("text", item.text, 500),
            ("placeholder", item.placeholder, 200),
            ("value", item.value, 300),
            ("input_type", item.input_type, 40),
        ):
            if value:
                payload[key] = value[:limit]
    if item.href:
        payload["href"] = display_url(item.href)
    if item.form_action:
        payload["form_action"] = display_url(item.form_action)
    if item.form_method:
        payload["form_method"] = item.form_method
    if item.form_target:
        payload["form_target"] = item.form_target[:100]
    if item.selected:
        payload["selected"] = True
    flags: list[str] = []
    if not item.enabled:
        flags.append("disabled")
    if not item.visible:
        flags.append("hidden; content withheld")
    if item.occluded:
        flags.append("occluded")
    if flags:
        payload["flags"] = flags
    return payload


def display_element_value(field: str, value: Any) -> Any:
    if field in {"href", "form_action"} and isinstance(value, str):
        return display_url(value)
    if field in {"name", "value"} and isinstance(value, str):
        return value[:300]
    if field in {"text", "placeholder"} and isinstance(value, str):
        return value[:500]
    return value


def display_url(value: str) -> str:
    """Retain destination meaning but omit tracking payloads from model context."""
    if len(value) <= 320:
        return value
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return value[:320]


def model_site_context(
    context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep bounded normalized meaning without exposing adapter evidence."""
    if not context:
        return None
    if context.get("adapter") == "GenericSiteAdapter":
        return {"adapter": "GenericSiteAdapter", "site": "generic", "page_kind": "page",
                "file_count": len(context.get("files", [])),
                "file_metadata_tool": "get_page_files"}
    bounded = _bounded_site_value(context)
    return _fit_adapter_site_context(bounded) if isinstance(bounded, dict) else None


_FORBIDDEN_CONTEXT_KEYS = {
    "html",
    "raw_html",
    "inner_html",
    "outer_html",
    "markup",
    "adapter_payload",
}


def _bounded_site_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, item in list(value.items())[:64]:
            child_key = str(raw_key)[:100]
            if child_key.casefold() in _FORBIDDEN_CONTEXT_KEYS:
                continue
            child = _bounded_site_value(item, key=child_key, depth=depth + 1)
            if child not in (None, "", [], {}):
                output[child_key] = child
        return output
    if isinstance(value, list):
        return [
            item
            for item in (
                _bounded_site_value(item, key=key, depth=depth + 1)
                for item in value[:200]
            )
            if item not in (None, "", [], {})
        ]
    if isinstance(value, str):
        limit = 1_200 if key in {"excerpt", "notes", "byline"} else 6_000
        text = value[:limit]
        if key == "href" or key == "url" or key.endswith("_url"):
            return display_url(text)
        return text
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return str(value)[:500]


def _fit_adapter_site_context(context: dict[str, Any]) -> dict[str, Any]:
    """Bound large normalized record families without emitting invalid JSON."""
    if _json_chars(context) <= _ADAPTER_SITE_CONTEXT_CHAR_LIMIT:
        return context

    record_lists: list[tuple[str, list[Any]]] = []
    search = context.get("search")
    if isinstance(search, dict) and isinstance(search.get("results"), list):
        record_lists.append(("search_results", search["results"]))
    library = context.get("study_library")
    if isinstance(library, dict) and isinstance(library.get("studies"), list):
        record_lists.append(("studies", library["studies"]))
    if isinstance(context.get("attachments"), list):
        record_lists.append(("attachments", context["attachments"]))

    omitted = {name: 0 for name, _items in record_lists}
    context["model_context"] = {
        "truncated": True,
        "omitted_records": omitted,
    }
    while (
        _json_chars(context) > _ADAPTER_SITE_CONTEXT_CHAR_LIMIT
        and any(items for _name, items in record_lists)
    ):
        name, items = max(
            (
                (name, items)
                for name, items in record_lists
                if items
            ),
            key=lambda pair: _json_chars(pair[1][-1]),
        )
        items.pop()
        omitted[name] += 1

    if _json_chars(context) <= _ADAPTER_SITE_CONTEXT_CHAR_LIMIT:
        return context

    # Maliciously maximal scalar metadata must not defeat the aggregate cap.
    return {
        key: context[key]
        for key in ("site", "platform", "page_kind", "auth")
        if key in context
    } | {
        "model_context": {
            "truncated": True,
            "omitted_records": omitted,
            "metadata_reduced": True,
        }
    }


def _json_chars(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def adapter_agent_ids(context: dict[str, Any] | None) -> set[str]:
    if not context:
        return set()
    output: set[str] = set()
    stack: list[Any] = [context]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            agent_id = value.get("agent_element_id")
            if isinstance(agent_id, str) and agent_id:
                output.add(agent_id)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return output


def site_values(context: dict[str, Any] | None) -> set[str]:
    if not context:
        return set()
    output: set[str] = set()
    stack: list[Any] = [context]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str) and len(value) <= 2_000:
            normalized_value = normalized(value)
            if normalized_value:
                output.add(normalized_value)
    return output


def text_regions(value: str) -> list[str]:
    return [
        " ".join(block.split())
        for block in value.splitlines()
        if " ".join(block.split())
    ]


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())
