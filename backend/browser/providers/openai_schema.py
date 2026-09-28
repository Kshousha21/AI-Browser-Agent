"""OpenAI strict-schema projection; original full schemas stay authoritative."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

def _strict_function_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Return OpenAI strict-mode JSON Schema without changing shared schemas."""
    output = deepcopy(schema)
    _make_optional_properties_nullable(output)
    _drop_unsupported_schema_keywords(output)
    _require_all_object_properties(output)
    return output


_OPENAI_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"default", "title", "minLength", "maxLength"}
)


def _drop_unsupported_schema_keywords(node: object) -> None:
    if isinstance(node, dict):
        for key in _OPENAI_UNSUPPORTED_SCHEMA_KEYS:
            node.pop(key, None)
        for value in node.values():
            _drop_unsupported_schema_keywords(value)
    elif isinstance(node, list):
        for value in node:
            _drop_unsupported_schema_keywords(value)


def _make_optional_properties_nullable(node: object) -> None:
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            required = set(node.get("required", []))
            for name, value in list(properties.items()):
                if (
                    name not in required
                    and isinstance(value, dict)
                    and not _allows_null(value)
                ):
                    properties[name] = {
                        "anyOf": [value, {"type": "null"}]
                    }
        for value in node.values():
            _make_optional_properties_nullable(value)
    elif isinstance(node, list):
        for value in node:
            _make_optional_properties_nullable(value)


def _allows_null(schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    if kind == "null" or (isinstance(kind, list) and "null" in kind):
        return True
    return any(
        isinstance(choice, dict) and _allows_null(choice)
        for keyword in ("anyOf", "oneOf")
        for choice in (
            schema.get(keyword, [])
            if isinstance(schema.get(keyword), list)
            else []
        )
    )


def _restore_optional_defaults(value: Any, schema: dict[str, Any]) -> Any:
    """Turn transport-only nulls back into omissions before local validation."""
    if isinstance(value, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return value
        required = set(schema.get("required", []))
        restored: dict[str, Any] = {}
        for key, item in value.items():
            child_schema = properties.get(key)
            if item is None and key in properties and key not in required:
                continue
            restored[key] = (
                _restore_optional_defaults(item, child_schema)
                if isinstance(child_schema, dict)
                else item
            )
        return restored
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [
            _restore_optional_defaults(item, schema["items"])
            for item in value
        ]
    return value


def _require_all_object_properties(node: object) -> None:
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = list(properties)
            node["additionalProperties"] = False
        for value in node.values():
            _require_all_object_properties(value)
    elif isinstance(node, list):
        for value in node:
            _require_all_object_properties(value)


