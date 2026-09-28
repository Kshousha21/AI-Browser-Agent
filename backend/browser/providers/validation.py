"""Strict provider-neutral validation before a native tool becomes an action."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator, SchemaError

from .base import ProviderError, ToolCall, ToolSchema, tool_decision


def validated_tool(name: str, arguments: Any, tools: list[ToolSchema] | None) -> ToolCall:
    schema = next((tool.input_schema for tool in tools or [] if tool.name == name), None)
    if schema is None:
        raise ProviderError("provider returned an unadvertised native tool", code="invalid_tool")
    try:
        Draft202012Validator.check_schema(schema)
        errors = list(Draft202012Validator(schema).iter_errors(arguments))
    except SchemaError:
        raise ProviderError("provider tool definition has an invalid schema", code="invalid_schema") from None
    if errors:
        # Locations/keywords suffice for diagnosis; never echo model arguments.
        keywords = sorted({str(error.validator) for error in errors})
        raise ProviderError("native tool arguments failed schema validation: " + ", ".join(keywords), code="invalid_arguments")
    return tool_decision(name, arguments)


def usage_counts(prompt: Any, completion: Any, total: Any = None) -> dict[str, int]:
    out: dict[str, int] = {}
    if type(prompt) is int and prompt >= 0:
        out["prompt_tokens"] = prompt
    if type(completion) is int and completion >= 0:
        out["completion_tokens"] = completion
    if len(out) == 2:
        out["total_tokens"] = prompt + completion
    if type(total) is int and total >= 0:
        out["total_tokens"] = total
    return out
