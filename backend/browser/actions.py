"""Strict browser-tool schemas shared by the agent loop and providers."""

from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .providers import ToolCall, ToolSchema

ElementId = str
ToolError = Literal[
    "tool_not_available_in_current_profile",
    "no_active_search_results",
    "element_not_found",
    "stale_snapshot",
    "target_tab_changed",
    "target_document_changed",
    "target_tab_closed",
    "run_paused",
    "element_occluded",
    "element_disabled",
    "timeout",
    "blocked_by_dialog",
    "navigation_blocked",
    "permission_denied",
    "frame_unreachable",
    "injection_failed",
    "invalid_arguments",
    "password_field",
]

_ELEMENT_ID = re.compile(r"^f\d+:e-\d{3,}$")
_SELECTORISH = re.compile(
    r"^(?:[#.][\w-]+|\[[^\]]+\]|(?:[a-zA-Z][\w-]*)(?:[#.\[][\s\S]*))$"
)


class StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyArgs(StrictArgs):
    pass


class GetPageStateArgs(StrictArgs):
    full: bool = False


class ReadPageArgs(StrictArgs):
    max_chars: int | None = Field(default=None, ge=1, le=100_000)


class ElementArgs(StrictArgs):
    element_id: ElementId

    @field_validator("element_id")
    @classmethod
    def valid_element_id(cls, value: str) -> str:
        if not _ELEMENT_ID.fullmatch(value):
            hint = " (CSS selectors are forbidden)" if looks_like_selector(value) else ""
            raise ValueError(f"must be a frame-qualified agent_id{hint}")
        return value


class TypeTextArgs(ElementArgs):
    text: str = Field(max_length=20_000)
    submit: bool = False


class PressKeyArgs(StrictArgs):
    key: str = Field(min_length=1, max_length=64)


class ScrollArgs(StrictArgs):
    direction: Literal["up", "down", "left", "right"]
    amount: int | None = Field(default=600, ge=1, le=20_000)
    element_id: ElementId | None = None

    @field_validator("element_id")
    @classmethod
    def valid_optional_element_id(cls, value: str | None) -> str | None:
        if value is not None and not _ELEMENT_ID.fullmatch(value):
            raise ValueError("must be a frame-qualified agent_id; CSS selectors are forbidden")
        return value


class NavigateArgs(StrictArgs):
    url: str

    @field_validator("url")
    @classmethod
    def http_url_only(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an absolute http(s) URL")
        return value


class TabArgs(StrictArgs):
    tab_id: int = Field(ge=0, strict=True)


class WaitForStableArgs(StrictArgs):
    timeout_ms: int = Field(default=10_000, ge=100, le=60_000)


class FindTextArgs(StrictArgs):
    text: str = Field(min_length=1, max_length=2_000)


class ActionResult(BaseModel):
    """Typed result returned by the extension for every browser tool."""

    success: bool
    action: str
    element_id: str | None = None
    element_name: str | None = None
    snapshot_id: str
    page_changed: bool = False
    reresolved: bool = False
    duration_ms: int = Field(default=0, ge=0)
    error: ToolError | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def failure_has_error(self) -> "ActionResult":
        if not self.success and self.error is None:
            raise ValueError("failed tool results require an error")
        return self


ARGUMENT_MODELS: dict[str, type[StrictArgs]] = {
    "get_page_state": GetPageStateArgs,
    "read_page": ReadPageArgs,
    "get_element_details": ElementArgs,
    "click": ElementArgs,
    "type_text": TypeTextArgs,
    "clear_input": ElementArgs,
    "press_key": PressKeyArgs,
    "scroll": ScrollArgs,
    "navigate": NavigateArgs,
    "go_back": EmptyArgs,
    "go_forward": EmptyArgs,
    "refresh": EmptyArgs,
    "get_tabs": EmptyArgs,
    "switch_tab": TabArgs,
    "open_tab": NavigateArgs,
    "close_tab": TabArgs,
    "wait_for_stable": WaitForStableArgs,
    "find_text": FindTextArgs,
}

_DESCRIPTIONS = {
    "get_tabs": "List bounded tab metadata; unrelated tab details are redacted.",
    "switch_tab": "Switch to an allowed tab without navigating; obtain fresh full state.",
    "open_tab": "Open an allowed product URL in a new tab and read fresh full state.",
    "close_tab": "Close one allowed tab only after explicit confirmation. Never closes a window.",
    "get_page_state": "Read fresh page state. Set full=true only when a complete snapshot is required instead of a safe delta.",
    "read_page": "Read visible page text, optionally capped to max_chars.",
    "get_element_details": "Inspect one retained element without resending the full page.",
    "click": "Click an element by agent_id using a full pointer sequence.",
    "type_text": "Replace text in an input, textarea, or contenteditable using native setters and events.",
    "clear_input": "Clear a non-password text control using its native setter and events.",
    "press_key": "Dispatch a keyboard key to the currently focused element.",
    "scroll": "Scroll the page or bring an element into view.",
    "navigate": "Navigate to an observed or user-supplied absolute HTTP(S) URL.",
    "go_back": "Go back in this tab's history and read the destination.",
    "go_forward": "Go forward in this tab's history and read the destination.",
    "refresh": "Reload this tab and read a fresh snapshot.",
    "wait_for_stable": "Wait for readyState complete, network idle, and a DOM mutation quiet period.",
    "find_text": "Find visible occurrences of text without using a selector.",
}


def looks_like_selector(value: str) -> bool:
    """Return true for common CSS-selector shapes used as element addresses."""
    return bool(_SELECTORISH.fullmatch(value.strip()))


def tool_schemas() -> list[ToolSchema]:
    """Return strict native schemas, including decision metadata."""
    schemas: list[ToolSchema] = []
    for name, model in ARGUMENT_MODELS.items():
        schema = model.model_json_schema()
        properties = dict(schema.get("properties", {}))
        properties["snapshot_id"] = {
            "type": "string",
            "description": "The snapshot_id this decision was made against.",
        }
        properties["reason"] = {
            "type": "string",
            "description": "A concise explanation shown in the step timeline.",
        }
        required = list(schema.get("required", []))
        required.extend(["snapshot_id", "reason"])
        schemas.append(
            ToolSchema(
                name=name,
                description=_DESCRIPTIONS[name],
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            )
        )
    return schemas


def validate_tool_call(call: ToolCall) -> StrictArgs:
    """Validate a normalized native tool call against its exact schema."""
    model = ARGUMENT_MODELS.get(call.tool)
    if model is None:
        raise ValueError(f"unknown browser tool {call.tool!r}")
    return model.model_validate(call.arguments)
