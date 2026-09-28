from __future__ import annotations

import pytest
from pydantic import ValidationError

from browser.actions import ActionResult, tool_schemas, validate_tool_call
from browser.providers import ToolCall


def call(tool: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        tool=tool,
        arguments=arguments,
        snapshot_id="s-1234",
        reason="test",
    )


def test_every_native_schema_is_strict_and_requires_decision_metadata() -> None:
    schemas = tool_schemas()
    assert {schema.name for schema in schemas} == {
            "get_page_state",
            "read_page",
            "get_element_details",
        "click",
        "type_text",
        "clear_input",
        "press_key",
        "scroll",
        "navigate", "go_back", "go_forward", "refresh",
        "wait_for_stable",
        "find_text",
        "get_tabs",
        "switch_tab",
        "open_tab",
        "close_tab",
    }
    for item in schemas:
        assert item.input_schema["additionalProperties"] is False
        assert "snapshot_id" in item.input_schema["required"]
        assert "reason" in item.input_schema["required"]


@pytest.mark.parametrize(
    "selector", ["#search", ".primary", "button#search", "[aria-label=Search]"]
)
def test_css_selectors_are_rejected_as_element_addresses(selector: str) -> None:
    with pytest.raises(ValidationError, match="CSS selectors"):
        validate_tool_call(call("click", {"element_id": selector}))


def test_frame_qualified_agent_id_is_accepted() -> None:
    parsed = validate_tool_call(call("click", {"element_id": "f0:e-034"}))
    assert parsed.element_id == "f0:e-034"  # type: ignore[attr-defined]


def test_extra_tool_arguments_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        validate_tool_call(
            call("click", {"element_id": "f0:e-034", "selector": "#search"})
        )


def test_password_field_is_a_typed_error() -> None:
    result = ActionResult(
        success=False,
        action="type_text",
        element_id="f0:e-001",
        snapshot_id="s-1",
        error="password_field",
    )
    assert result.error == "password_field"


def test_failed_result_requires_an_error() -> None:
    with pytest.raises(ValidationError):
        ActionResult(success=False, action="click", snapshot_id="s-1")
