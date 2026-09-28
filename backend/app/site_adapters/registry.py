"""Select one configured adapter by exact URL hostname."""

from __future__ import annotations

from browser.actions import tool_schemas as browser_tool_schemas
from browser.actions import validate_tool_call
from browser.config import SiteAdaptersSettings
from browser.protocol import PageState
from browser.providers import ToolCall, ToolSchema
from browser.tool_execution import BuiltNavigationCall, BuiltStateReadCall
from browser.actions import ActionResult
from browser.state_reads import unavailable

from .base import BaseSiteAdapter
from .generic import GenericSiteAdapter


class SiteAdapterRegistry:
    """Host-scoped adapter registry with a pass-through generic default."""

    def __init__(self, settings: SiteAdaptersSettings) -> None:
        self.generic = GenericSiteAdapter()
        self.adapters: list[BaseSiteAdapter] = []

    def adapter_for(self, url: str) -> BaseSiteAdapter:
        return next(
            (adapter for adapter in self.adapters if adapter.matches(url)),
            self.generic,
        )

    def refine_page_state(self, state: PageState) -> PageState:
        return self.adapter_for(state.url).refine_page_state(state)

    def system_prompt_addendum(self, url: str) -> str:
        adapter = self.adapter_for(url)
        guidance = adapter.system_prompt_addendum()
        return guidance

    def tool_schemas(self, url: str) -> list[ToolSchema]:
        """Return only the custom tools owned by the active URL adapter."""
        built_in = {schema.name for schema in browser_tool_schemas()}
        schemas: list[ToolSchema] = []
        for item in self.adapter_for(url).extra_tools():
            schema = (
                item
                if isinstance(item, ToolSchema)
                else ToolSchema.model_validate(item)
            )
            input_schema = dict(schema.input_schema)
            properties = dict(input_schema.get("properties", {}))
            properties["snapshot_id"] = {
                "type": "string",
                "description": (
                    "The snapshot_id this decision was made against."
                ),
            }
            properties["reason"] = {
                "type": "string",
                "description": (
                    "A concise explanation shown in the step timeline."
                ),
            }
            required = list(input_schema.get("required", []))
            for common_field in ("snapshot_id", "reason"):
                if common_field not in required:
                    required.append(common_field)
            schemas.append(
                schema.model_copy(
                    update={
                        "input_schema": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False,
                        }
                    }
                )
            )
        names = [schema.name for schema in schemas]
        if len(names) != len(set(names)):
            raise ValueError("site adapter exposes duplicate tool names")
        if any(name in built_in for name in names):
            raise ValueError("site adapter tool shadows a browser tool")
        return schemas

    def translate_tool_call(
        self, url: str, call: ToolCall
    ) -> BuiltNavigationCall | None:
        """Validate and lower an active adapter tool to browser navigation."""
        adapter = self.adapter_for(url)
        names = {schema.name for schema in self.tool_schemas(url)}
        if call.tool not in names:
            return None
        translated = adapter.translate_tool_call(call.tool, call.arguments)
        if translated is None:
            raise ValueError(f"adapter did not translate owned tool {call.tool!r}")
        name, arguments = translated
        if name != "navigate":
            raise ValueError("site adapter tools may only lower to navigate")
        destination = str(arguments.get("url", ""))
        if not adapter.matches(destination):
            raise ValueError("site adapter generated a destination outside its scope")
        execution = BuiltNavigationCall(
            tool=name,
            arguments=arguments,
            snapshot_id=call.snapshot_id,
            reason=call.reason,
        )
        validate_tool_call(execution)
        return execution

    def prepare_state_read(self, url: str, call: ToolCall) -> BuiltStateReadCall | None:
        if call.tool not in {schema.name for schema in self.tool_schemas(url)}:
            return None
        arguments = self.adapter_for(url).validate_state_read(call.tool, call.arguments)
        if arguments is None:
            return None
        return BuiltStateReadCall(tool=call.tool, arguments=arguments,
                                  snapshot_id=call.snapshot_id, reason=call.reason)

    def read_state(self, call: ToolCall, state: PageState) -> ActionResult:
        if not isinstance(call, BuiltStateReadCall):
            return unavailable(call, state)
        return self.adapter_for(state.url).read_state(call, state) or unavailable(call, state)

    def tool_call_satisfied(
        self, url: str, call: ToolCall, state: PageState
    ) -> bool:
        """Delegate semantic post-action matching to the active adapter."""
        return self.adapter_for(url).tool_call_satisfied(
            call.tool, call.arguments, state
        )
