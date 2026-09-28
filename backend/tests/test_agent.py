from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.site_adapters import SiteAdapterRegistry
from browser.actions import ActionResult
from browser.agent import LoopDetector, TokenBudget, run_agent
from browser.config import (
    AgentSettings,
    SecuritySettings,
    SiteAdapterEntrySettings,
    SiteAdaptersSettings,
)
from browser.context_control import compact_result
from browser.protocol import PageElement, PageState
from browser.providers import FinalAnswer, MockProvider, ProviderResponse, ToolCall
from browser.security import ConfirmationChoice, RiskAssessment
from browser.tool_execution import BuiltNavigationCall


PAGE = PageState(
    snapshot_id="s-start",
    document_id="d-test",
    url="http://127.0.0.1:8080/",
    title="Search",
    visible_text="Search the handbook",
    elements=[
        PageElement(
            agent_id="f0:e-002",
            tag="input",
            role="searchbox",
            name="Search the handbook",
        ),
        PageElement(
            agent_id="f0:e-003",
            tag="button",
            role="button",
            name="Search",
        ),
    ],
)


@dataclass
class FakeBridge:
    results: list[ActionResult]
    calls: list[ToolCall] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    timeline_steps: list[str] = field(default_factory=list)
    confirmation_choices: list[ConfirmationChoice] = field(default_factory=list)
    confirmation_scopes: list[str] = field(default_factory=list)
    audit_records: list[tuple[str, str]] = field(default_factory=list)
    stopped: bool = False

    async def execute(
        self, call: ToolCall, step: int, max_steps: int
    ) -> ActionResult:
        del step, max_steps
        self.calls.append(call)
        return self.results[len(self.calls) - 1]

    async def status(
        self,
        status: str,
        step: int,
        max_steps: int,
        elapsed_ms: int,
        usage: dict[str, int],
    ) -> None:
        del step, max_steps, elapsed_ms, usage
        self.statuses.append(status)

    async def timeline(
        self, call: ToolCall, result: ActionResult, step: int
    ) -> None:
        del result, step
        self.timeline_steps.append(call.tool)

    async def confirm(
        self, call: ToolCall, assessment: RiskAssessment, step: int
    ) -> ConfirmationChoice:
        del call, step
        self.confirmation_scopes.append(assessment.scope)
        return self.confirmation_choices.pop(0) if self.confirmation_choices else "cancel"

    async def audit(
        self,
        call: ToolCall,
        result: ActionResult,
        confirmation: str,
        password_target: bool = False,
    ) -> None:
        del result, password_target
        self.audit_records.append((call.tool, confirmation))


class ToolRecordingMockProvider(MockProvider):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.tool_names: list[list[str]] = []

    async def generate(
        self, messages: list[object], tools: list[object] | None = None,
        response_schema: object = None,
    ) -> ProviderResponse:
        self.tool_names.append(
            [str(getattr(item, "name", "")) for item in (tools or [])]
        )
        return await super().generate(  # type: ignore[arg-type]
            messages, tools=tools, response_schema=response_schema
        )


def tool(name: str, arguments: dict[str, object], snapshot: str = "s-start") -> ToolCall:
    return ToolCall(
        tool=name,
        arguments=arguments,
        snapshot_id=snapshot,
        reason=f"Use {name}",
    )








@pytest.mark.asyncio
async def test_full_mocked_action_run_needs_no_api_key_or_network() -> None:
    provider = MockProvider(
        scripted=[
            tool("type_text", {"element_id": "f0:e-002", "text": "photonics"}),
            tool("click", {"element_id": "f0:e-003"}),
            ProviderResponse(
                decision=FinalAnswer(answer="Found the photonics results."),
                raw_text="Found the photonics results.",
                usage={"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            ),
        ]
    )
    bridge = FakeBridge(
        results=[
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id="s-fresh",
                data={
                    "page_state": PAGE.model_copy(
                        update={"snapshot_id": "s-fresh"}
                    ).model_dump()
                },
            ),
            ActionResult(
                success=True,
                action="type_text",
                element_id="f0:e-002",
                snapshot_id="s-fresh",
            ),
            ActionResult(
                success=True,
                action="click",
                element_id="f0:e-003",
                snapshot_id="s-start",
                page_changed=True,
            ),
        ]
    )

    outcome = await run_agent(
        "Search for photonics",
        PAGE,
        ["http://127.0.0.1:8080/*"],
        provider,
        AgentSettings(),
        bridge,
        security_settings=SecuritySettings(),
        read_only=False,
    )

    assert [call.tool for call in bridge.calls] == [
        "get_page_state",
        "type_text",
        "click",
    ]
    assert bridge.timeline_steps == ["get_page_state", "type_text", "click"]
    assert outcome.text == "Found the photonics results."
    assert len(provider.calls) == 3
    assert [item[0] for item in bridge.audit_records] == [
        "get_page_state",
        "type_text",
        "click",
    ]


@pytest.mark.asyncio
async def test_same_page_job_click_waits_reads_and_sends_a_diff() -> None:
    stable_text = "\n".join(f"Stable result {index}" for index in range(40))
    start = PAGE.model_copy(
        update={
            "snapshot_id": "s-list",
            "visible_text": stable_text,
            "elements": [
                PageElement(
                    agent_id="f0:e-010",
                    tag="a",
                    role="button",
                    name="Open AI Engineer",
                    href="https://example.test/job/1",
                )
            ],
        }
    )
    detail = start.model_copy(
        update={
            "snapshot_id": "s-detail",
            "visible_text": stable_text.replace(
                "Stable result 20", "Selected detail for AI Engineer"
            ),
        }
    )
    provider = MockProvider(
        scripted=[
            tool("click", {"element_id": "f0:e-010"}, snapshot="s-list"),
            ProviderResponse(
                decision=FinalAnswer(answer="The selected detail is visible."),
                raw_text="The selected detail is visible.",
            ),
        ]
    )
    bridge = FakeBridge(
        results=[
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id="s-list",
                data={"page_state": start.model_dump()},
            ),
            ActionResult(
                success=True,
                action="click",
                element_id="f0:e-010",
                snapshot_id="s-list",
                page_changed=False,
            ),
            ActionResult(
                success=True,
                action="wait_for_stable",
                snapshot_id="s-list",
            ),
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id="s-detail",
                data={"page_state": detail.model_dump()},
            ),
        ]
    )
    outcome = await run_agent(
        "Open the AI Engineer job",
        start,
        ["https://example.test/*"],
        provider,
        AgentSettings(),
        bridge,
        security_settings=SecuritySettings(),
        read_only=False,
    )
    assert outcome.text == "The selected detail is visible."
    assert [item.tool for item in bridge.calls] == [
        "get_page_state",
        "click",
        "wait_for_stable",
        "get_page_state",
    ]
    assert bridge.timeline_steps == [
        "get_page_state",
        "click",
        "wait_for_stable",
        "get_page_state",
    ]
    assert "page_state_diff" in str(provider.calls[-1][-1].content)
    assert "Selected detail for AI Engineer" in str(
        provider.calls[-1][-1].content
    )


@pytest.mark.asyncio
async def test_occluded_target_requires_wait_fresh_state_and_reacquisition() -> None:
    fresh = PAGE.model_copy(
        update={
            "snapshot_id": "s-fresh",
            "elements": [
                PAGE.elements[0],
                PAGE.elements[1].model_copy(update={"occluded": True}),
            ],
        }
    )
    after_overlay = fresh.model_copy(
        update={
            "snapshot_id": "s-overlay-gone",
            "elements": [
                fresh.elements[0],
                fresh.elements[1].model_copy(update={"occluded": False}),
            ],
        }
    )
    provider = MockProvider(
        scripted=[
            tool("click", {"element_id": "f0:e-003"}, snapshot="s-fresh"),
            tool("wait_for_stable", {}, snapshot="s-fresh"),
            tool("get_page_state", {}, snapshot="s-fresh"),
            tool(
                "click",
                {"element_id": "f0:e-003"},
                snapshot="s-overlay-gone",
            ),
            ProviderResponse(
                decision=FinalAnswer(answer="Search started."),
                raw_text="Search started.",
            ),
        ]
    )
    bridge = FakeBridge(
        results=[
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id="s-fresh",
                data={"page_state": fresh.model_dump()},
            ),
            ActionResult(
                success=False,
                action="click",
                element_id="f0:e-003",
                snapshot_id="s-fresh",
                error="element_occluded",
            ),
            ActionResult(
                success=True,
                action="wait_for_stable",
                snapshot_id="s-fresh",
            ),
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id="s-overlay-gone",
                data={"page_state": after_overlay.model_dump()},
            ),
            ActionResult(
                success=True,
                action="click",
                element_id="f0:e-003",
                snapshot_id="s-overlay-gone",
                page_changed=True,
            ),
        ]
    )
    outcome = await run_agent(
        "Search",
        fresh,
        ["http://127.0.0.1:8080/*"],
        provider,
        AgentSettings(),
        bridge,
        security_settings=SecuritySettings(),
        read_only=False,
    )
    assert outcome.text == "Search started."
    assert [item.tool for item in bridge.calls] == [
        "get_page_state",
        "click",
        "wait_for_stable",
        "get_page_state",
        "click",
    ]
    assert bridge.calls[-1].snapshot_id == "s-overlay-gone"
    assert "element_occluded" in str(provider.calls[1][-1].content)


@pytest.mark.asyncio
async def test_cancelled_high_risk_action_is_not_dispatched_or_reconfirmed() -> None:
    submit_page = PAGE.model_copy(
        update={
            "elements": [
                *PAGE.elements,
                PageElement(
                    agent_id="f0:e-004",
                    tag="button",
                    role="button",
                    name="Delete request",
                    input_type="submit",
                ),
            ]
        }
    )
    submit = tool("click", {"element_id": "f0:e-004"})
    provider = MockProvider(
        scripted=[
            submit,
            submit,
            ProviderResponse(
                decision=FinalAnswer(answer="The user cancelled; no request was sent."),
                raw_text="The user cancelled; no request was sent.",
            ),
        ]
    )
    bridge = FakeBridge(
        results=[
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id=submit_page.snapshot_id,
                data={"page_state": submit_page.model_dump()},
            )
        ],
        confirmation_choices=["cancel"],
    )
    outcome = await run_agent(
        "Delete the request",
        submit_page,
        ["http://127.0.0.1:8080/*"],
        provider,
        AgentSettings(),
        bridge,
        security_settings=SecuritySettings(),
        read_only=False,
    )
    assert outcome.text == "The user cancelled; no request was sent."
    assert [item.tool for item in bridge.calls] == ["get_page_state"]
    assert bridge.confirmation_scopes == ["click:element:f0:e-004"]
    assert bridge.audit_records[-2:] == [
        ("click", "cancel"),
        ("click", "loop_blocked"),
    ]


@pytest.mark.asyncio
async def test_changed_approved_link_is_not_dispatched_and_requires_confirmation_again() -> None:
    original = PAGE.model_copy(
        update={
            "elements": [
                PageElement(
                    agent_id="f0:e-004",
                    tag="a",
                    role="button",
                    name="Delete request",
                    href="https://portal.example/submit?version=1",
                )
            ]
        }
    )
    changed = original.model_copy(
        update={
            "snapshot_id": "s-changed",
            "elements": [
                original.elements[0].model_copy(
                    update={"href": "https://portal.example/submit?version=2"}
                )
            ],
        }
    )
    action = tool("click", {"element_id": "f0:e-004"})
    provider = MockProvider(
        scripted=[
            action,
            action.model_copy(update={"snapshot_id": changed.snapshot_id}),
            ProviderResponse(
                decision=FinalAnswer(answer="The changed target was not clicked."),
                raw_text="The changed target was not clicked.",
            ),
        ]
    )
    bridge = FakeBridge(
        results=[
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id=original.snapshot_id,
                data={"page_state": original.model_dump()},
            ),
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id=changed.snapshot_id,
                data={"page_state": changed.model_dump()},
            ),
        ],
        confirmation_choices=["allow_run", "cancel"],
    )
    outcome = await run_agent(
        "Delete the request",
        original,
        ["https://portal.example/*"],
        provider,
        AgentSettings(),
        bridge,
        security_settings=SecuritySettings(),
        read_only=False,
    )
    assert outcome.text == "The changed target was not clicked."
    assert [item.tool for item in bridge.calls] == [
        "get_page_state",
        "get_page_state",
    ]
    assert bridge.confirmation_scopes == [
        "click:element:f0:e-004",
        "click:element:f0:e-004",
    ]
    assert ("click", "allow_run_invalidated") in bridge.audit_records
    assert bridge.audit_records[-1] == ("click", "cancel")


@pytest.mark.asyncio
async def test_schema_violation_is_returned_once_then_model_can_recover() -> None:
    provider = MockProvider(
        scripted=[
            tool("click", {"element_id": "#search"}),
            ProviderResponse(
                decision=FinalAnswer(answer="I corrected the invalid call."),
                raw_text="I corrected the invalid call.",
            ),
        ]
    )
    bridge = FakeBridge(
        results=[
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id="s-fresh",
                data={
                    "page_state": PAGE.model_copy(
                        update={"snapshot_id": "s-fresh"}
                    ).model_dump()
                },
            )
        ]
    )

    outcome = await run_agent(
        "Search",
        PAGE,
        ["http://127.0.0.1:8080/*"],
        provider,
        AgentSettings(),
        bridge,
    )

    assert outcome.text == "I corrected the invalid call."
    assert provider.calls[1][-1].role == "tool"
    assert "validation failed" in str(provider.calls[1][-1].content).lower()
    assert [call.tool for call in bridge.calls] == ["get_page_state"]


@pytest.mark.asyncio
async def test_stop_is_honored_after_the_current_step_boundary() -> None:
    class StoppingBridge(FakeBridge):
        async def execute(
            self, call: ToolCall, step: int, max_steps: int
        ) -> ActionResult:
            result = await super().execute(call, step, max_steps)
            self.stopped = True
            return result

    bridge = StoppingBridge(
        results=[
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id="s-fresh",
                data={"page_state": PAGE.model_dump()},
            )
        ]
    )
    provider = MockProvider(scripted=["must not be called"])

    outcome = await run_agent(
        "Search",
        PAGE,
        ["http://127.0.0.1:8080/*"],
        provider,
        AgentSettings(),
        bridge,
    )

    assert outcome.finish_reason == "stopped"
    assert [item.tool for item in bridge.calls] == ["get_page_state"]
    assert provider.calls == []


@pytest.mark.asyncio
async def test_run_aborts_when_provider_usage_reaches_token_budget() -> None:
    provider = MockProvider(
        scripted=[
            ProviderResponse(
                decision=FinalAnswer(answer="This answer must not be returned."),
                raw_text="This answer must not be returned.",
                usage={
                    "prompt_tokens": 900,
                    "completion_tokens": 100,
                    "total_tokens": 1000,
                },
            )
        ]
    )
    bridge = FakeBridge(
        results=[
            ActionResult(
                success=True,
                action="get_page_state",
                snapshot_id=PAGE.snapshot_id,
                data={"page_state": PAGE.model_dump()},
            )
        ]
    )
    outcome = await run_agent(
        "Search",
        PAGE,
        ["http://127.0.0.1:8080/*"],
        provider,
        AgentSettings(token_budget=1000),
        bridge,
    )
    assert outcome.finish_reason == "token_budget"


@pytest.mark.asyncio
async def test_three_consecutive_tool_failures_abort_run() -> None:
    provider = MockProvider(
        scripted=[
            tool("type_text", {"element_id": "f0:e-002", "text": value})
            for value in ("one", "two", "three")
        ]
    )
    initial = ActionResult(
        success=True,
        action="get_page_state",
        snapshot_id=PAGE.snapshot_id,
        data={"page_state": PAGE.model_dump()},
    )
    failures = [
        ActionResult(
            success=False,
            action="type_text",
            snapshot_id=PAGE.snapshot_id,
            error="element_not_found",
        )
        for _ in range(3)
    ]
    bridge = FakeBridge(results=[initial, *failures])
    outcome = await run_agent(
        "Search",
        PAGE,
        ["http://127.0.0.1:8080/*"],
        provider,
        AgentSettings(),
        bridge,
    )
    assert outcome.finish_reason == "tool_failures"
    assert len(bridge.calls) == 4


def test_token_budget_charges_total_once() -> None:
    budget = TokenBudget(100)
    budget.record({"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40})
    budget.record({"prompt_tokens": 45, "completion_tokens": 15, "total_tokens": 60})
    assert budget.used == 100
    assert budget.exceeded
    assert budget.usage == {
        "prompt_tokens": 75,
        "completion_tokens": 25,
        "total_tokens": 100,
    }


def test_loop_detector_counts_same_action_on_same_page() -> None:
    detector = LoopDetector()
    action = tool("click", {"element_id": "f0:e-003"})
    assert detector.record(action, PAGE) == 1
    assert detector.record(action, PAGE) == 2
    assert detector.record(action, PAGE) == 3


def test_loop_detector_distinguishes_arguments_and_page() -> None:
    detector = LoopDetector()
    first = tool("type_text", {"element_id": "f0:e-002", "text": "a"})
    second = tool("type_text", {"element_id": "f0:e-002", "text": "b"})
    changed = PAGE.model_copy(update={"visible_text": "changed"})
    assert detector.record(first, PAGE) == 1
    assert detector.record(second, PAGE) == 1
    assert detector.record(first, changed) == 1


def test_same_page_snapshot_is_compacted_to_a_diff() -> None:
    previous = PAGE.model_copy(
        update={
            "visible_text": "\n".join(
                f"Stable region {index}" for index in range(40)
            )
        }
    )
    current = previous.model_copy(
        update={
            "snapshot_id": "s-next",
            "visible_text": previous.visible_text.replace(
                "Stable region 20", "Changed region 20"
            ),
        }
    )
    result = ActionResult(
        success=True,
        action="get_page_state",
        snapshot_id="s-next",
        data={"page_state": current.model_dump()},
    )
    compact = compact_result(result, previous)
    page_data = compact.data["page_state"]
    assert page_data["type"] == "page_state_diff"
    assert page_data["base_snapshot_id"] == previous.snapshot_id
    assert page_data["text"]["changed_regions"]


def test_navigation_keeps_a_full_snapshot() -> None:
    current = PAGE.model_copy(
        update={
            "snapshot_id": "s-results",
            "document_id": "d-results",
            "url": "http://127.0.0.1:8080/results.html",
        }
    )
    result = ActionResult(
        success=True,
        action="get_page_state",
        snapshot_id="s-results",
        data={"page_state": current.model_dump()},
    )
    preserved = compact_result(result, PAGE)
    assert preserved.data["page_state"]["type"] == "page_state_full"
    assert preserved.data["page_state"]["full_snapshot_reason"] == "document_changed"


async def test_default_run_has_no_token_cap_and_still_reports_usage():
    provider = MockProvider(scripted=[ProviderResponse(
        decision=FinalAnswer(answer="Completed after more than the old token cap."), raw_text="",
        usage={"prompt_tokens":100000, "completion_tokens":1000, "total_tokens":101000})])
    bridge = FakeBridge([ActionResult(success=True, action="get_page_state", snapshot_id=PAGE.snapshot_id,
        data={"page_state":PAGE.model_dump()})])
    result = await run_agent("Read this page", PAGE, ["http://127.0.0.1:*"], provider, AgentSettings(), bridge)
    assert result.finish_reason == "stop" and result.usage["total_tokens"] == 101000
    assert "Completed" in result.text


def test_unlimited_token_accounting_does_not_stop_after_repeated_usage():
    budget = TokenBudget(None)
    for _ in range(10):
        budget.record({"prompt_tokens":57000, "completion_tokens":611, "total_tokens":57611})
    assert not budget.exceeded and budget.usage["total_tokens"] == 576110
