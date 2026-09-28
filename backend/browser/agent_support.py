"""Small state and message helpers for the bounded agent loop."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from .actions import ActionResult
from .prompting import render_tool_result
from .protocol import PageState
from .providers import Message, ProviderResponse, ToolCall
from .security import ConfirmationChoice, RiskAssessment


@dataclass
class TokenBudget:
    limit: int | None
    used: int = 0
    usage: dict[str, int] = field(default_factory=dict)

    def record(self, usage: dict[str, int]) -> None:
        for key in ("prompt_tokens", "completion_tokens"):
            self.usage[key] = self.usage.get(key, 0) + max(0, usage.get(key, 0))
        charged = usage.get("total_tokens")
        if charged is None:
            charged = usage.get("prompt_tokens", 0) + usage.get(
                "completion_tokens", 0
            )
        self.used += max(0, charged)
        self.usage["total_tokens"] = self.used

    @property
    def exceeded(self) -> bool:
        return self.limit is not None and self.used >= self.limit


class LoopDetector:
    """Count repeated (tool, args, page signature) decisions."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._semantic_counts: dict[str, int] = {}

    def fingerprint(self, call: ToolCall, page: PageState) -> str:
        normalized = json.dumps(call.arguments, sort_keys=True, separators=(",", ":"))
        raw = f"{call.tool}\0{normalized}\0{page_signature(page)}".encode()
        return hashlib.sha256(raw).hexdigest()

    def semantic_fingerprint(self, call: ToolCall) -> str:
        normalized = json.dumps(call.arguments, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{call.tool}\0{normalized}".encode()).hexdigest()

    def record(
        self,
        call: ToolCall,
        page: PageState,
        *,
        equivalent_state: bool = False,
    ) -> int:
        key = self.fingerprint(call, page)
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        semantic_key = self.semantic_fingerprint(call)
        semantic_count = self._semantic_counts.get(semantic_key, 0) + 1
        self._semantic_counts[semantic_key] = semantic_count
        return max(count, semantic_count) if equivalent_state else count

    def discard_rejected_attempt(self, fingerprint: str, semantic_fingerprint: str) -> None:
        """Undo one pre-execution rejection, keeping every other recorded attempt."""
        for counts, key in ((self._counts, fingerprint), (self._semantic_counts, semantic_fingerprint)):
            count = counts.get(key, 0)
            if count > 1:
                counts[key] = count - 1
            else:
                counts.pop(key, None)


class AgentBridge(Protocol):
    @property
    def stopped(self) -> bool: ...

    async def execute(self, call: ToolCall, step: int, max_steps: int) -> ActionResult: ...

    async def status(
        self,
        status: str,
        step: int,
        max_steps: int,
        elapsed_ms: int,
        usage: dict[str, int],
    ) -> None: ...

    async def timeline(
        self, call: ToolCall, result: ActionResult, step: int
    ) -> None: ...

    async def confirm(
        self, call: ToolCall, assessment: RiskAssessment, step: int
    ) -> ConfirmationChoice: ...

    async def audit(
        self,
        call: ToolCall,
        result: ActionResult,
        confirmation: str,
        password_target: bool = False,
    ) -> None: ...


def tool_exchange(
    response: ProviderResponse, call: ToolCall, result: ActionResult
) -> list[Message]:
    call_id = response.tool_call_id or f"call-{uuid.uuid4()}"
    provider_args = {
        **call.arguments,
        "snapshot_id": call.snapshot_id,
        "reason": call.reason,
    }
    return [
        Message(
            role="assistant",
            content=response.raw_text,
            tool_call_id=call_id,
            tool_name=call.tool,
            tool_arguments=provider_args,
        ),
        Message(
            role="tool",
            content=render_tool_result(result),
            tool_call_id=call_id,
        ),
    ]


def validation_result(
    call: ToolCall, page: PageState, error: Exception
) -> ActionResult:
    return ActionResult(
        success=False,
        action=call.tool,
        snapshot_id=page.snapshot_id,
        error="invalid_arguments",
        message=f"Tool schema validation failed: {error}",
    )


def trim_messages(messages: list[Message], history_steps: int) -> list[Message]:
    fixed = messages[:3]
    tail = messages[3:][-(history_steps * 2) :]
    return [*fixed, *tail]


def retire_superseded_page_context(messages: list[Message]) -> None:
    """Remove page bodies superseded by a later full navigation snapshot."""
    retire_bounded_state_reads(messages)
    marker = (
        "[RETIRED PAGE CONTEXT]\n"
        "A later full navigation snapshot supersedes this page observation."
    )
    for index, message in enumerate(messages):
        is_initial_page = index == 2
        is_prior_full_result = (
            message.role == "tool"
            and isinstance(message.content, str)
            and '"type":"page_state_' in message.content
        )
        if not is_initial_page and not is_prior_full_result:
            continue
        messages[index] = Message(
            role=message.role,
            content=marker,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            tool_arguments=message.tool_arguments,
        )


def retire_bounded_state_reads(messages: list[Message]) -> None:
    """Keep native call/result pairing, retaining only the latest bounded read."""
    for index, message in enumerate(messages):
        if (message.role == "tool" and isinstance(message.content, str)
                and '"context_kind":"bounded_state_read"' in message.content):
            messages[index] = Message(
                role=message.role,
                content="[RETIRED STATE READ] Superseded bounded metadata; use current state only.",
                tool_call_id=message.tool_call_id,
                tool_name=message.tool_name,
                tool_arguments=message.tool_arguments,
            )


def page_signature(page: PageState) -> str:
    raw = json.dumps(
        {
            "tab_id": page.tab_id,
            "document_id": page.document_id,
            "url": page.url,
            "title": page.title,
            "text": page.visible_text[:4_000],
            "elements": [
                (
                    item.agent_id,
                    item.role,
                    item.name,
                    item.tag,
                    item.href,
                    item.form_action,
                    item.form_method,
                    item.form_target,
                    item.enabled,
                    item.visible,
                    item.occluded,
                )
                for item in page.elements
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def action_status(call: ToolCall) -> str:
    target = str(call.arguments.get("element_id", ""))
    return f"Step: {call.tool}" + (f" {target}" if target else "")
