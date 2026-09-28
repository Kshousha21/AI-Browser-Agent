"""Bounded, backend-only preservation of signed Anthropic assistant blocks.

Opaque local handles prevent a compatible endpoint reusing a native tool ID
from replacing another session's continuation. Never expose thinking/signatures
in ProviderResponse text, logs, UI, or persistent storage. Eviction fails closed.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from copy import deepcopy
from uuid import uuid4

from .base import Message, ProviderError

PREFIX = "anthropic-cont-"


class Continuations:
    def __init__(self, *, max_entries=256, max_bytes=8 * 1024 * 1024):
        self._entries = OrderedDict()
        self._bytes = 0
        self.max_entries = max_entries
        self.max_bytes = max_bytes

    def remember(self, blocks, native_id: str) -> str:
        # Tool-only legacy responses need no opaque vendor context.
        if len(blocks) == 1 and blocks[0].type == "tool_use":
            return native_id
        content = [block.model_dump(mode="json", exclude_none=True) for block in blocks]
        size = len(json.dumps(content, ensure_ascii=True).encode("utf-8"))
        if size > min(self.max_bytes, 1024 * 1024) or self.max_entries < 1:
            raise ProviderError("Anthropic continuation exceeds local memory limit", code="continuation_limit")
        while self._entries and (len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes):
            _, (_, old_size) = self._entries.popitem(last=False)
            self._bytes -= old_size
        handle = PREFIX + uuid4().hex
        self._entries[handle] = (content, size)
        self._bytes += size
        return handle

    def restore(self, messages: list[Message], turns: list[dict]) -> None:
        pending = None
        for message, turn in zip((m for m in messages if m.role != "system"), turns, strict=True):
            handle = message.tool_call_id or ""
            if pending is not None and not (message.role == "tool" and handle == pending):
                raise ProviderError("Anthropic continuation requires its paired tool result", code="invalid_continuation")
            if not handle.startswith(PREFIX):
                pending = None
                continue
            entry = self._entries.get(handle)
            if entry is None:
                raise ProviderError("Anthropic continuation is unavailable; start a fresh run", code="continuation_missing")
            blocks = entry[0]
            call = next(block for block in blocks if block["type"] == "tool_use")
            if message.role == "assistant":
                if message.tool_name != call["name"] or message.tool_arguments != call["input"]:
                    raise ProviderError("Anthropic continuation does not match the tool call", code="invalid_continuation")
                turn["content"] = deepcopy(blocks)
                pending = handle
            elif message.role == "tool" and pending == handle:
                turn["content"][0]["tool_use_id"] = call["id"]
                pending = None
            else:
                raise ProviderError("Anthropic continuation requires its paired tool result", code="invalid_continuation")
        if pending is not None:
            raise ProviderError("Anthropic continuation requires its paired tool result", code="invalid_continuation")

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0
