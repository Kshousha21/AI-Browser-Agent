"""Wire protocol shared by the backend and the extension.

Mirror of shared/protocol.ts. The two files are kept in sync by hand; the field
names here are the contract.

Every frame on the socket is an Envelope. Nothing is sent bare.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .actions import ActionResult
from .build_info import CONTRACT, APPLICATION_VERSION, PROTOCOL_REVISION

PROTOCOL_VERSION = CONTRACT["protocol_version"]

ClientMessageType = Literal[
    "auth",
    "hello",
    "user_message",
    "document_prepare",
    "document_question",
    "document_refresh",
    "document_file_action",
    "document_research",
    "document_context_clear",
    "tool_result",
    "confirmation_response",
    "stop",
    "pause",
    "resume",
    "session_event",
    "ping",
]
ServerMessageType = Literal[
    "hello_ack",
    "assistant_delta",
    "assistant_done",
    "tool_request",
    "confirmation_request",
    "run_status",
    "control_state",
    "timeline_step",
    "document_choices",
    "document_progress",
    "document_ready",
    "document_done",
    "research_plan",
    "research_file",
    "routing_status",
    "error",
    "pong",
]

ErrorCode = Literal[
    "unauthorized",
    "bad_request",
    "protocol_error",
    "protocol_mismatch",
    "provider_error",
    "internal_error",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Envelope(BaseModel):
    """The envelope.

    ``id`` is unique per frame. ``corr_id`` points at the ``id`` of the frame
    that caused this one, so a stream of deltas can be tied to the request that
    produced it.
    """

    v: int = PROTOCOL_VERSION
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    ts: str = Field(default_factory=_now)
    corr_id: str | None = None
    payload: Any = None


# --------------------------------------------------------------------------
# Page state
# --------------------------------------------------------------------------


class PageElement(BaseModel):
    agent_id: str
    tag: str
    role: str
    name: str = ""
    text: str = ""
    placeholder: str = ""
    value: str = ""
    input_type: str = ""
    href: str = ""
    form_action: str = ""
    form_method: str = ""
    form_target: str = ""
    dom_id: str = ""
    test_id: str = ""
    data_key: str = ""
    semantic_region: str = ""
    selected: bool = False
    enabled: bool = True
    visible: bool = True
    occluded: bool = False
    in_viewport: bool = False
    bbox: list[float] = Field(default_factory=list)
    document_order: int = 0


class PageHeading(BaseModel):
    level: int
    text: str


class PageFrame(BaseModel):
    frame_id: str = "f0"
    url: str
    origin: str


class PageDialog(BaseModel):
    type: Literal["alert", "confirm", "prompt"]
    message: str
    dismissed: bool = True


class PageRegion(BaseModel):
    """Generic DOM structure retained for site-adapter normalization.

    Regions are backend-internal evidence. Raw region text is not rendered
    directly into model context; an enabled adapter may normalize it.
    """

    region_id: str
    tag: str
    role: str = ""
    name: str = ""
    text: str = ""
    dom_id: str = ""
    test_id: str = ""
    data_key: str = ""
    class_tokens: list[str] = Field(default_factory=list)
    href: str = ""
    visible: bool = True
    ancestor_region_ids: list[str] = Field(default_factory=list)
    agent_element_ids: list[str] = Field(default_factory=list)


class Truncation(BaseModel):
    text: bool = False
    elements: bool = False


class PageState(BaseModel):
    tab_id: int | None = Field(default=None, ge=0)
    snapshot_id: str = "s-legacy"
    document_id: str = ""
    navigation_id: int = 0
    url: str
    title: str = ""
    frames: list[PageFrame] = Field(default_factory=list)
    visible_text: str = ""
    headings: list[PageHeading] = Field(default_factory=list)
    elements: list[PageElement] = Field(default_factory=list)
    truncated: Truncation = Field(default_factory=Truncation)
    opaque_regions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    dialogs: list[PageDialog] = Field(default_factory=list)
    looks_like_login: bool = False
    semantic_regions: list[PageRegion] = Field(default_factory=list)
    # Site-specific, bounded DOM facts for adapter normalization. This internal
    # evidence is never rendered directly into model context.
    adapter_payload: dict[str, Any] | None = None
    site_context: dict[str, Any] | None = None

    @field_validator("truncated", mode="before")
    @classmethod
    def accept_phase_zero_truncation(cls, value: Any) -> Any:
        if isinstance(value, bool):
            return {"text": value, "elements": False}
        return value


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------


class AuthPayload(BaseModel):
    token: str


class HelloPayload(BaseModel):
    client: str = "unknown"
    protocol_version: int = PROTOCOL_VERSION
    protocol_revision: int = Field(strict=True)


class HelloAckPayload(BaseModel):
    protocol_revision: int = PROTOCOL_REVISION
    backend_version: str = APPLICATION_VERSION
    server: str
    protocol_version: int = PROTOCOL_VERSION
    session_id: str
    provider: str
    model: str
    research_max_pdfs: int = 3
    research_max_visual_pages: int = 3


class UserMessagePayload(BaseModel):
    text: str
    page_state: PageState
    allowlist: list[str] = Field(default_factory=list)
    read_only: bool = Field(default=False, exclude=True)  # Legacy clients only; no UI mode.
    allow_remote_context: bool = False


class AssistantDeltaPayload(BaseModel):
    text: str


class AssistantDonePayload(BaseModel):
    text: str
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)
    elapsed_ms: int = 0


class ToolRequestPayload(BaseModel):
    tab_id: int | None = Field(default=None, ge=0)
    document_id: str = ""
    control_epoch: int = 0
    reset_target: bool = False
    tab_revision: str | None = None
    run_id: str
    step: int
    max_steps: int
    tool: str
    arguments: dict[str, Any]
    snapshot_id: str
    reason: str


class ToolResultPayload(BaseModel):
    result: ActionResult


class ConfirmationRequestPayload(BaseModel):
    run_id: str
    step: int
    tool: str
    target: str = ""
    arguments: dict[str, Any]
    reason: str
    risk: Literal["high"]
    risk_reasons: list[str] = Field(default_factory=list)
    scope: str


class ConfirmationResponsePayload(BaseModel):
    decision: Literal["allow_once", "allow_run", "cancel"]


class SessionControlPayload(BaseModel):
    run_id: str | None = None
    event: Literal["pause", "resume", "active_target_changed", "target_tab_closed", "target_document_changed", "side_panel_closed"] = "pause"


class ControlStatePayload(BaseModel):
    run_id: str
    state: Literal["running", "paused", "resuming", "waiting", "stopped"]
    epoch: int
    reason: str


class StopPayload(BaseModel):
    run_id: str | None = None


class RunStatusPayload(BaseModel):
    run_id: str
    status: str
    step: int = 0
    max_steps: int
    elapsed_ms: int
    usage: dict[str, int] = Field(default_factory=dict)


class TimelineStepPayload(BaseModel):
    run_id: str
    step: int
    tool: str
    target: str = ""
    reason: str
    result: ActionResult


class ErrorPayload(BaseModel):
    code: str
    message: str


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_envelope(
    type_: str, payload: BaseModel | dict[str, Any], corr_id: str | None = None
) -> Envelope:
    body = payload.model_dump() if isinstance(payload, BaseModel) else payload
    return Envelope(type=type_, payload=body, corr_id=corr_id)


def error_envelope(
    code: str, message: str, corr_id: str | None = None
) -> Envelope:
    return make_envelope("error", ErrorPayload(code=code, message=message), corr_id)
