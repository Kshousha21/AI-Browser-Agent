"""WebSocket authentication, origin checking, and one full turn.

The three gates from the spec are: loopback binding, origin allow-listing, and
the shared token. The first is a configuration property (covered in
test_config.py); origin is enforced on the handshake and the token must arrive
in the first WebSocket frame.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import browser.server as server_module
from browser.actions import ActionResult
from browser.config import Settings
from browser.protocol import PROTOCOL_REVISION, PageElement, PageState, UserMessagePayload, make_envelope
from browser.providers import MockProvider, ToolCall
from browser.server import create_app, origin_allowed, token_valid

from .conftest import TEST_TOKEN

EXTENSION_ORIGIN = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
PAGE = PageState(
    url="http://127.0.0.1:8080/",
    title="Test Page",
    visible_text="The handbook covers travel policy and expenses.",
)


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider(scripted=["This page describes the travel policy."])


@pytest.fixture
def client(settings: Settings, provider: MockProvider) -> Iterator[TestClient]:
    # TestClient only runs the app's lifespan when used as a context manager,
    # and the provider is wired up in lifespan.
    with TestClient(create_app(settings, provider=provider)) as test_client:
        yield test_client


def connect(client: TestClient, *, origin: str | None) -> Any:
    headers = {} if origin is None else {"origin": origin}
    return client.websocket_connect("/ws", headers=headers)


def send(ws: Any, type_: str, payload: Any) -> str:
    envelope = make_envelope(type_, payload)
    ws.send_text(envelope.model_dump_json())
    return envelope.id


def authenticate(ws: Any, token: str = TEST_TOKEN) -> None:
    send(ws, "auth", {"token": token})


def greet(ws: Any) -> dict[str, Any]:
    authenticate(ws)
    send(ws, "hello", {"client": "test", "protocol_version": 1, "protocol_revision": PROTOCOL_REVISION})
    return json.loads(ws.receive_text())


def answer_initial_page_read(ws: Any, frame: dict[str, Any]) -> None:
    assert frame["type"] == "tool_request"
    assert frame["payload"]["tool"] == "get_page_state"
    result = ActionResult(
        success=True,
        action="get_page_state",
        snapshot_id=PAGE.snapshot_id,
        data={"page_state": PAGE.model_dump()},
    )
    response = make_envelope(
        "tool_result",
        {"result": result.model_dump()},
        corr_id=frame["id"],
    )
    ws.send_text(response.model_dump_json())


# --------------------------------------------------------------------------
# Rejection
# --------------------------------------------------------------------------


def test_first_frame_must_be_auth(client: TestClient) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        send(ws, "hello", {"client": "test", "protocol_version": 1, "protocol_revision": PROTOCOL_REVISION})
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_text()

    assert excinfo.value.code == 1008


def test_connect_with_a_wrong_token_is_rejected(client: TestClient) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        authenticate(ws, "not-the-token-xxxxxxxx")
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_text()

    assert excinfo.value.code == 1008


def test_connect_with_an_empty_token_is_rejected(client: TestClient) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        authenticate(ws, "")
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_text()

    assert excinfo.value.code == 1008


def test_auth_must_arrive_within_two_seconds(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_module, "AUTH_TIMEOUT_SECONDS", 0.01)
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_text()

    assert excinfo.value.code == 1008
    assert excinfo.value.reason == "authentication timeout"


def test_query_string_token_does_not_authenticate(client: TestClient) -> None:
    with client.websocket_connect(
        f"/ws?token={TEST_TOKEN}", headers={"origin": EXTENSION_ORIGIN}
    ) as ws:
        send(ws, "hello", {"client": "test", "protocol_version": 1, "protocol_revision": PROTOCOL_REVISION})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


def test_connect_from_a_wrong_origin_is_rejected(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect):
        with connect(client, origin="https://evil.example"):
            pass


def test_connect_with_no_origin_header_is_rejected(client: TestClient) -> None:
    """A missing Origin is not a pass; the header is one of the three gates."""
    with pytest.raises(WebSocketDisconnect):
        with connect(client, origin=None):
            pass


def test_a_valid_token_from_a_bad_origin_still_fails(client: TestClient) -> None:
    """The gates are independent: passing one does not excuse the other."""
    with pytest.raises(WebSocketDisconnect):
        with connect(client, origin="http://localhost:3000"):
            pass


def test_rejection_happens_before_the_provider_is_touched(
    client: TestClient, provider: MockProvider
) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        authenticate(ws, "wrong-token-abcdefghij")
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()

    assert provider.calls == []


# --------------------------------------------------------------------------
# Acceptance
# --------------------------------------------------------------------------


def test_a_valid_handshake_is_accepted(client: TestClient) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        ack = greet(ws)

    assert ack["type"] == "hello_ack"
    assert ack["payload"]["provider"] == "mock"
    assert ack["payload"]["session_id"]


def test_an_explicitly_allowed_origin_is_accepted(provider: MockProvider) -> None:
    settings = Settings(
        auth={"token": TEST_TOKEN},
        provider={"name": "mock"},
        server={"allowed_origins": ["http://127.0.0.1:8080"]},
    )
    with TestClient(create_app(settings, provider=provider)) as client:
        with connect(client, origin="http://127.0.0.1:8080") as ws:
            assert greet(ws)["type"] == "hello_ack"


def test_extension_origins_can_be_disallowed(provider: MockProvider) -> None:
    settings = Settings(
        auth={"token": TEST_TOKEN},
        provider={"name": "mock"},
        server={"allow_any_extension_origin": False},
    )
    with TestClient(create_app(settings, provider=provider)) as client:
        with pytest.raises(WebSocketDisconnect):
            with connect(client, origin=EXTENSION_ORIGIN):
                pass


def test_a_bearer_header_does_not_replace_the_auth_frame(
    client: TestClient,
) -> None:
    with client.websocket_connect(
        "/ws",
        headers={"origin": EXTENSION_ORIGIN, "authorization": f"Bearer {TEST_TOKEN}"},
    ) as ws:
        send(ws, "hello", {"client": "test", "protocol_version": 1, "protocol_revision": PROTOCOL_REVISION})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_text()


# --------------------------------------------------------------------------
# The turn
# --------------------------------------------------------------------------


def test_one_turn_streams_deltas_then_done(
    client: TestClient, provider: MockProvider
) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        greet(ws)
        turn_id = send(
            ws,
            "user_message",
            UserMessagePayload(text="Summarize this page", page_state=PAGE),
        )

        deltas: list[str] = []
        while True:
            frame = json.loads(ws.receive_text())
            if frame["type"] == "assistant_done":
                done = frame
                break
            if frame["type"] in {"run_status", "timeline_step"}:
                continue
            if frame["type"] == "tool_request":
                answer_initial_page_read(ws, frame)
                continue
            assert frame["type"] == "assistant_delta"
            assert frame["corr_id"] == turn_id
            deltas.append(frame["payload"]["text"])

    assert deltas
    assert done["corr_id"] == turn_id
    assert done["payload"]["text"] == "This page describes the travel policy."
    assert "".join(deltas).strip() == done["payload"]["text"]


def test_the_page_snapshot_reaches_the_model(
    client: TestClient, provider: MockProvider
) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        greet(ws)
        send(
            ws,
            "user_message",
            UserMessagePayload(text="Summarize this page", page_state=PAGE),
        )
        while True:
            frame = json.loads(ws.receive_text())
            if frame["type"] == "tool_request":
                answer_initial_page_read(ws, frame)
            elif frame["type"] == "assistant_done":
                break

    prompt = provider.calls[0][-1].content
    assert "The handbook covers travel policy and expenses." in prompt
    assert "http://127.0.0.1:8080/" in prompt


def test_high_risk_confirmation_cancel_returns_control_without_dispatch(
    tmp_path,
) -> None:
    submit_page = PAGE.model_copy(
        update={
            "elements": [
                PageElement(
                    agent_id="f0:e-900",
                    tag="button",
                    role="button",
                    name="Delete request",
                    input_type="submit",
                )
            ]
        }
    )
    provider = MockProvider(
        scripted=[
            ToolCall(
                tool="click",
                arguments={"element_id": "f0:e-900"},
                snapshot_id=submit_page.snapshot_id,
                reason="Submit the access request.",
            ),
            "The request was not submitted because you cancelled it.",
        ]
    )
    audit_path = tmp_path / "actions.jsonl"
    settings = Settings(
        auth={"token": TEST_TOKEN},
        provider={"name": "mock"},
        security={"audit_log_path": audit_path},
    )
    app = create_app(settings, provider=provider)
    seen_tool_requests: list[str] = []
    confirmation = None
    with TestClient(app) as test_client:
        with connect(test_client, origin=EXTENSION_ORIGIN) as ws:
            greet(ws)
            send(
                ws,
                "user_message",
                UserMessagePayload(
                    text="Submit the request",
                    page_state=submit_page,
                    read_only=False,
                ),
            )
            while True:
                frame = json.loads(ws.receive_text())
                if frame["type"] == "tool_request":
                    seen_tool_requests.append(frame["payload"]["tool"])
                    result = ActionResult(
                        success=True,
                        action="get_page_state",
                        snapshot_id=submit_page.snapshot_id,
                        data={"page_state": submit_page.model_dump()},
                    )
                    ws.send_text(
                        make_envelope(
                            "tool_result",
                            {"result": result.model_dump()},
                            corr_id=frame["id"],
                        ).model_dump_json()
                    )
                elif frame["type"] == "confirmation_request":
                    confirmation = frame["payload"]
                    ws.send_text(
                        make_envelope(
                            "confirmation_response",
                            {"decision": "cancel"},
                            corr_id=frame["id"],
                        ).model_dump_json()
                    )
                elif frame["type"] == "assistant_done":
                    done = frame
                    break

    assert seen_tool_requests == ["get_page_state"]
    assert confirmation is not None
    assert confirmation["tool"] == "click"
    assert confirmation["target"] == "Delete request"
    assert "submit" in confirmation["risk_reasons"]
    assert "cancelled" in done["payload"]["text"]
    audit = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["tool"] for item in audit] == ["get_page_state", "click"]
    assert audit[-1]["confirmation"] == "cancel"


def test_user_message_before_hello_is_a_protocol_error(client: TestClient) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        authenticate(ws)
        send(ws, "user_message", UserMessagePayload(text="hi", page_state=PAGE))
        frame = json.loads(ws.receive_text())

    assert frame["type"] == "error"
    assert frame["payload"]["code"] == "protocol_error"


def test_garbage_frames_get_an_error_and_the_socket_survives(
    client: TestClient,
) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        greet(ws)
        ws.send_text("this is not json")
        frame = json.loads(ws.receive_text())
        assert frame["payload"]["code"] == "protocol_error"

        # Still usable afterwards.
        ping_id = send(ws, "ping", {})
        pong = json.loads(ws.receive_text())
        assert pong["type"] == "pong"
        assert pong["corr_id"] == ping_id


def test_a_malformed_payload_is_a_bad_request(client: TestClient) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        greet(ws)
        send(ws, "user_message", {"text": "hi"})  # page_state missing
        frame = json.loads(ws.receive_text())

    assert frame["type"] == "error"
    assert frame["payload"]["code"] == "bad_request"


def test_an_unknown_message_type_is_reported(client: TestClient) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        greet(ws)
        send(ws, "teleport", {})
        frame = json.loads(ws.receive_text())

    assert frame["payload"]["code"] == "protocol_error"


def test_ping_gets_a_pong(client: TestClient) -> None:
    with connect(client, origin=EXTENSION_ORIGIN) as ws:
        greet(ws)
        turn_id = send(ws, "ping", {})
        frame = json.loads(ws.receive_text())

    assert frame["type"] == "pong"
    assert frame["corr_id"] == turn_id


def test_a_provider_failure_becomes_an_error_frame(settings: Settings) -> None:
    app = create_app(settings, provider=MockProvider(fail_with="model unavailable"))

    with TestClient(app) as client:
        with connect(client, origin=EXTENSION_ORIGIN) as ws:
            greet(ws)
            send(ws, "user_message", UserMessagePayload(text="hi", page_state=PAGE))
            while True:
                frame = json.loads(ws.receive_text())
                if frame["type"] == "tool_request":
                    answer_initial_page_read(ws, frame)
                    continue
                if frame["type"] not in {"run_status", "timeline_step"}:
                    break

    assert frame["type"] == "error"
    assert frame["payload"]["code"] == "provider_error"
    assert "model unavailable" in frame["payload"]["message"]


# --------------------------------------------------------------------------
# /health and the gate helpers
# --------------------------------------------------------------------------


def test_health_reports_the_active_provider(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["protocol_version"] == 1
    assert body["provider"] == "mock"


def test_health_needs_no_token(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    "origin,expected",
    [
        (EXTENSION_ORIGIN, True),
        ("chrome-extension://another", True),
        ("https://evil.example", False),
        ("", False),
        (None, False),
    ],
)
def test_origin_allowed(origin: str | None, expected: bool, settings: Settings) -> None:
    assert origin_allowed(origin, settings) is expected


@pytest.mark.parametrize(
    "token,expected",
    [(TEST_TOKEN, True), ("wrong", False), ("", False), (None, False)],
)
def test_token_valid(token: str | None, expected: bool, settings: Settings) -> None:
    assert token_valid(token, settings) is expected


def test_no_token_configured_means_nothing_validates() -> None:
    """A blank configured token must not turn into an open door."""
    settings = Settings(auth={"token": ""}, provider={"name": "mock"})

    assert token_valid("", settings) is False
    assert token_valid("anything", settings) is False
