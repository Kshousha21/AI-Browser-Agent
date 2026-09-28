"""Existing actual SDKs against synthetic loopback only; no paid calls."""
import base64
import json
import pytest
from browser.providers.base import Message, ProviderError
from browser.providers.images import image_data, translate
from .provider_wire import CASES, answer, make_provider, running_server, protocol
from .test_pdf_visual import prepared


@pytest.fixture(scope="module")
def wire():
    with running_server() as server:
        yield server


@pytest.mark.parametrize("case", CASES)
async def test_inline_png_wire_formats_keep_model_and_zero_tools(tmp_path, wire, case):
    wire.reset()
    renderer, doc, _ = prepared(tmp_path)
    image = await renderer.render(doc.document_id, 4, [doc.document_id])
    encoded = base64.b64encode(image.png).decode("ascii")
    messages = [Message(role="system", content="No tools. Interpret only supplied synthetic data."),
        Message(role="user", content=[{"type": "text", "text": "S1: synthetic chart page"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}}])]
    provider, settings = make_provider(case, wire, supports_images=True)
    wire.replies.append(answer(case, text="Synthetic visual result."))
    try:
        response = await provider.generate(messages, tools=[])
    finally:
        await provider.aclose()
    assert response.decision.answer == "Synthetic visual result." and len(wire.received) == 1
    body = wire.received[0]["body"]
    assert not body.get("tools") and body["model"] == settings.model
    turns = body.get("messages", body.get("input"))
    user = next(turn for turn in turns if turn["role"] == "user")
    kind = protocol(case)
    if kind == "ollama":
        assert user["images"] == [encoded]
    elif kind == "anthropic":
        assert user["content"][1] == messages[1].content[1]
    elif kind == "responses":
        assert user["content"][1]["type"] == "input_image"
        assert user["content"][1]["image_url"] == "data:image/png;base64," + encoded
        assert body["store"] is False
    else:
        assert user["content"][1]["image_url"]["url"] == "data:image/png;base64," + encoded
    assert "filename" not in json.dumps(body) and str(tmp_path) not in json.dumps(body)


@pytest.mark.parametrize("source", [
    {"type": "url", "url": "https://unauthorized.invalid/file.png"},
    {"type": "file", "file_id": "remote-file"},
    {"type": "base64", "media_type": "image/png", "data": "not-base64"},
    {"type": "base64", "media_type": "application/pdf", "data": "abcd"},
])
def test_only_bounded_inline_png_no_remote_fetch(source):
    with pytest.raises(ProviderError, match="invalid bounded inline"):
        image_data({"type": "image", "source": source})


async def test_image_count_and_role_bounds_before_wire(tmp_path):
    renderer, doc, _ = prepared(tmp_path)
    result = await renderer.render(doc.document_id, 4, [doc.document_id])
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
        "data": base64.b64encode(result.png).decode("ascii")}}
    for role, blocks in [("user", [block] * 6), ("assistant", [block]), ("system", [block])]:
        with pytest.raises(ProviderError):
            translate(blocks, "anthropic", role)
