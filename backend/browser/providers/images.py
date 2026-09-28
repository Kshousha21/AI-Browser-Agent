"""Provider-neutral inline PNG translation; never a remote image URL/file ID."""
import base64
import binascii
import struct
from .base import ProviderError


def image_data(block):
    try:
        if set(block) != {"type", "source"} or block["type"] != "image":
            raise ValueError()
        source = block["source"]
        if (set(source) != {"type", "media_type", "data"} or source["type"] != "base64"
                or source["media_type"] != "image/png" or not isinstance(source["data"], str)
                or len(source["data"]) > 5592408):
            raise ValueError()
        raw = base64.b64decode(source["data"], validate=True)
        if raw[:8] != b"\x89PNG\r\n\x1a\n" or len(raw) < 33 or len(raw) > 4 * 1024 * 1024:
            raise ValueError()
        width, height = struct.unpack(">II", raw[16:24])
        if min(width, height) < 1 or max(width, height) > 2000 or width * height > 4000000:
            raise ValueError()
        return source["data"]
    except (KeyError, TypeError, ValueError, binascii.Error):
        raise ProviderError("invalid bounded inline page image", code="invalid_request") from None


def translate(content, family, role):
    if not isinstance(content, list):
        return content
    result = []
    image_count = 0
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            result.append({"type": "input_text" if family == "responses" else "text", "text": block["text"]})
            continue
        if role != "user":
            raise ProviderError("page images require a user evidence message", code="invalid_request")
        encoded = image_data(block)
        image_count += 1
        if image_count > 5:
            raise ProviderError("too many page images", code="invalid_request")
        if family == "anthropic":
            result.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}})
        elif family == "responses":
            result.append({"type": "input_image", "image_url": "data:image/png;base64," + encoded, "detail": "high"})
        else:
            result.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded, "detail": "high"}})
    return result
