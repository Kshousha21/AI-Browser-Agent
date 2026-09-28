"""Non-sensitive application and explicit wire-compatibility identifiers."""
import json
from pathlib import Path

CONTRACT = json.loads((Path(__file__).resolve().parents[2] / "shared/build-contract.json").read_text(encoding="utf-8"))
APPLICATION_VERSION = CONTRACT["application_version"]
PROTOCOL_REVISION = CONTRACT["protocol_revision"]
PROTOCOL_MISMATCH = "Backend/extension protocol mismatch. Rebuild and reload extension/dist, then restart backend."


def compatible_hello(payload: object) -> bool:
    return (isinstance(payload, dict)
            and type(payload.get("protocol_version")) is int
            and payload["protocol_version"] == CONTRACT["protocol_version"]
            and type(payload.get("protocol_revision")) is int
            and payload["protocol_revision"] == PROTOCOL_REVISION)
