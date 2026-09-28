"""Explicit public model acquisition only; never imported by retrieval execution."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import build_opener, HTTPRedirectHandler, Request
from urllib.parse import urlsplit

from .errors import RetrievalError
from .local_model import DEFAULT_MODEL_DIRECTORY, manifest, model_identity, verified_files
from .source import local_path
from .artifacts import artifact_directory, artifact_manifest, CPU_KIND


class _HttpsRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        if urlsplit(new_url).scheme != "https":
            raise RetrievalError("model_prepare_failed")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def prepare_model(directory: Path | str | None = None, *, artifact_kind: str = CPU_KIND) -> dict[str, object]:
    """Download a fixed public revision over normal verified HTTPS, then publish.

    No auth token, cookie, query or document is sent. Existing directories are
    verified, never overwritten. Interrupted downloads cannot become a model.
    """
    target = local_path(directory if directory is not None else artifact_directory(artifact_kind))
    info = artifact_manifest(artifact_kind)
    try:
        if target.exists():
            verified_files(target, artifact_kind)
            return {"status": "already_prepared", "artifact_kind": artifact_kind}
        target.parent.mkdir(parents=True, exist_ok=True)
        opener = build_opener(_HttpsRedirect())
        with TemporaryDirectory(prefix="prepare-model-", dir=target.parent) as temporary:
            stage = Path(temporary).resolve()
            assert stage.is_relative_to(target.parent.resolve()) and target != stage
            for spec in info["files"]:
                url = f"https://huggingface.co/{info['model_id']}/resolve/{info['revision']}/{spec['remote']}"
                request = Request(url, headers={"User-Agent": "local-retrieval-explicit-model-setup/1"})
                digest, count = hashlib.sha256(), 0
                with opener.open(request, timeout=30) as response, (stage / spec["local"]).open("xb") as output:
                    while block := response.read(1024 * 1024):
                        count += len(block)
                        if count > spec["bytes"]:
                            raise RetrievalError("model_prepare_failed")
                        digest.update(block)
                        output.write(block)
                if count != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
                    raise RetrievalError("model_prepare_failed")
            verified_files(stage, artifact_kind)
            stage.rename(target)  # Both resolved paths checked above; target must be new.
        return {"status": "prepared", "artifact_kind": artifact_kind,
                "artifact_sha256": info["files"][0]["sha256"],
                "license": info["license"], "artifact_bytes": sum(file["bytes"] for file in info["files"])}
    except RetrievalError:
        raise
    except Exception:
        raise RetrievalError("model_prepare_failed") from None
