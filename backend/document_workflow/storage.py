"""Stage only completed nonce-named browser downloads into owned local storage."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import stat
import hashlib
import tempfile
from .models import DocumentError, DownloadReceipt, MAX_BYTES
from local_storage import StorageError, data_root


def plain_path(path: Path) -> Path:
    """Reject UNC, symlinks, junctions/reparse points and non-absolute sources."""
    if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
        raise DocumentError("invalid_download_path")
    try:
        for part in [path, *path.parents]:
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise DocumentError("invalid_download_path")
        return path.resolve(strict=True)
    except OSError:
        raise DocumentError("invalid_download_path") from None


class AcquisitionStorage:
    def __init__(self, root: Path | None = None):
        try:
            self.root = Path(root).absolute() if root is not None else data_root() / "acquired"
        except StorageError:
            raise DocumentError("invalid_download_path") from None

    def directory(self):
        # Verify the existing ancestor before creating any owned directories.
        ancestor = self.root
        while not ancestor.exists() and not ancestor.is_symlink():
            ancestor = ancestor.parent
        plain_path(ancestor)
        self.root.mkdir(parents=True, exist_ok=True)
        return plain_path(self.root)

    def prepared_path(self, document_id: str):
        if not re.fullmatch(r"[a-f0-9]{64}", document_id):
            raise DocumentError("invalid_download_path")
        target = self.root / "prepared" / (document_id + ".pdf")
        return plain_path(target) if target.exists() else None

    def retain_pdf(self, source: Path, document_id: str):
        """Retain a validated content-hash copy; never persist browser absolute paths."""
        if not re.fullmatch(r"[a-f0-9]{64}", document_id):
            raise DocumentError("invalid_pdf")
        root = self.directory()
        folder = root / "prepared"
        if folder.exists() or folder.is_symlink():
            plain_path(folder)
        folder.mkdir(exist_ok=True)
        folder = plain_path(folder)
        target = folder / (document_id + ".pdf")
        with plain_path(source).open("rb") as incoming:
            info = os.fstat(incoming.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_BYTES:
                raise DocumentError("invalid_pdf")
            data = incoming.read(MAX_BYTES + 1)
        if not 0 < len(data) <= MAX_BYTES or hashlib.sha256(data).hexdigest() != document_id:
            raise DocumentError("invalid_pdf")
        if target.exists():
            existing = plain_path(target)
            if existing.stat().st_size > MAX_BYTES or hashlib.sha256(existing.read_bytes()).hexdigest() != document_id:
                raise DocumentError("invalid_pdf")
            return
        fd, name = tempfile.mkstemp(prefix="render-source-", suffix=".tmp", dir=folder)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Content-addressed and validated; never overwrite a different document.
            if target.exists():
                raise DocumentError("invalid_pdf")
            os.rename(temporary, target)
        except OSError:
            raise DocumentError("download_failed") from None
        finally:
            if temporary.exists() and temporary.parent == folder and not temporary.is_symlink():
                temporary.unlink()

    @contextmanager
    def stage(self, receipt: DownloadReceipt, acquisition_id: str):
        if receipt.acquisition_id != acquisition_id:
            raise DocumentError("download_failed")
        source = Path(receipt.filename)
        expected = "acq-" + acquisition_id + ".pdf"
        if source.name != expected or source.parent.name != "AIBrowserAgent":
            raise DocumentError("invalid_download_path")
        source = plain_path(source)
        root = self.directory()
        destination = root / expected
        if destination.exists() or not destination.resolve().is_relative_to(root):
            raise DocumentError("invalid_download_path")
        created = False
        try:
            with source.open("rb") as incoming:
                info = os.fstat(incoming.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise DocumentError("invalid_download_path")
                if info.st_size > MAX_BYTES:
                    raise DocumentError("download_too_large")
                if info.st_size != receipt.size or info.st_size == 0:
                    raise DocumentError("download_incomplete")
                with destination.open("xb") as outgoing:
                    created = True
                    remaining = receipt.size
                    while remaining:
                        block = incoming.read(min(1024 * 1024, remaining))
                        if not block:
                            raise DocumentError("download_incomplete")
                        outgoing.write(block)
                        remaining -= len(block)
                    if incoming.read(1) or os.fstat(incoming.fileno()).st_mtime_ns != info.st_mtime_ns:
                        raise DocumentError("download_incomplete")
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
            yield destination
        except OSError:
            raise DocumentError("download_failed") from None
        finally:
            # Never delete the original browser/user download. Only this exact
            # newly created file under our verified directory can be removed.
            if created and destination.parent.resolve() == root and destination.is_file() and not destination.is_symlink():
                destination.unlink()

    def record(self, acquisition_id: str, document_id: str, source_fingerprint: str, ingestion: str, indexing: str):
        if (not re.fullmatch(r"[a-f0-9]{32}", acquisition_id)
            or not re.fullmatch(r"[a-f0-9]{64}", document_id)
            or not re.fullmatch(r"[a-f0-9]{64}", source_fingerprint)
            or ingestion not in {"ingested", "already_ingested"}
            or indexing not in {"pending", "ready", "failed"}):
            raise DocumentError("provenance_failed")
        root = self.directory()
        folder = root / "provenance"
        if folder.exists() or folder.is_symlink():
            plain_path(folder)
        folder.mkdir(exist_ok=True)
        plain_path(folder)
        target = folder / (acquisition_id + ".json")
        temporary = folder / (acquisition_id + ".tmp")
        data = {"version": 1, "source_type": "web_attachment", "document_id": document_id,
                "source_fingerprint": source_fingerprint, "acquired_at": datetime.now(timezone.utc).isoformat(),
                "ingestion_status": ingestion, "index_status": indexing}
        created = False
        try:
            with temporary.open("x", encoding="utf-8") as output:
                created = True
                json.dump(data, output, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        except OSError:
            raise DocumentError("provenance_failed") from None
        finally:
            if created and temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
