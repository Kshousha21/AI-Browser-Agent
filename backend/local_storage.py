"""Shared private runtime paths and fail-closed local SQLite opening.

No provider/config-file imports, migration, database inspection or directory
creation during discovery. BROWSER_DATA_ROOT is an explicit process setting.
Public read-only model artifacts keep their existing separate cache location.
"""
import ctypes
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]


class StorageError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def filesystem_kind(path: Path) -> str:
    """Classify the resolved backing volume, not the apparent drive spelling."""
    path = path.resolve()
    if str(path).startswith(("\\\\", "//")):
        return "network"
    if sys.platform == "win32":
        get_type = ctypes.windll.kernel32.GetDriveTypeW
        get_type.argtypes = [ctypes.c_wchar_p]
        get_type.restype = ctypes.c_uint
        kind = get_type(path.anchor)
        return "local" if kind in {2, 3, 6} else "network" if kind == 4 else "unknown"
    if sys.platform.startswith("linux"):
        # Reject unrecognized/network mounts. Longest mount point wins.
        mounts = []
        for row in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = row.split(" - ", 1)
            mount = Path(left.split()[4].replace("\\040", " ").replace("\\134", "\\"))
            if path.is_relative_to(mount):
                mounts.append((len(mount.parts), right.split()[0]))
        kind = max(mounts, default=(0, "unknown"))[1]
        if kind in {"ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "overlay", "zfs", "f2fs"}:
            return "local"
        return "network" if kind in {"nfs", "nfs4", "cifs", "smb3", "9p", "ceph", "afs"} else "unknown"
    return "unknown"


def require_local(path: Path) -> Path:
    try:
        kind = filesystem_kind(path)
    except (OSError, ValueError):
        raise StorageError("invalid_storage_path") from None
    if kind == "network":
        raise StorageError("network_storage_unsupported")
    if kind != "local":
        raise StorageError("invalid_storage_path")
    return path


def private_path(value: Path | str) -> Path:
    """Preserve local .local paths, reject Git source paths and special files."""
    try:
        raw = os.fspath(value)
        if not raw or not isinstance(raw, str) or "\x00" in raw or raw.startswith("file:"):
            raise StorageError("invalid_storage_path")
        if raw.startswith(("\\\\", "//")):
            candidate = Path(raw)
            if candidate.is_relative_to(ROOT) and not candidate.is_relative_to(ROOT / ".local"):
                raise StorageError("invalid_storage_path")
            raise StorageError("network_storage_unsupported")
        path = Path(raw).expanduser().resolve()
        if path == Path(path.anchor):
            raise StorageError("invalid_storage_path")
        if sys.platform == "win32" and any(":" in part for part in path.parts[1:]):
            raise StorageError("invalid_storage_path")  # No NTFS alternate streams.
        if path.is_relative_to(ROOT):
            if not path.is_relative_to(ROOT / ".local"):
                raise StorageError("invalid_storage_path")
        elif any((parent / ".git").exists() for parent in [path, *path.parents]):
            raise StorageError("invalid_storage_path")
        return path
    except (OSError, ValueError, TypeError):
        raise StorageError("invalid_storage_path") from None


def data_root() -> Path:
    """Local checkout -> ignored .local; network checkout -> local per-user root.

    An explicit absolute override never silently falls back or relocates data.
    No existing database is opened, copied, deleted or migrated here.
    """
    override = os.environ.get("BROWSER_DATA_ROOT")
    try:
        if override is not None:
            if not override.strip() or not Path(override).is_absolute():
                raise StorageError("invalid_storage_path")
            target = private_path(override)
        elif filesystem_kind(ROOT) == "local":
            target = private_path(ROOT / ".local")
        elif sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA", "")
            if not base or not Path(base).is_absolute():
                raise StorageError("invalid_storage_path")
            target = private_path(Path(base) / "AIBrowserAgent" / "data")
        else:
            base = os.environ.get("XDG_DATA_HOME")
            if base and not Path(base).is_absolute():
                raise StorageError("invalid_storage_path")
            target = private_path((Path(base) if base else Path.home() / ".local/share")
                                  / "ai-browser-agent")
        return require_local(target)
    except (OSError, ValueError, TypeError):
        raise StorageError("invalid_storage_path") from None


def database_path(value: Path | str | None, filename: str) -> Path:
    path = private_path(data_root() / filename if value is None else value)
    return require_local(path)


def readonly_uri(path: Path) -> str:
    """Pathlib encodes drive letters, Unicode, %, spaces, # and ? correctly.

    Nonempty UNC authorities are NOT made usable: the network safety gate
    rejects them before SQLite, even on builds accepting URI authorities.
    """
    return require_local(private_path(path)).as_uri() + "?mode=ro"


def connect_database(path: Path, *, write: bool = False) -> sqlite3.Connection:
    path = require_local(private_path(path))
    if write:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Recheck after creation/resolution. Normal filename API needs no URI.
        require_local(private_path(path))
        return sqlite3.connect(path, timeout=5, isolation_level=None)
    return sqlite3.connect(readonly_uri(path), uri=True, timeout=5, isolation_level=None)


def readiness() -> dict:
    """Read-only metadata check. Never open a database, probe-write or log paths."""
    try:
        root = data_root()
        ancestor = root
        while not ancestor.exists():
            ancestor = ancestor.parent
        ready = ancestor.is_dir() and os.access(ancestor, os.W_OK | os.X_OK)
        return {"category": "local", "ready": ready,
                "state": "present" if root.is_dir() else "not_created" if ready else "unavailable",
                "source": "override" if "BROWSER_DATA_ROOT" in os.environ else "automatic"}
    except (StorageError, OSError, ValueError):
        return {"category": "unsupported", "ready": False, "state": "unavailable"}
