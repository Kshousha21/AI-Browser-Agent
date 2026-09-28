"""Acceptance-only Python network/process/import guards, never runtime configuration."""
from contextlib import contextmanager
import importlib.abc
import socket
import sys
from unittest.mock import patch


class BlockRemoteImports(importlib.abc.MetaPathFinder):
    prefixes = {"browser", "anthropic", "openai", "ollama", "huggingface_hub"}

    def __init__(self, attempts: list[str]):
        self.attempts = attempts

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.prefixes:
            self.attempts.append("remote_import")
            raise AssertionError("remote_import")
        return None


@contextmanager
def offline_guard():
    """Count and reject Python DNS/socket requests; not an OS packet capture."""
    from contextlib import ExitStack
    # Python 3.14 may query Windows metadata through cmd.exe on first use.
    import platform
    platform.uname()
    attempts = []

    def reject(*args, **kwargs):
        attempts.append("network")
        raise AssertionError("network")

    finder = BlockRemoteImports(attempts)
    sys.meta_path.insert(0, finder)
    try:
        with ExitStack() as stack:
            for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "create_connection"):
                stack.enter_context(patch.object(socket, name, reject))
            for name in ("connect", "connect_ex", "sendto"):
                stack.enter_context(patch.object(socket.socket, name, reject))
            stack.enter_context(patch("urllib.request.urlopen", reject))
            stack.enter_context(patch("http.client.HTTPConnection.request", reject))
            stack.enter_context(patch("subprocess.Popen", reject))
            yield attempts
    finally:
        sys.meta_path.remove(finder)
