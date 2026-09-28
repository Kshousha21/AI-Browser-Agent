"""Provider-independent interruption boundary for manual browser take-over."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class ControlInterrupted(Exception):
    """Work belongs to a target/decision epoch that is no longer actionable."""


class RunStopped(ControlInterrupted):
    pass


class RunControl:
    def __init__(self) -> None:
        self.state = "running"
        self.epoch = 0
        self.reason = "Ready"
        self.needs_refresh = False
        self._changed = asyncio.Event()
        self._ready = asyncio.Event()
        self._ready.set()
        self._discarded: set[asyncio.Future[Any]] = set()
        self.on_state: Callable[[], Awaitable[Any]] | None = None

    def invalidate(self, reason: str, *, state: str = "paused") -> None:
        self.epoch += 1
        self.state = state
        self.reason = reason
        self.needs_refresh = True
        self._changed.set()
        self._changed = asyncio.Event()
        self._ready.clear()
        if state == "stopped":
            self._ready.set()

    def resume(self) -> None:
        if self.state == "stopped":
            return
        self.state = "resuming"
        self.reason = "Resuming - reading the current allowed page"
        self.needs_refresh = True
        self._ready.set()

    async def notify(self) -> None:
        if self.on_state is not None:
            await self.on_state()

    async def wait_ready(self) -> None:
        await self._ready.wait()
        if self.state == "stopped":
            raise RunStopped()

    def check(self, *, refreshing: bool = False) -> None:
        if self.state == "stopped":
            raise RunStopped()
        if self.state != "running" and not (refreshing and self.state == "resuming"):
            raise ControlInterrupted()

    async def complete_refresh(self) -> None:
        self.state = "running"
        self.reason = "Fresh page observed"
        self.needs_refresh = False
        await self.notify()

    async def interruptible(self, operation: Awaitable[T], *, refreshing: bool = False) -> T:
        task = asyncio.ensure_future(operation)
        changed = asyncio.create_task(self._changed.wait())
        epoch = self.epoch
        try:
            self.check(refreshing=refreshing)
            done, _pending = await asyncio.wait({task, changed}, return_when=asyncio.FIRST_COMPLETED)
            if changed in done or epoch != self.epoch:
                raise ControlInterrupted()
            self.check(refreshing=refreshing)
            return await task
        finally:
            for pending in (task, changed):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(changed, return_exceptions=True)
            # Give cooperative cancellation a turn, but never wait for a provider
            # that ignores it. Its result has no route back into the agent loop.
            if not task.done():
                await asyncio.sleep(0)
            if task.done():
                if not task.cancelled():
                    task.exception()
            else:
                self._discarded.add(task)
                task.add_done_callback(self._discard_result)

    def _discard_result(self, task: asyncio.Future[Any]) -> None:
        self._discarded.discard(task)
        if not task.cancelled():
            task.exception()
