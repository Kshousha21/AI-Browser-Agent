"""Owner-panel closure cancels and collects its run, even if it already failed."""
import asyncio

import pytest
from starlette.websockets import WebSocketDisconnect

from browser.protocol import make_envelope, PROTOCOL_VERSION, PROTOCOL_REVISION
from browser.providers import MockProvider
from browser.run_session import session_loop


@pytest.mark.asyncio
@pytest.mark.parametrize('completed', [False, True], ids=['pending-run', 'failed-run'])
async def test_owner_socket_close_always_collects_its_run(settings, monkeypatch, completed):
    entered = asyncio.Event()
    observed = dict(cancelled=False, exception_retrieved=False)
    tasks = []

    class RecordedTask(asyncio.Task):
        def exception(self):
            observed['exception_retrieved'] = True
            return super().exception()

    original_create = asyncio.create_task
    def create(coro, **kwargs):
        if coro.cr_code.co_name == 'document_run':
            task = RecordedTask(coro, **kwargs)
            tasks.append(task)
            return task
        return original_create(coro, **kwargs)
    monkeypatch.setattr(asyncio, 'create_task', create)

    async def document_run(self, kind, raw, bridge):
        entered.set()
        if completed:
            raise RuntimeError('Synthetic completed document transport failure')
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            observed['cancelled'] = True
            raise
    monkeypatch.setattr('document_workflow.session.DocumentSession.run', document_run)

    class Socket:
        def __init__(self):
            self.closed = False
            self.messages = iter([
                make_envelope('hello', dict(client='sidepanel/test', protocol_version=PROTOCOL_VERSION, protocol_revision=PROTOCOL_REVISION)).model_dump_json(),
                make_envelope('document_question', {}).model_dump_json(),
            ])
        async def receive_text(self):
            message = next(self.messages, None)
            if message is not None:
                return message
            await entered.wait()
            await asyncio.sleep(0)
            self.closed = True
            raise WebSocketDisconnect()
        async def send_text(self, text):
            assert not self.closed, 'No notification may be sent after owner socket closes'

    with pytest.raises(WebSocketDisconnect):
        await session_loop(Socket(), MockProvider(), settings, 'synthetic-owner', 'test')
    assert len(tasks) == 1 and tasks[0].done()
    if completed:
        assert observed['exception_retrieved'], 'A completed failed run was abandoned without retrieval'
    else:
        assert tasks[0].cancelled() and observed['cancelled']
