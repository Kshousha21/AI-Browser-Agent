"""Synthetic attachments and real local 4C/4D/4E services for Phase 4F tests."""
import hashlib
from pathlib import Path
from browser.config import Settings
from browser.actions import ActionResult
from browser.session_control import RunControl
from browser.provider_settings import ProviderSettings
from app.site_adapters.page_files import retrieve
from document_workflow.selection import CurrentSelection
from document_workflow.services import DocumentServices
from document_workflow.storage import AcquisitionStorage
from local_pdf import DocumentStore, PdfIngestionService
from local_retrieval import RetrievalService
from local_rag import RagService
from .pdf_fixtures import pdf_bytes
from .retrieval_fixtures import TestEmbedder
from .rag_fixtures import DeterministicRagProvider
from app.site_adapters import SiteAdapterRegistry
from browser.config import SiteAdaptersSettings
from browser.protocol import PageState, PageElement
from browser.providers import ToolCall
ALLOWLIST = []
def _registry(): return SiteAdapterRegistry(SiteAdaptersSettings())
def call(args, snapshot): return ToolCall(tool="get_page_files", arguments=args, snapshot_id=snapshot, reason="Observed files")
from .conftest import TEST_TOKEN

PDFS = [pdf_bytes([f'Synthetic photonics package {i} uses {12 + i} W. Copper removes heat.',
                  'Synthetic photonics engineering data. Ceramic housing.']) for i in range(2)]
URLS = [f'https://example.org/files/synthetic-{i}.pdf' for i in range(2)]


def settings():
    return Settings(auth={'token': TEST_TOKEN}, provider={'active': 'synthetic'},
        providers={'synthetic': {'type': 'ollama', 'model': 'synthetic', 'auth': {'mode': 'none'}}})


def source_page(**updates):
    raw = PageState(url="https://example.org/library", tab_id=1, document_id="document", snapshot_id="snapshot",
        elements=[PageElement(agent_id=f"f0:e-{i+1:03}", tag="a", role="link", name=f"synthetic-{i}.pdf", href=url)
                  for i,url in enumerate(URLS)])
    for key, value in updates.items():
        setattr(raw, key, value)
    return raw


def offer(selection=None, raw=None):
    selection = selection or CurrentSelection()
    page = _registry().refine_page_state(raw or source_page())
    result = retrieve(call({'limit': 2}, page.snapshot_id), page)
    payload = selection.offer(page, result)
    assert payload and len(payload['results']) == 2
    return selection, page, payload


def request(payload, count=1, **kwargs):
    return dict(offer_id=payload['offer_id'], result_refs=[r['result_ref'] for r in payload['results'][:count]],
                read_only=False, **kwargs)


def receipt(folder, nonce, data=PDFS[0]):
    folder = Path(folder) / 'AIBrowserAgent'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ('acq-' + nonce + '.pdf')
    path.write_bytes(data)
    return dict(acquisition_id=nonce, download_id=1, filename=str(path.absolute()), size=len(data), state='complete')


def services(folder, provider=None, remote=False):
    folder.mkdir(parents=True, exist_ok=True)
    ingest = PdfIngestionService(DocumentStore(folder / 'documents.sqlite3'))
    retrieval = RetrievalService(TestEmbedder(), source_path=ingest.store.path, index_path=folder / 'retrieval.sqlite3')
    provider = provider or DeterministicRagProvider()
    def rag_factory(retrieval, bridge):
        # Preserve the actual Phase 4E policy; replace only the provider client.
        bridge._factory = lambda _: provider
        return RagService(retrieval, bridge)
    config = ProviderSettings(name='ollama', model='synthetic', auth_mode='none',
                              base_url='https://provider.invalid' if remote else 'http://127.0.0.1:11434')
    service = DocumentServices(config, ingestion=ingest, retrieval=retrieval,
        storage=AcquisitionStorage(folder / 'acquired'), rag_factory=rag_factory)
    return service, provider


class Bridge:
    def __init__(self, folder, choices=None, raw=None):
        self.folder = folder
        self.control = RunControl()
        self.target = None
        self.run_id = 'synthetic-document-run'
        self.frames, self.calls, self.confirmations, self.audits = [], [], [], []
        self.choices = iter(choices or ['allow_once', 'allow_once'])
        self.raw = raw or source_page()
        self.download_error = False
        self.closed = False
        self.confirm_hook = None

    async def send(self, kind, data, *args):
        self.frames.append((kind, data))

    async def execute(self, call, *args):
        self.control.check()
        self.calls.append(call)
        if call.tool == 'get_page_state':
            raw = self.raw.model_copy(update={'snapshot_id': 'refreshed-' + str(len(self.calls))}, deep=True)
            return ActionResult(success=True, action=call.tool, snapshot_id=raw.snapshot_id, data={'page_state': raw.model_dump()})
        assert call.tool == 'acquire_attachment'
        if self.download_error:
            return ActionResult(success=False, action=call.tool, snapshot_id=call.snapshot_id, error='timeout')
        data = PDFS[URLS.index(call.arguments['url'])]
        return ActionResult(success=True, action=call.tool, snapshot_id=call.snapshot_id,
            data=receipt(self.folder, call.arguments['acquisition_id'], data))

    async def confirm(self, call, assessment, *args):
        self.confirmations.append((call.model_copy(deep=True), assessment))
        if self.confirm_hook:
            self.confirm_hook()
        return next(self.choices)

    async def audit(self, call, result, confirmation):
        self.audits.append((call.model_dump(), result.model_dump(), confirmation))

    def stop(self):
        self.control.invalidate('Stopped', state='stopped')

    async def close(self):
        self.closed = True

    @property
    def done(self):
        return [data for kind, data in self.frames if kind == 'document_done'][-1]

    @property
    def downloads(self):
        return [call for call in self.calls if call.tool == 'acquire_attachment']
