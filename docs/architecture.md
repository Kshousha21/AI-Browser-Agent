# Architecture

AI Browser Agent uses a Chrome Manifest V3 extension and a loopback Python backend.
The extension observes and acts on the real DOM of the toolbar-authorized HTTP/HTTPS
tab. Its native side panel owns the session. The backend chooses actions through a
provider-neutral tool layer; it never directly controls an unrelated browser tab.

## Request flow

1. The panel authenticates to the backend and checks the shared protocol contract.
2. A fresh page observation carries tab, document, navigation and snapshot identity.
3. The default local 2B router selects a tool group. The configurable main model
   chooses typed tool calls and produces an answer. Search uses observed DOM inputs.
4. The extension validates ownership, snapshot freshness and the execution lease
   before dispatch. Stale observations trigger a fresh read and a new decision.

`GenericSiteAdapter` is the default. `get_page_files` exposes bounded metadata for
visible file links, with references bound to the current observation. No site setup
is required. Chrome temporary access ends when its authorization no longer applies;
click the toolbar on the destination or use optional persistent site access.

## Control and trust

Ordinary browser actions run without confirmation. Recognized deletion or destructive
actions require one approval per action. Pause invalidates pending execution; Resume
observes the owner tab again. Stop or panel closure cancels the run. Another tab has
its own panel/session. Password fields are blocked. Page and tool data are separated
from trusted instructions, and sensitive values are redacted from diagnostics.

Providers include Ollama, OpenAI, Azure OpenAI, Anthropic, compatible endpoints and
mock. API keys and provider configuration remain backend-only. Cloud providers receive
the context needed for their selected tasks; local models keep inference local.

## Documents

Selected direct PDF links pass exact URL, filename, download, hash and storage checks.
Local ingestion extracts pages/chunks. BM25 and MiniLM retrieve text; optional Nomic
page-image retrieval combines evidence by reciprocal rank fusion. RAG validates
citations against supplied evidence and supports prepared-document follow-ups.
Cloud PDF-context sharing requires explicit consent. Application-specific viewers,
extensionless links and unexpected navigation destinations can remain unsupported.

## Source map

- `extension/src/content/`: DOM observations and actions.
- `extension/src/sidepanel/`: session, controls and browser tool execution.
- `backend/browser/`: orchestration, routing, providers and protocol validation.
- `backend/app/site_adapters/`: generic DOM metadata projection.
- `backend/document_workflow/`: selection, acquisition and follow-ups.
- `backend/local_pdf/`, `local_retrieval/`, `local_visual_retrieval/`, `local_rag/`:
  local document processing and retrieval.
- `shared/build-contract.json`: authoritative application/protocol versions.

See [setup](../README.md), [data storage](data-storage.md) and [testing](testing.md).
