# AI Browser Agent specification

The product is a local-backend Chrome side-panel assistant for normal HTTP/HTTPS
websites. Current behavior is defined by the implementation and shared contracts.

- Observe the real DOM and address controls by snapshot-bound element identifiers.
- Summarize, search through observed inputs, click, type, scroll and navigate.
- Use a small local router and configurable main model behind a provider-neutral API.
- Run ordinary actions immediately; require one approval for each recognized
  destructive action. Keep password blocking and prompt/data trust separation.
- Bind each panel/run to its owner tab and session. Recover from stale observations;
  support Pause, Resume and Stop without replaying retired actions.
- Use toolbar-triggered access; persistent site access is optional. Unsupported
  browser/internal pages must fail cleanly.
- Support local PDF ingestion, text and optional visual retrieval, grounded answers,
  citations, file cards and prepared-document follow-ups.
- Keep provider credentials on the backend. Validate files exactly and retain private
  data only in configured local storage. Explicitly control cloud PDF-context sharing.
- Use `shared/build-contract.json` for protocol compatibility and structural root
  detection. Source snapshots must work without Git metadata.

See [architecture](docs/architecture.md), [data storage](docs/data-storage.md),
[testing](docs/testing.md) and [setup](README.md) for supported workflows and limits.
