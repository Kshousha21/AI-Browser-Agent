# Developer reference

Use the current implementation and shared contracts as the source of truth.

- [Setup, models, Chrome permissions and troubleshooting](README.md)
- [Architecture and trust boundaries](docs/architecture.md)
- [Private data, backup and removal](docs/data-storage.md)
- [Tests, builds and live verification](docs/testing.md)
- [Current product specification](SPEC.md)
- [Backend configuration example](backend/config.example.yaml)
- [Private environment template](backend/.env.example)
- [Application/protocol contract](shared/build-contract.json)
- [Local developer tools](tools/README.md)

Root detection requires the backend and extension directories plus the shared build
contract. It does not search for a parent checkout or require Git metadata. Provider
keys belong only in backend configuration. Keep local data and test profiles private.
