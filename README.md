# Ditto Harness

Ditto Harness is the open-source memory and agent harness extracted from the
Ditto backend. It is intentionally smaller than the production backend: it
stores and retrieves agent memories, exposes those memories as tools, and
provides an importable agent loop with extension points for host applications.

The harness does not persist billing receipts or closed-source Ditto
application features. It does expose usage and cost data so an importing
service can store or bill for it separately.

## Implementation

This repository is a **Rust** workspace. The implementation is backed by
**embedded Turso/SQLite with native vector search** (no external database),
uses `rig-core` for model/embedder clients (Ollama, OpenRouter, vLLM), and
ships with **Node.js bindings** via `napi-rs`.

### Crates

- `crates/harness` — the library: `chat::Harness` (prepare → agent loop →
  save), `memory::Store` (ingest, vector + composite search, subjects),
  `retrieval` (composite V1/V2 scoring, the learned-weight `MlpPredictor`,
  and an optional second-stage `Reranker` hook), `models` (Ollama /
  OpenRouter / vLLM via `rig-core`), and `db` (embedded Turso schema +
  queries).
- `crates/cli` — a command-line interface over the library.
- `crates/node` — NAPI bindings for embedding in Node.js applications.

The retrieval pipeline mirrors the original production ranker 1:1: vector
candidate pool → composite V2 (7 signals + scale) with MLP-predicted fusion
weights → optional cross-encoder rerank (via the `Reranker` trait; the
concrete ONNX model lives in the consuming crate so this crate stays
inference-runtime-free).

```sh
cargo build
cargo test
```

Depend on the library from another crate (pin a `main` commit for
reproducible builds):

```toml
ditto-harness = { git = "https://github.com/ditto-assistant/ditto-harness", rev = "<main-commit>" }
```

## Database

The Turso/SQLite schema lives in `crates/harness/src/db/mod.rs`. It keeps
only the slice needed by the memory harness:

- `harness_users`
- `memory_pairs`
- `subjects`
- `subject_memory_pair_links`
- `retrieval_events`

No Postgres server or external vector extension is required; the database
opens locally as a file or in memory.

## Development

Run:

```sh
cargo build
cargo test
```

Some integration tests (for example Ollama-backed tests) are gated behind
environment variables and skip automatically when those variables are unset.

Run formatting and linting checks:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

Host applications bridge their model provider into `models::Model` and can
observe loop events by implementing `agent::EventHandler`. Backend-specific
streaming transports, billing persistence, and app-only tools stay outside
this repo.

The learned retrieval model is exposed as
`retrieval::load_mlp_predictor` / `retrieval::load_mlp_predictor_from_reader`.
This repo does not embed a model artifact; importing applications can load
their deployed model file and pass the resulting predictor into
`memory::Store::search_composite_memories`.

For backend-style chat integration, construct `chat::Harness` with a
`memory::Store`, a host model, and any application-specific tool values.
The standard memory tools can be included in the same agent loop while
keeping closed-source Ditto tools outside this crate.

Memory search tools return slim preview objects by default. `save_memory`
accepts optional subject links, and `fetch_memories` returns truncated full
user/assistant text for selected IDs. This keeps agent tool results
token-efficient while still letting the host fetch detailed memory content
when needed.

## License

ditto-harness is **dual-licensed** under:

- **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)** —
  see [`LICENSE`](LICENSE). Free to use for open-source and AGPL-compatible
  projects. The AGPL closes the "SaaS loophole": anyone who runs a modified
  version as a network service must publish their complete corresponding source
  to users of that service.

- **Commercial License** — see [`LICENSE-COMMERCIAL.md`](LICENSE-COMMERCIAL.md).
  For organizations that cannot comply with the AGPL (closed-source products,
  proprietary SaaS, embedded use without copyleft obligations). Contact
  [licensing@omniaura.ai](mailto:licensing@omniaura.ai).

See [`LICENSING.md`](LICENSING.md) for a plain-language explanation and a
"Do I need a commercial license?" decision guide.

**Contributors:** All contributors must sign the
[Contributor License Agreement](CLA.md) before their PR can be merged.
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
