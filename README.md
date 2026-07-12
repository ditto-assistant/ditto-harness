# Ditto Harness

The memory and agent harness extracted from the
Ditto backend. It is intentionally smaller than the production backend: it
stores and retrieves agent memories, exposes those memories as tools, and
provides an importable agent loop with extension points for host applications.

The harness does not persist billing receipts or closed-source Ditto
application features. It does expose usage and cost data so an importing
service can store or bill for it separately.

## Implementation

This repository is a Rust workspace. The implementation is backed by
embedded Turso/SQLite with native vector search, uses `rig-core` for
model/embedder clients (Ollama plus any OpenAI-compatible endpoint: OpenRouter,
vLLM, Chutes), and ships with Node.js bindings via `napi-rs`.

### Crates

- `crates/harness`: the library. `chat::Harness` (prepare → agent loop →
  save), `memory::Store` (ingest, vector + composite search, subjects),
  `retrieval` (composite V1/V2 scoring, the learned-weight `MlpPredictor`,
  and an optional second-stage `Reranker` hook), `models` (Ollama plus
  OpenAI-compatible endpoints — OpenRouter, vLLM, Chutes — via `rig-core`),
  and `db` (embedded Turso schema + queries).
- `crates/cli`: a command-line interface over the library.
- `crates/node`: NAPI bindings for embedding in Node.js applications.

The retrieval pipeline mirrors the original production ranker 1:1: vector
candidate pool → composite V2 (7 signals + scale) with MLP-predicted fusion
weights → optional cross-encoder rerank (via the `Reranker` trait; the
concrete ONNX model lives in the consuming crate so this crate stays
inference-runtime-free).

Depend on the library from another crate (pin a `main` commit for
reproducible builds):

```toml
ditto-harness = { git = "https://github.com/ditto-assistant/ditto-harness", rev = "<main-commit>" }
```

## Architecture

The crate serves a chat turn and ingests memories into the subject graph.

### Chat turn (`chat::Harness::run`)

1. `prepare`: normalize messages (system prompt first; `user_input` seeds
   the first user message when the history has no non-system messages),
   resolve ids (empty `kg_id` derives from the user id, empty `session_id`
   becomes "main"), run memory retrieval via `Store::get_prompt_memories`,
   and insert the memory-context system message after the leading system
   block.
2. `agent::Loop::run_streaming`: up to `max_turns` model turns; each turn
   either ends the run with final text or dispatches one tool call. Repeated
   identical tool calls trip loop detection, which substitutes a canned tool
   result and a synthesis prompt. `agent::EventHandler` observes each step.
3. Optional save: with `save_memory`, the final exchange is persisted via
   `Store::save_memory`, carrying the seed-memory and retrieval metadata from
   preparation.

### Ingest (`seed`, then `dream`)

1. `Store::save_memory`: embeds `prompt\nresponse\nsummary` as one text,
   upserts the pair, and embeds/upserts/links any provided subjects.
2. `dream::Dreamer::dream`: extract durable subjects per memory pair (the
   only LLM stage), dedup against existing subject embeddings (merge at
   cosine >= `SUBJECT_MERGE_THRESHOLD`), link subjects to pairs, and refine
   merge-accumulated subjects.
3. The resulting subject graph feeds composite retrieval's
   subject-frequency, subject-semantic-match, and neighbor-density signals.

Retrieval mode is selected by `chat::PrepareRequest::use_composite`, which
defaults to false. When false, long-term retrieval is plain vector search
(`Store::search_memories`): the composite scorer, the `WeightPredictor` /
`MlpPredictor`, and the `Reranker` are all bypassed, and the subject graph
contributes nothing to ranking. The DittoBench reference baseline sets it to
true, so the scored path exercises the full composite stack; a fork that
leaves it false is benchmarking bare vector search.

Extension points (trait -> what you replace):

- `types::Embedder`: the embedding backend (768-dim contract; see its docs).
- `retrieval::WeightPredictor`: per-query fusion weights; `MlpPredictor` is
  the loadable learned implementation.
- `retrieval::Reranker`: optional second-stage rerank over the composite
  pool.
- `types::Model`: the chat model driving the agent loop.
- `agent::EventHandler`: streaming observer for loop events.

## CLI quickstart

The `crates/cli` package builds a `ditto-harness` binary with `seed`, `dream`,
`search`, `subjects`, and `chat` subcommands. With a local Ollama server
running:

```sh
ollama pull embeddinggemma && ollama pull gemma3:4b

# Seed the built-in sample memories for a user (or pass --file <memories.json>):
cargo run -p ditto-harness-cli -- seed --db mem.db --user quinn

# Run the dream pipeline (subject extraction/consolidation):
cargo run -p ditto-harness-cli -- dream --db mem.db --user quinn

# Run one memory-augmented chat turn through the agent loop:
cargo run -p ditto-harness-cli -- chat --db mem.db --user quinn --message "What do you remember about my hobbies?"
```

`--provider openrouter` (requires `OPENROUTER_API_KEY`) and `--provider vllm`
(with `--base-url`) select other chat model providers. Both require `--model`
(e.g. `--model anthropic/claude-3.5-haiku`); only Ollama has a default chat
model (`gemma3:4b`), which `--model` overrides.

## Node.js bindings

`crates/node` exposes the harness to Node.js via a napi-rs binding
(`@ditto/harness-node`):

```sh
cd crates/node
npm install       # installs @napi-rs/cli
npm run build     # builds the native addon
node smoke.mjs    # verifies the binding
```

`Harness.open` accepts an
`ollamaBaseUrl` option (the smoke test reads the `OLLAMA_BASE_URL` env var)
and an `embedder: "hash"` option, a deterministic offline stub embedder for
tests and CI machines without Ollama.

## Database

The Turso/SQLite schema lives in `crates/harness/src/db/mod.rs`. It keeps
only the slice needed by the memory harness:

- `harness_users`
- `memory_pairs`
- `subjects`
- `subject_memory_pair_links`
- `retrieval_events`

Vector retrieval is a brute-force per-user scan over `F32_BLOB(768)`
columns (no HNSW/ANN index).

## Development

Run:

```sh
cargo build
cargo test
```

The Ollama-backed integration tests are gated behind `DITTO_HARNESS_OLLAMA=1`
(chat model overridable via `DITTO_HARNESS_OLLAMA_MODEL`, default `gemma3:4b`)
and skip automatically when the variable is unset.

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
`retrieval::MlpPredictor::load` (from a file path) /
`retrieval::MlpPredictor::load_from_reader` (from any `Read`).
This repo does not embed a model artifact; importing applications can load
their deployed model file and pass the resulting predictor into
`memory::StoreOptions::predictor` so `Store::search_composite_memories`
uses it.

For backend-style chat integration, construct `chat::Harness` with a
`memory::Store`, a host model, and any application-specific tool values.
The standard memory tools can be included in the same agent loop e.g. `fetch_memories` and `save_memory`.

Memory search tools return slim preview objects by default. `save_memory`
accepts optional subject links, and `fetch_memories` returns truncated full
user/assistant text for selected IDs.

## SN118 model lock

The harness is model-agnostic (Ollama, OpenRouter, vLLM), but SN118 scored runs
do not let a miner pick the model. The validator locks inference to one frozen
open-weight model, Qwen3-32B, served in a hardware-attested Trusted Execution
Environment (Chutes `Qwen/Qwen3-32B-TEE`). A model-pinning relay gateway forces
the model id and the reasoning mode (thinking off) on every request, and the
sandbox has fail-closed egress: it reaches only the relay, holds no upstream
key, and cannot route to another model. Local practice can run any provider;
only the locked model counts when scored.

## SN118 originality note

This crate is the shared reference harness that SN118 miners depend on.
Depending on it and converging on its structure, retrieval pipeline, and
prompts is expected. The subnet's duplicate-detection gate compares miners'
own uploaded submission crates against each other (across exact,
normalized-source, lexical, structural, prompt, and semantic-embedding
dimensions); it holds copies of another miner's submission for review, with
first-seen protecting the original author, and requires agreement across
independent signals before flagging near-duplicates, so this shared
dependency does not trip it. The miner-facing details live in the
[starter kit](https://github.com/ditto-assistant/dittobench-starter-kit) README.

## License

ditto-harness is licensed under the MIT License; see [`LICENSE`](LICENSE).
The license permits use in closed-source and hosted products. Contributions
are accepted under the same terms (see [CONTRIBUTING.md](CONTRIBUTING.md)).
