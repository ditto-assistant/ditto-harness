# Rust Rewrite Experiment

Branch: `rust-rewrite`. Goal: rewrite the Go harness in Rust backed by Turso (SQLite-family,
native vector search), with TypeScript bindings via napi-rs, LLM access via rig.rs
(ollama + OpenAI-compatible endpoints: vLLM, OpenRouter), and a locally verifiable
dream pipeline (subject extraction/consolidation) using gemma4:e4b + embeddinggemma.

## Turso vector spike (Scaffold phase, 2026-06-10)

VERDICT: **native**. turso 0.7.0-pre.6 (`Builder::new_local(":memory:")`) supports:

- `CREATE TABLE t (id TEXT, emb F32_BLOB(4))` — OK
- `INSERT ... VALUES ('a', vector32('[1.0, 0.0, 0.0, 0.0]'))` — OK
- `SELECT vector_distance_cos(emb, vector32('[...]'))` — OK (returns cosine *distance*, i.e. `1 - cosine_similarity`)

No errors on any step. Schema therefore uses `F32_BLOB(768)` columns with
`vector_distance_cos()` for search; HNSW indexes are skipped (brute-force scan
per user). The spike lives on as permanent tests in
`crates/harness/src/db/mod.rs` (`#[cfg(test)] mod tests`), which also
probe blob-parameter binding, `ON CONFLICT ... DO UPDATE`, and `RETURNING`
support (results recorded below once measured).

### Probe results (permanent tests in `crates/harness/src/db/mod.rs`)

All supported by turso 0.7.0-pre.6 — verified by passing tests:

- Binding a raw little-endian f32 blob (`turso::Value::Blob`) into an
  `F32_BLOB(n)` column and as the parameter of `vector_distance_cos(col, ?)`
  works and is interchangeable with `vector32('[...]')` literals.
- `vector_distance_cos` semantics: identical vectors -> 0.0, orthogonal -> 1.0
  (cosine distance). Similarity = `1 - distance`.
- `INSERT ... ON CONFLICT (col) DO UPDATE SET ... RETURNING ...` works
  (upserts + RETURNING usable for CreateMemoryPair/UpsertSubject ports).
- `DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))` column defaults work and
  produce RFC3339-parseable text.
- `INTEGER PRIMARY KEY AUTOINCREMENT`, multi-column UNIQUE constraints, and
  composite PRIMARY KEYs all create fine via `Db::open_memory()` migrations.

### Workspace scaffold (Scaffold phase)

the repository root is a 3-crate workspace: `crates/harness` (lib, all public signatures
stubbed with `todo!()`), `crates/cli` (clap skeleton; handlers in
`crates/cli/src/commands/*.rs`), `crates/node` (napi-rs cdylib stub).
`cargo check --workspace` and `cargo test -p ditto-harness` pass.

## Integration summary (Integrate phase)

`cargo build --workspace`, `cargo test --workspace` (81 tests, 0 ignored,
no `DITTO_HARNESS_OLLAMA`), `cargo clippy --workspace --all-targets --
-D warnings`, and `cargo fmt --all --check` all pass.

### What drifted

- **The retrieval module never landed.** The agent assigned `retrieval`
  (composite retrieval, retrieval-event logging, aux features, MLP predictor)
  did not deliver; `retrieval/{mod,features,mlp}.rs` were still scaffold
  `todo!()` stubs while the other four modules were complete and calling into
  them. The Integrate phase ported the whole of Go `pkg/retrieval` from
  scratch:
  - `features.rs`: `extract_auxiliary_features[_context]` — keyword counts,
    named-entity pattern, question-type one-hot, hour-of-day sin/cos, log
    time/corpus features. Verbatim keyword lists; tests ported from
    `mlp_test.go` plus extra edge cases.
  - `mlp.rs`: tensor binary loader (`load`/`load_from_reader`), derived
    config (aux_dim/output_dim/use_scale), `predict_v2` forward pass
    (linear/layernorm/relu/softmax, scale head), `weights_from_slice`, and
    the `WeightPredictor` impl. Two defensive divergences from Go (which
    panics on malformed artifacts): tensor data lengths are validated at
    load time and short query embeddings are zero-padded to 768 — lib code
    here must not panic.
  - `mod.rs`: `composite_retrieve` ports `compositeSQLV1`/`compositeSQLV2`.
    The Postgres CTE pipeline is split for Turso: the candidate pool comes
    from a `vector_distance_cos` query with the same WHERE filters; the
    pair-frequency, bounds, subject-sem-match, neighbor-density aggregates
    and the weighted score/order (composite DESC, timestamp DESC, pair id
    DESC) are computed in Rust. Score semantics match Go exactly, including
    V1 *not* applying the scale factor (only V2 multiplies by `scale`), the
    global (not pool-restricted) per-subject link counts, and best-effort
    `log_event` (errors swallowed). `log_event` inserts into
    `retrieval_events` with weights/aux/ids as JSON text and the query
    embedding as an LE f32 blob (empty -> NULL).

### Fixes during integration

- Re-enabled `chat::tests::harness_prepare_run_and_save` (was `#[ignore]`d
  pending memory/retrieval); passes unmodified.
- Clippy `-D warnings` cleanups (no behavior change): redundant guard in
  `db::embedding_or_null` (`None | Some([])`), needless `.into_iter()` in
  `dream::run_model_stage`, collapsible match in `models::chunk_from_choice`
  (tool-call branch now a match guard), `expect_err` in an mlp test, needless
  borrow in a retrieval test, unused binding in a retrieval test fixture.
- One known contract divergence kept (documented on `RetrievalEvent` tests):
  Rust `Weights::default()` is Go `DefaultWeights()`, so Go's zero-value
  `Weights{}` must be spelled `Weights::ZERO`.

### Notes for later phases

- CLI `--json` output and a `tracing-subscriber` dependency were mentioned in
  the CLI task text but are not in the architect-owned clap surface; still
  open if wanted.
- `Model::next_streaming` uses the trait default (single aggregated chunk);
  rig's per-provider streaming can be wired later without trait changes.
- Ollama integration tests remain gated behind `DITTO_HARNESS_OLLAMA=1` and
  were not exercised (no local models pulled).
