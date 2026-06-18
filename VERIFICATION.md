# Rust Harness End-to-End Verification

Date: 2026-06-10. Run from `/Users/peyton/code/ditto/ditto-harness (repo root)` on macOS (Darwin 25.4.0).

## Environment pre-flight

- Ollama daemon 0.30.7 running at `localhost:11434` (started from
  `/Users/peyton/.local/ollama-latest/ollama serve`; the `ollama` on PATH is an
  outdated 0.15.2 client and was not used).
- `curl -s http://localhost:11434/api/tags` and
  `/Users/peyton/.local/ollama-latest/ollama list` both show the required models:

  ```
  NAME                     ID              SIZE      MODIFIED
  embeddinggemma:latest    85462619ee72    621 MB    ...
  gemma4:e4b               c6eb396dbd59    9.6 GB    ...
  ```

  `gemma4:e4b` reports capabilities `completion, tools, thinking`;
  `embeddinggemma` reports `embedding` with `embedding_length: 768`.
- `cargo build --workspace` — clean.

Fresh database for the run: `/tmp/ditto-harness-verify.db` (plus
`/tmp/ditto-harness-verify-or.db` for the OpenRouter pass).

## 1. Seed — PASS

```
cargo run -p ditto-harness-cli -- seed --db /tmp/ditto-harness-verify.db --user quinn
```

Output (trimmed):

```
seeding 12 memories for user "quinn" into /tmp/ditto-harness-verify.db
[ 1/12] seed-quinn-001  Quinn kicked off the Rust rewrite of the Go memory service; plan is to p...
[ 4/12] seed-quinn-004  Quinn planned a 3-day late-September Mount Rainier hiking trip with sist...
[ 6/12] seed-quinn-006  Quinn started a sourdough starter named Ferris; got a 1:1:1 daily feedin...
[12/12] seed-quinn-012  The Rust rewrite passed the full Go test suite with 40% faster memory se...
seeded 12 memories for user "quinn"
```

12 memories embedded via `embeddinggemma` and stored. The built-in sample data
also pre-seeds 9 baseline subjects (verified separately: a seed-only DB has
`SELECT COUNT(*) FROM subjects` = 9), which matters for interpreting the dream
counts below.

## 2. Dream (ollama, gemma4:e4b) — PASS

```
cargo run -p ditto-harness-cli -- dream --db /tmp/ditto-harness-verify.db --user quinn \
  --provider ollama --model gemma4:e4b
```

Post-dream DB state: 31 subjects, 47 subject–memory links over the 12 memories
(`sqlite3` counts), i.e. **22 subjects created by the dream pipeline** on top of
the 9 seeded ones. Subject-graph output (trimmed):

```
  sourdough baking [key]  (4 linked memories)   Quinn's sourdough hobby; starter named Ferris
  Mount Rainier trip [key]  (3 linked memories) Quinn's 3-day September hiking trip with sister Ava
  vector search  (3 linked memories)            Embedding similarity search design decisions
  Ava [key]  (2 linked memories)
  Ferris [key]  (2 linked memories)
  Rainier trip [key]  (2 linked memories)
  Borrow checker [key] / Database choice [key] / Helix [key] / Mount Rainier [key] /
  Rust Rewrite [key] / Spray Park to Mowich Lake loop [key] / preferences [key] ...
```

Quality: subjects are sensible and grounded in the seeded memories (people,
places, projects, preferences). One hallucinated description was observed
("Ferris ... perceived betrayal" — gemma4:e4b flavor text on an otherwise
correct subject); names/links themselves are all grounded.

## 3. Subjects — PASS

`subjects` requires `--query` (it is a vector search over the subject graph):

```
cargo run -p ditto-harness-cli -- subjects --db /tmp/ditto-harness-verify.db --user quinn \
  --query "rust rewrite"
```

```
 1. Rust rewrite [key]  similarity=0.6165  memories=7
 2. Rust rewrite storage layer [key]  similarity=0.5724  memories=1
 3. Rust Rewrite [key]  similarity=0.5295  memories=1
 4. borrow checker  similarity=0.5261  memories=1
 5. Borrow checker [key]  similarity=0.5059  memories=1
 ...
```

Subjects come back ranked with link counts and descriptions.

## 4. Search — PASS

```
cargo run -p ditto-harness-cli -- search --db /tmp/ditto-harness-verify.db --user quinn \
  --query "what is quinn building in rust"
```

```
intent: semantic  variant: v1
 1. seed-quinn-011  composite=0.6077 cosine=0.4079 recency=0.9630 frequency=1.0000
    Quinn returned from the Mount Rainier trip with Ava; ... talked through the Rust rewrite at camp.
 2. seed-quinn-012  composite=0.4920 ...  The Rust rewrite passed the full Go test suite with 40% faster memory search ...
 3. seed-quinn-010  composite=0.4464 ...  Debugged vector search in the Rust rewrite ...
 ...
 7. seed-quinn-001  ...  Quinn kicked off the Rust rewrite of the Go memory service ...
```

The seeded rust-rewrite memories dominate the top of the composite ranking
(composite = weighted cosine/recency/frequency, matching the Go retrieval
semantics; #1 is the most recent memory that mentions the rewrite).

## 5. Chat (ollama agent loop) — PASS

```
cargo run -p ditto-harness-cli -- chat --db /tmp/ditto-harness-verify.db --user quinn \
  --message "What do you remember about my hobbies?" --provider ollama --model gemma4:e4b
```

```
-> tool call search_memories [search_memories] {"queries":["hobbies"]}
<- tool result search_memories: ok
... It looks like you've been into **hiking** (specifically ... Mount Rainier),
**baking/sourdough**, and perhaps programming/technology ...
*   **Mount Rainier Trip:** ... 3-day late-September hiking trip with your sister Ava ...
*   **Sourdough Baking:** You started a sourdough starter named Ferris ... 75% hydration ...
memory tools called: search_memories
tokens: 1465 in / 271 out / 1736 total (2 calls)
saved memory pair: b4320295-4fdc-453f-9adf-1a1ee00183a2
```

The agent called the `search_memories` memory tool, the answer references the
seeded hobbies (sourdough + hiking, with correct details: Ferris, Ava,
Spray Park – Mowich Lake, 75% hydration), and the turn was persisted as a new
memory pair.

## 5b. vLLM / OpenAI-compatible path — PASS

Ollama exposes an OpenAI-compatible endpoint at `/v1`, so `--provider vllm
--base-url http://localhost:11434/v1` exercises exactly the code path a real
vLLM server would hit (shared OpenAI-compat client).

```
cargo run -p ditto-harness-cli -- search --db /tmp/ditto-harness-verify.db --user quinn \
  --query "what is quinn building in rust" \
  --provider vllm --base-url http://localhost:11434/v1 --model gemma4:e4b
```

Same ranking as step 4 (seed-quinn-011/012/010 on top; the chat-turn pair from
step 5 now also appears mid-list) — embedding + retrieval work through the
OpenAI-compat configuration.

```
cargo run -p ditto-harness-cli -- chat --db /tmp/ditto-harness-verify.db --user quinn \
  --message "What do you remember about my hobbies?" \
  --provider vllm --base-url http://localhost:11434/v1 --model gemma4:e4b
```

```
Based on our past conversations, I remember you having several hobbies! ...
*   Hiking: ... 3-day late-September hiking trip to Mount Rainier ... Spray Park - Mowich Lake loop.
*   Sourdough Baking: You started a sourdough starter named Ferris ... 75% hydration.
*   Programming/Technology: ... a rewrite of something using Rust ...
memory tools called: none
tokens: 772 in / 527 out / 1299 total (1 calls)
saved memory pair: da147f39-6316-4d0f-9cb6-f537be2ba581
```

Response is correct and grounded. On this turn the model answered from injected
context / prior turn instead of calling a tool ("memory tools called: none") —
the tool definitions were accepted by the OpenAI-compat endpoint without error,
and the turn completed and persisted. vllm-path check recorded as PASS.

## 6. OpenRouter dream — PASS

`OPENROUTER_API_KEY` sourced from `~/.zshrc` (present and valid). Second fresh
DB `/tmp/ditto-harness-verify-or.db`, seeded identically, then:

```
cargo run -p ditto-harness-cli -- dream --db /tmp/ditto-harness-verify-or.db --user quinn \
  --provider openrouter --model google/gemma-3-27b-it
```

```
dreaming over memories for user "quinn" ...
subjects created: 29
links created: 37
tokens: 4633 in / 1889 out / 6522 total (32 calls)

subject graph for user "quinn":
  Rust Memory Service [key]  (7 linked memories)
  Rust Rewrite Project [key]  (6 linked memories)
  sourdough baking [key]  (4 linked memories)
  Mount Rainier trip [key]  (3 linked memories)
  Ava Miller [key] / Ferris [key] / Sourdough [key]  (2 linked memories each)
  ... (29 created + 9 seeded = 38 subjects, 59 links total per sqlite3 counts)
```

The larger gemma-3-27b model produced a notably cleaner graph (e.g. "Rust
Memory Service" consolidating 7 memories). OpenRouter auth, request, and usage
accounting all worked on the first attempt with the slug
`google/gemma-3-27b-it`.

## 7. Test suite with gated ollama integration tests — PASS (after fix)

First run **failed**: the two gated tests (`models::tests::
ollama_chat_and_embed_integration`, `dream::tests::dream_ollama_integration`)
hardcoded the default chat model `gemma3:4b`, which is not pulled locally
(`404 model 'gemma3:4b' not found`); only `gemma4:e4b` is available.

**Fix applied** (test-only, no library behavior change): both gated tests now
read the chat model from `DITTO_HARNESS_OLLAMA_MODEL`, defaulting to
`DEFAULT_OLLAMA_CHAT_MODEL` (`gemma3:4b`) when unset.

- `crates/harness/src/models/mod.rs`: added `ollama_test_chat_model()` helper
  in the test module; integration test uses it.
- `crates/harness/src/dream.rs`: `dream_ollama_integration` reads the same env
  var inline.

Re-run:

```
DITTO_HARNESS_OLLAMA=1 DITTO_HARNESS_OLLAMA_MODEL=gemma4:e4b cargo test --workspace
```

```
test models::tests::ollama_chat_and_embed_integration ... ok
test dream::tests::dream_ollama_integration ... ok
test result: ok. 79 passed; 0 failed; 0 ignored  (ditto-harness lib)
test result: ok. 2 passed; 0 failed              (cli)
```

Also verified after the fix:

- plain `cargo test --workspace` (offline, no env) — 79 + 2 pass, gated tests
  no-op.
- `cargo clippy --workspace --all-targets -- -D warnings` — clean (one
  `needless_borrows_for_generic_args` introduced by the first version of the
  fix was corrected).
- `cargo fmt -p ditto-harness -p ditto-harness-cli --check` — clean.
  Note: `cargo fmt --all --check` reports a pre-existing import-ordering diff
  in `crates/node/src/lib.rs`; that file is owned by the node-bindings
  finalize work and was left untouched.

## Summary

| Step | Result |
| --- | --- |
| 1. seed (12 memories, embeddinggemma) | PASS |
| 2. dream via ollama gemma4:e4b (22 subjects created, 47 links) | PASS |
| 3. subjects listing with links | PASS |
| 4. vector search ranks rust-rewrite memories top | PASS |
| 5. chat agent loop calls `search_memories`, recalls sourdough + hiking | PASS |
| 5b. vLLM/OpenAI-compat path (ollama `/v1`) for search + chat | PASS |
| 6. OpenRouter dream (`google/gemma-3-27b-it`, 29 subjects) | PASS |
| 7. `DITTO_HARNESS_OLLAMA=1 cargo test --workspace` | PASS after test fix |
