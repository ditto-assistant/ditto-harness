# Ditto Harness Development

Use this skill when changing the open-source Ditto agent memory harness.

## Rules

- Keep this repo importable by the closed-source backend; avoid dependencies on `github.com/ditto-assistant/backend`.
- Keep billing host-owned. Expose `CostedUsage` values; do not write receipt tables here.
- Add app-specific tools through the `agent::Tool` / `ToolExecutor` traits; do not add closed-source Ditto tools directly.
- Keep learned retrieval model artifacts host-owned. Use `retrieval::load_mlp_predictor` when an importing app supplies the model file.
- Keep schema changes minimal and memory-focused: users, memory pairs, subjects, links, retrieval events.
- Run `cargo build` and `cargo test` before committing.
- Run `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings` before opening a PR.

## Important Crates

- `crates/harness` owns save/search/fetch/subject retrieval and prompt context assembly.
- `crates/harness/src/retrieval` owns composite retrieval, retrieval event logging, and learned-weight extension hooks.
- `crates/harness/src/agent` owns the importable agent loop, stream-style event hooks, tool execution, and loop detection.
- `crates/harness/src/chat` owns the importable backend-style facade that prepares memory context and runs/saves the agent turn.
- `crates/harness/src/db` owns the embedded Turso/SQLite schema and queries.
- `crates/cli` is the command-line interface.
- `crates/node` exposes NAPI bindings.
