# Contributing to ditto-harness

## Licensing of Contributions

ditto-harness is licensed under the MIT License ([LICENSE](LICENSE)). By
submitting a pull request you agree that your contribution is licensed under
the same terms.

---

## Development Setup

Requirements: Rust stable toolchain, `cargo`.

```sh
cargo build
cargo test
```

The workspace crates live at:

- `crates/harness` — the library
- `crates/cli` — the command-line interface
- `crates/node` — Node.js NAPI bindings

Some integration tests (the Ollama-backed tests, gated behind
`DITTO_HARNESS_OLLAMA=1`) skip automatically when the gating environment
variable is unset — see "Verifying locally" below.

---

## Verifying locally

Prerequisites for the full (Ollama-backed) test pass:

- a running Ollama daemon (`ollama serve`)
- `ollama pull embeddinggemma` (embeddings)
- `ollama pull gemma3:4b` (default chat model)

Then:

```sh
cargo test --workspace
```

The Ollama integration tests are gated behind `DITTO_HARNESS_OLLAMA=1` and
skip automatically when it is unset. To use a different local chat model, set
`DITTO_HARNESS_OLLAMA_MODEL` (defaults to `gemma3:4b`):

```sh
DITTO_HARNESS_OLLAMA=1 DITTO_HARNESS_OLLAMA_MODEL=gemma3:4b cargo test --workspace
```

---

## Code Style

- Follow `rustfmt` defaults (`cargo fmt`).
- Run `cargo clippy --workspace --all-targets -- -D warnings` and address
  warnings before opening a PR.
- Keep packages focused; avoid adding new dependencies without discussion.

---

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear, focused commits.
3. Ensure the workspace builds and tests pass (`cargo build`, `cargo test`).
4. Run `cargo fmt --check` and `cargo clippy`.
5. Open a pull request against `main`.
6. Address review feedback.

PRs that add new public API surface, change the database schema, or affect the
retrieval pipeline should include updated tests and, where appropriate, updated
documentation.

---

## Reporting Issues

Open a GitHub issue. For security vulnerabilities, email
[licensing@omniaura.ai](mailto:licensing@omniaura.ai) directly rather than
filing a public issue.

---

## Questions

For licensing questions: [licensing@omniaura.ai](mailto:licensing@omniaura.ai)
For project questions: open a GitHub issue or discussion.
