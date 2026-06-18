# Contributing to ditto-harness

Thank you for your interest in contributing! This document explains how to
contribute, the licensing terms that apply to contributions, and the CLA
requirement.

---

## Licensing of Contributions

ditto-harness is dual-licensed under **AGPL-3.0-or-later** (open source) and a
**Commercial License** (see [LICENSING.md](LICENSING.md)).

All inbound contributions are licensed under **AGPL-3.0-or-later**. By
submitting a contribution you also grant Omni Aura LLC the additional rights
described in the [Contributor License Agreement (CLA)](CLA.md), which are
necessary to maintain the dual-license model.

---

## Contributor License Agreement (CLA)

**All contributors must sign the CLA before their pull request can be merged.**

The CLA is collected automatically:

1. Open a pull request.
2. The **CLA Assistant** bot will post a comment on your PR.
3. Follow the link in the bot comment and sign electronically.
4. Once signed, the bot marks your PR as CLA-compliant and the check passes.

If you are contributing on behalf of a company or other legal entity, see
**Part II** of [CLA.md](CLA.md) and contact
[licensing@omniaura.ai](mailto:licensing@omniaura.ai) to arrange an entity
signature.

Prior contributors who have not yet signed the CLA may be asked to do so
retroactively.

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

Some integration tests (for example Ollama-backed tests) are gated behind
environment variables and skip automatically when those variables are unset.

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
6. Sign the CLA when prompted by the bot.
7. Address review feedback.

PRs that add new public API surface, change the database schema, or affect the
retrieval pipeline should include updated tests and, where appropriate, updated
documentation.

---

## Reporting Issues

Open a GitHub issue. For security vulnerabilities, please email
[licensing@omniaura.ai](mailto:licensing@omniaura.ai) directly rather than
filing a public issue.

---

## Questions

For licensing questions: [licensing@omniaura.ai](mailto:licensing@omniaura.ai)
For project questions: open a GitHub issue or discussion.
