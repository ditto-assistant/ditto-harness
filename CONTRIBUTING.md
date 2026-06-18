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

### Go (primary implementation)

Requirements: Go 1.22+, `sqlc` (installed as a Go tool in `go.mod`).

```sh
# Regenerate the sqlc query package (required after editing db/query/*.sql)
go tool sqlc generate -f sqlc.yaml

# Run all tests (unit tests run without Postgres; integration tests skip
# automatically when DITTO_HARNESS_TEST_DATABASE_URL is unset)
go test ./...

# Run integration tests (requires a Postgres instance with pgvector)
DITTO_HARNESS_TEST_DATABASE_URL=postgres://user:pass@localhost:5432/dbname \
  go test ./...
```

The generated `pkg/db` package is checked in. Always run `sqlc generate` and
commit the result alongside any SQL changes.

### Rust (portable rewrite)

Requirements: Rust stable toolchain, `cargo`.

```sh
cd rust
cargo build
cargo test
```

The Rust crate lives in `rust/crates/harness` (library), `rust/crates/cli`
(CLI), and `rust/crates/node` (NAPI bindings).

---

## Code Style

### Go

- Follow standard Go conventions (`gofmt`, `go vet`).
- Keep packages focused; avoid adding dependencies to `go.mod` without
  discussion.
- New public APIs should have at least one example test (`Example*` function).

### Rust

- Follow `rustfmt` defaults (`cargo fmt`).
- Run `cargo clippy` and address warnings before opening a PR.
- Mirror the Go package structure where applicable (the Rust port is intended
  to be a 1:1 functional equivalent).

---

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear, focused commits.
3. Ensure all tests pass (`go test ./...` and/or `cargo test`).
4. Open a pull request against `main`.
5. Sign the CLA when prompted by the bot.
6. Address review feedback.

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
