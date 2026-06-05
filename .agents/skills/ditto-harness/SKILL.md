# Ditto Harness Development

Use this skill when changing the open-source Ditto agent memory harness.

## Rules

- Keep this repo importable by the closed-source backend; avoid dependencies on `github.com/ditto-assistant/backend`.
- Keep billing host-owned. Expose `harness.CostedUsage` values; do not write receipt tables here.
- Add app-specific tools through `harness.Tool` injection or MCP registration; do not add closed-source Ditto tools directly.
- Keep schema changes minimal and memory-focused: users, memory pairs, subjects, links, retrieval events.
- Run `go test ./...` before committing. Postgres tests require `DITTO_HARNESS_TEST_DATABASE_URL`.

## Important Packages

- `pkg/memory` owns save/search/fetch/subject retrieval.
- `pkg/retrieval` owns composite retrieval and learned-weight extension hooks.
- `pkg/agent` owns the importable agent loop.
- `pkg/mcpserver` exposes memory tools over MCP.
