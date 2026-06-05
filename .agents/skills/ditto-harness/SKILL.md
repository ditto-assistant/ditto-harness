# Ditto Harness Development

Use this skill when changing the open-source Ditto agent memory harness.

## Rules

- Keep this repo importable by the closed-source backend; avoid dependencies on `github.com/ditto-assistant/backend`.
- Keep billing host-owned. Expose `harness.CostedUsage` values; do not write receipt tables here.
- Add app-specific tools through `harness.Tool` injection or MCP registration; do not add closed-source Ditto tools directly.
- Keep learned retrieval model artifacts host-owned. Use `retrieval.LoadMLPPredictor` when an importing app supplies the model file.
- Keep schema changes minimal and memory-focused: users, memory pairs, subjects, links, retrieval events.
- Run `go tool sqlc generate -f sqlc.yaml` after query/schema edits.
- Run `go test ./...` before committing. Postgres tests require `DITTO_HARNESS_TEST_DATABASE_URL`.

## Important Packages

- `pkg/memory` owns save/search/fetch/subject retrieval and prompt context assembly.
- `pkg/retrieval` owns composite retrieval, retrieval event logging, and learned-weight extension hooks.
- `pkg/agent` owns the importable agent loop, stream-style event hooks, tool execution, and loop detection.
- `pkg/chatv2` owns the importable backend-style facade that prepares memory context and runs/saves the agent turn.
- `pkg/mcpserver` exposes memory tools over MCP.
