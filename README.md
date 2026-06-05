# Ditto Harness

Ditto Harness is the open-source memory and agent harness extracted from Ditto backend. It is intentionally smaller than the production backend: it stores and retrieves agent memories, exposes those memories as tools and an MCP server, and provides an importable agent loop with extension points for host applications.

The harness does not persist billing receipts or closed-source Ditto application features. It does expose usage and cost data so an importing service can store or bill for it separately.

## Packages

- `pkg/memory`: memory ingestion, fetch, vector search, composite search, subject search, subject-scoped memory search, slim memory tool payloads, retrieval metadata, and prompt context.
- `pkg/retrieval`: composite retrieval, retrieval event logging, auxiliary feature extraction, and a loadable learned-weight MLP predictor.
- `pkg/agent`: importable multi-turn agent loop with injectable models/tools, stream-style event hooks, tool loop detection, and cost collection.
- `pkg/chatv2`: importable chat harness facade that prepares memory context, combines injected tools with memory tools, runs the agent loop, returns cost data, and saves the resulting memory pair with seed/retrieval metadata.
- `pkg/mcpserver`: MCP tools for `save_memory`, `search_memories`, `search_subjects`, `search_memories_in_subjects`, and `fetch_memories`.
- `pkg/db`: generated Postgres/sqlc query package for wiring `memory.Store` from an importing service.
- `pkg/harness`: shared content, memory, subject, usage, and tool types.
- `pkg/testpg`: ephemeral PostgreSQL test helpers.

## Database

The required Postgres schema is in `db/migrations`. It requires `pgvector` and keeps only the slice needed by memory harness code:

- `harness_users`
- `memory_pairs`
- `subjects`
- `subject_memory_pair_links`
- `retrieval_events`

`db/query/memory.sql` is the source for the sqlc query surface. Generated code is checked in under `pkg/db` and can be constructed directly with `pkg/db.New`.

## Development

Run:

```sh
go tool sqlc generate -f sqlc.yaml
go test ./...
```

Postgres integration tests require `DITTO_HARNESS_TEST_DATABASE_URL` pointing at an admin database that can create and drop test databases.

The default test command is still useful without Postgres: integration tests skip when `DITTO_HARNESS_TEST_DATABASE_URL` is unset.

Host applications bridge their model provider into `harness.Model` and can observe loop events by implementing `agent.EventHandler`. Backend-specific streaming transports, billing persistence, and app-only tools stay outside this repo.

The learned retrieval model is exposed as `retrieval.LoadMLPPredictor` /
`retrieval.LoadMLPPredictorFromReader`. This repo does not embed a model
artifact; importing applications such as `chatv2` can load their deployed model
file and pass the resulting predictor into `memory.Store.SearchCompositeMemories`.

For backend-style chat integration, construct `chatv2.Harness` with a
`memory.Store`, host `harness.Model`, and any application-specific
`harness.Tool` values. Set `IncludeMemoryTools` to expose the standard memory
tools in the same agent loop while keeping closed-source Ditto tools outside
this module.

Memory search tools and MCP search handlers return slim preview objects by
default. `save_memory` accepts optional subject links, and `fetch_memories`
returns truncated full user/assistant text for selected IDs. This keeps agent
tool results token-efficient while still letting the host fetch detailed memory
content when needed.
