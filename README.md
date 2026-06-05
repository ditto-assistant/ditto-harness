# Ditto Harness

Ditto Harness is the open-source memory and agent harness extracted from Ditto backend. It is intentionally smaller than the production backend: it stores and retrieves agent memories, exposes those memories as tools and an MCP server, and provides an importable agent loop with extension points for host applications.

The harness does not persist billing receipts or closed-source Ditto application features. It does expose usage and cost data so an importing service can store or bill for it separately.

## Packages

- `pkg/memory`: memory ingestion, fetch, vector search, composite search, subject search, subject-scoped memory search, retrieval metadata, and prompt context.
- `pkg/retrieval`: composite retrieval, retrieval event logging, auxiliary feature extraction, and a loadable learned-weight MLP predictor.
- `pkg/agent`: importable multi-turn agent loop with injectable models/tools, stream-style event hooks, tool loop detection, and cost collection.
- `pkg/mcpserver`: MCP tools for `save_memory`, `search_memories`, `search_subjects`, `search_memories_in_subjects`, and `fetch_memories`.
- `pkg/harness`: shared content, memory, subject, usage, and tool types.
- `pkg/testpg`: ephemeral PostgreSQL test helpers.
- `internal/db`: minimal sqlc-style query layer over the harness schema.

## Database

The required Postgres schema is in `db/migrations`. It requires `pgvector` and keeps only the slice needed by memory harness code:

- `harness_users`
- `memory_pairs`
- `subjects`
- `subject_memory_pair_links`
- `retrieval_events`

`db/query/memory.sql` is the source for the sqlc query surface. Generated code is checked in under `internal/sqlc`; `internal/db` is a small adapter with stable, friendlier harness types.

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
