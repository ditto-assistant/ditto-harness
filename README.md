# Ditto Harness

Ditto Harness is the open-source memory and agent harness extracted from Ditto backend. It is intentionally smaller than the production backend: it stores and retrieves agent memories, exposes those memories as tools and an MCP server, and provides an importable agent loop with extension points for host applications.

The harness does not persist billing receipts or closed-source Ditto application features. It does expose usage and cost data so an importing service can store or bill for it separately.

## Packages

- `pkg/memory`: memory ingestion, fetch, vector search, composite search, subject search, subject-scoped memory search, retrieval metadata, and prompt context.
- `pkg/retrieval`: composite retrieval, retrieval event logging, and learned-weight extension hooks.
- `pkg/agent`: importable multi-turn agent loop with injectable models and tools.
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

`db/query/memory.sql` is the source for the sqlc query surface. The generated-compatible Go wrapper in `internal/db` is kept small so the backend can import the public packages without taking Ditto backend dependencies.

## Development

Run:

```sh
go test ./...
```

Postgres integration tests require `DITTO_HARNESS_TEST_DATABASE_URL` pointing at an admin database that can create and drop test databases.

The default test command is still useful without Postgres: integration tests skip when `DITTO_HARNESS_TEST_DATABASE_URL` is unset.
