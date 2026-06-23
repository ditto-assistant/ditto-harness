<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Database migrations

Versioned, forward-only SQL migrations for the harness's Turso (SQLite-family)
store. They are embedded into the binary at compile time (`include_str!`) and
applied in order by [`Db::migrate`](../src/db/mod.rs), which runs on every
`Db::open` / `Db::open_memory`.

## How it works

- Each `NNNN_name.sql` file is one migration. The numeric prefix is its
  **version** and defines apply order.
- Applied versions are recorded in a `schema_migrations(version, name,
  applied_at)` ledger table. A migration whose version is already in the ledger
  is skipped, so `migrate()` is idempotent and safe to call on every open.
- Each migration is applied atomically: the file's statements and the ledger
  insert run inside a single `BEGIN; … ; COMMIT;` batch. If any statement
  fails, the whole migration rolls back and no ledger row is written.
- The list of migrations lives in the `MIGRATIONS` array in
  [`src/db/mod.rs`](../src/db/mod.rs). **Adding a `.sql` file is not enough** —
  add a matching `Migration` entry there too (this keeps the embedded set
  explicit and reviewable, and lets `cargo build` catch a missing file).

## Authoring a new migration

1. Create the next-numbered file, e.g. `0002_add_widgets.sql`. Keep versions
   contiguous (no gaps, no reuse).
2. Use lowercase `snake_case` for the name in both the filename and the
   `MIGRATIONS` entry — the name is inlined into the ledger insert, so avoid
   quotes/apostrophes.
3. One logical change per file. Terminate every statement with `;`.
4. **Do not use `IF NOT EXISTS`** (or other "absorb existing state" guards) in
   new migrations — the ledger already guarantees exactly-once application.
   `0001_initial_schema.sql` is the sole exception: it is the baseline and uses
   `IF NOT EXISTS` so it applies cleanly to databases that predate this ledger.
5. **Never edit an already-released migration.** Applied databases will not
   re-run it. Fix forward with a new migration instead.
6. Add the `Migration { version, name, sql: include_str!(...) }` entry to
   `MIGRATIONS` in `src/db/mod.rs`, then run `cargo test -p ditto-harness` to
   verify it applies (the `migrations_apply_and_are_idempotent` test re-runs
   `migrate()` and checks the ledger).

## Convention notes

- Timestamps: store RFC3339 UTC text via
  `DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))` so lexicographic order
  matches time order.
- Vectors: 768-dim embeddings use Turso-native `F32_BLOB(768)` columns; bind
  little-endian f32 blobs (see `encode_f32_blob` in `src/db/mod.rs`).
