-- 0001_initial_schema
--
-- Baseline schema for the Turso (SQLite-family) memory store. Ports the
-- Postgres schema from the Go harness (`db/migrations/000001_memory_schema.up.sql`)
-- to Turso's SQLite dialect. See `migrations/README.md` for the authoring
-- convention and `crates/harness/src/db/mod.rs` for the schema-mapping rationale
-- (UUID->TEXT, BIGSERIAL->INTEGER AUTOINCREMENT, TIMESTAMPTZ->RFC3339 TEXT,
-- JSONB/TEXT[]->TEXT JSON, vector(768)->native F32_BLOB(768)).
--
-- This is the baseline migration: it uses `IF NOT EXISTS` so it applies cleanly
-- both to a fresh database and to one already provisioned by the pre-migration
-- idempotent schema bootstrap. Later migrations must NOT use `IF NOT EXISTS`
-- (the `schema_migrations` ledger guarantees each runs exactly once).

CREATE TABLE IF NOT EXISTS harness_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS memory_pairs (
    id TEXT PRIMARY KEY,
    firestore_pair_id TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES harness_users(uid) ON UPDATE CASCADE ON DELETE CASCADE,
    kg_id TEXT NOT NULL,
    session_id TEXT,
    title TEXT,
    description TEXT,
    prompt TEXT,
    response TEXT,
    input TEXT,
    output TEXT,
    source TEXT,
    source_context TEXT,
    timestamp TEXT NOT NULL,
    timezone_offset INTEGER,
    seed_memories TEXT,
    retrieval_metadata TEXT,
    conversation_embedding F32_BLOB(768),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (user_id, firestore_pair_id)
);

CREATE TABLE IF NOT EXISTS subjects (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES harness_users(uid) ON UPDATE CASCADE ON DELETE CASCADE,
    kg_id TEXT NOT NULL,
    subject_text TEXT NOT NULL,
    description_text TEXT,
    is_key_subject INTEGER NOT NULL DEFAULT 0,
    embedding F32_BLOB(768),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (user_id, kg_id, subject_text)
);

CREATE TABLE IF NOT EXISTS subject_memory_pair_links (
    subject_id TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    pair_id TEXT NOT NULL REFERENCES memory_pairs(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES harness_users(uid) ON UPDATE CASCADE ON DELETE CASCADE,
    kg_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (subject_id, pair_id)
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    kg_id TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    request_path TEXT NOT NULL DEFAULT '',
    query TEXT NOT NULL DEFAULT '',
    query_embedding F32_BLOB(768),
    retrieved_pair_ids TEXT NOT NULL DEFAULT '[]',
    weights TEXT NOT NULL DEFAULT '{}',
    aux_features TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_memory_pairs_user_session_timestamp
    ON memory_pairs (user_id, session_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_memory_pairs_user_kg_timestamp
    ON memory_pairs (user_id, kg_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_subjects_user_kg_key
    ON subjects (user_id, kg_id, is_key_subject, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_subject_memory_pair_links_pair
    ON subject_memory_pair_links (pair_id);

CREATE INDEX IF NOT EXISTS idx_subject_memory_pair_links_user_kg
    ON subject_memory_pair_links (user_id, kg_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_events_user_created
    ON retrieval_events (user_id, created_at DESC);
