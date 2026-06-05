CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS harness_users (
    id BIGSERIAL PRIMARY KEY,
    uid TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memory_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    firestore_pair_id TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES harness_users(uid) ON UPDATE CASCADE ON DELETE CASCADE,
    kg_id TEXT NOT NULL,
    session_id TEXT,
    title TEXT,
    description TEXT,
    prompt TEXT,
    response TEXT,
    input JSONB,
    output JSONB,
    source TEXT,
    source_context TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    timezone_offset INTEGER,
    seed_memories JSONB,
    retrieval_metadata JSONB,
    conversation_embedding vector(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, firestore_pair_id)
);

CREATE TABLE IF NOT EXISTS subjects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES harness_users(uid) ON UPDATE CASCADE ON DELETE CASCADE,
    kg_id TEXT NOT NULL,
    subject_text TEXT NOT NULL,
    description_text TEXT,
    is_key_subject BOOLEAN NOT NULL DEFAULT FALSE,
    embedding vector(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, kg_id, subject_text)
);

CREATE TABLE IF NOT EXISTS subject_memory_pair_links (
    subject_id UUID NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    pair_id UUID NOT NULL REFERENCES memory_pairs(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES harness_users(uid) ON UPDATE CASCADE ON DELETE CASCADE,
    kg_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (subject_id, pair_id)
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    kg_id TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    request_path TEXT NOT NULL DEFAULT '',
    query TEXT NOT NULL DEFAULT '',
    query_embedding vector(768),
    retrieved_pair_ids TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    weights JSONB NOT NULL DEFAULT '{}'::JSONB,
    aux_features JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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

CREATE INDEX IF NOT EXISTS idx_memory_pairs_conversation_embedding_hnsw
    ON memory_pairs USING hnsw (conversation_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_subjects_embedding_hnsw
    ON subjects USING hnsw (embedding vector_cosine_ops);
