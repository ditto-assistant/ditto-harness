// Package db is a small sqlc-compatible query layer for the harness schema.
package db

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/pgvector/pgvector-go"
)

type DBTX interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
	Query(context.Context, string, ...any) (pgx.Rows, error)
	QueryRow(context.Context, string, ...any) pgx.Row
}

type Queries struct {
	db DBTX
}

func New(db DBTX) *Queries {
	return &Queries{db: db}
}

func (q *Queries) DB() DBTX {
	return q.db
}

type MemoryPair struct {
	ID                    uuid.UUID
	FirestorePairID       string
	UserID                string
	KgID                  string
	SessionID             string
	Title                 string
	Description           string
	Prompt                string
	Response              string
	Input                 json.RawMessage
	Output                json.RawMessage
	Source                string
	SourceContext         string
	Timestamp             time.Time
	TimezoneOffset        *int32
	SeedMemories          json.RawMessage
	RetrievalMetadata     json.RawMessage
	ConversationEmbedding []float32
}

type ScoredMemoryPair struct {
	MemoryPair
	Similarity float64
}

type Subject struct {
	ID              uuid.UUID
	UserID          string
	KgID            string
	Text            string
	Description     string
	IsKeySubject    bool
	Embedding       []float32
	Similarity      float64
	MemoryPairCount int64
}

type CreateMemoryPairParams struct {
	FirestorePairID       string
	UserID                string
	KgID                  string
	SessionID             string
	Title                 string
	Description           string
	Prompt                string
	Response              string
	Input                 json.RawMessage
	Output                json.RawMessage
	Source                string
	SourceContext         string
	Timestamp             time.Time
	TimezoneOffset        *int32
	SeedMemories          json.RawMessage
	RetrievalMetadata     json.RawMessage
	ConversationEmbedding []float32
}

type UpsertSubjectParams struct {
	UserID       string
	KgID         string
	Text         string
	Description  string
	IsKeySubject bool
	Embedding    []float32
}

type LinkSubjectMemoryPairParams struct {
	SubjectID uuid.UUID
	PairID    uuid.UUID
	UserID    string
	KgID      string
}

type SearchMemoriesParams struct {
	Embedding      []float32
	UserID         string
	KgID           string
	SessionID      string
	ExcludePairIDs []string
	MinSimilarity  float64
	Limit          int32
}

type SearchSubjectsParams struct {
	Embedding     []float32
	UserID        string
	KgID          string
	MinSimilarity float64
	Limit         int32
}

type SearchMemoriesBySubjectParams struct {
	Embedding     []float32
	SubjectID     uuid.UUID
	UserID        string
	MinSimilarity float64
	Limit         int32
}

func (q *Queries) UpsertUser(ctx context.Context, uid string) error {
	_, err := q.db.Exec(ctx, `INSERT INTO harness_users (uid) VALUES ($1) ON CONFLICT (uid) DO NOTHING`, uid)
	return err
}

func (q *Queries) CreateMemoryPair(ctx context.Context, p CreateMemoryPairParams) (MemoryPair, error) {
	if p.Timestamp.IsZero() {
		p.Timestamp = time.Now().UTC()
	}
	row := q.db.QueryRow(ctx, `
INSERT INTO memory_pairs (
    firestore_pair_id, user_id, kg_id, session_id, title, description,
    prompt, response, input, output, source, source_context, timestamp,
    timezone_offset, seed_memories, retrieval_metadata, conversation_embedding
) VALUES (
    $1, $2, $3, NULLIF($4, ''), NULLIF($5, ''), NULLIF($6, ''),
    NULLIF($7, ''), NULLIF($8, ''), $9, $10, NULLIF($11, ''), NULLIF($12, ''),
    $13, $14, $15, $16, $17
)
ON CONFLICT (user_id, firestore_pair_id) DO UPDATE SET
    kg_id = EXCLUDED.kg_id,
    session_id = EXCLUDED.session_id,
    title = EXCLUDED.title,
    description = EXCLUDED.description,
    prompt = EXCLUDED.prompt,
    response = EXCLUDED.response,
    input = EXCLUDED.input,
    output = EXCLUDED.output,
    source = EXCLUDED.source,
    source_context = EXCLUDED.source_context,
    timestamp = EXCLUDED.timestamp,
    timezone_offset = EXCLUDED.timezone_offset,
    seed_memories = EXCLUDED.seed_memories,
    retrieval_metadata = EXCLUDED.retrieval_metadata,
    conversation_embedding = EXCLUDED.conversation_embedding,
    updated_at = NOW()
RETURNING id, firestore_pair_id, user_id, kg_id, COALESCE(session_id, ''), COALESCE(title, ''),
    COALESCE(description, ''), COALESCE(prompt, ''), COALESCE(response, ''), input, output,
    COALESCE(source, ''), COALESCE(source_context, ''), timestamp, timezone_offset,
    seed_memories, retrieval_metadata, conversation_embedding`,
		p.FirestorePairID, p.UserID, p.KgID, p.SessionID, p.Title, p.Description,
		p.Prompt, p.Response, nullableJSON(p.Input), nullableJSON(p.Output), p.Source, p.SourceContext,
		p.Timestamp, p.TimezoneOffset, nullableJSON(p.SeedMemories), nullableJSON(p.RetrievalMetadata),
		vectorOrNil(p.ConversationEmbedding),
	)
	return scanMemoryPair(row)
}

func (q *Queries) UpsertSubject(ctx context.Context, p UpsertSubjectParams) (Subject, error) {
	row := q.db.QueryRow(ctx, `
INSERT INTO subjects (user_id, kg_id, subject_text, description_text, is_key_subject, embedding)
VALUES ($1, $2, $3, NULLIF($4, ''), $5, $6)
ON CONFLICT (user_id, kg_id, subject_text) DO UPDATE SET
    description_text = COALESCE(EXCLUDED.description_text, subjects.description_text),
    is_key_subject = subjects.is_key_subject OR EXCLUDED.is_key_subject,
    embedding = COALESCE(EXCLUDED.embedding, subjects.embedding),
    updated_at = NOW()
RETURNING id, user_id, kg_id, subject_text, COALESCE(description_text, ''), is_key_subject, embedding`,
		p.UserID, p.KgID, p.Text, p.Description, p.IsKeySubject, vectorOrNil(p.Embedding),
	)
	return scanSubject(row, false)
}

func (q *Queries) LinkSubjectMemoryPair(ctx context.Context, p LinkSubjectMemoryPairParams) error {
	_, err := q.db.Exec(ctx, `
INSERT INTO subject_memory_pair_links (subject_id, pair_id, user_id, kg_id)
VALUES ($1, $2, $3, $4)
ON CONFLICT (subject_id, pair_id) DO NOTHING`, p.SubjectID, p.PairID, p.UserID, p.KgID)
	return err
}

func (q *Queries) FetchMemories(ctx context.Context, userID string, pairIDs []string) ([]MemoryPair, error) {
	rows, err := q.db.Query(ctx, `
SELECT id, firestore_pair_id, user_id, kg_id, COALESCE(session_id, ''), COALESCE(title, ''),
    COALESCE(description, ''), COALESCE(prompt, ''), COALESCE(response, ''), input, output,
    COALESCE(source, ''), COALESCE(source_context, ''), timestamp, timezone_offset,
    seed_memories, retrieval_metadata, conversation_embedding
FROM memory_pairs
WHERE user_id = $1 AND firestore_pair_id = ANY($2::text[])
ORDER BY array_position($2::text[], firestore_pair_id)`, userID, pairIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanMemoryPairs(rows)
}

func (q *Queries) SearchMemories(ctx context.Context, p SearchMemoriesParams) ([]ScoredMemoryPair, error) {
	rows, err := q.db.Query(ctx, `
SELECT id, firestore_pair_id, user_id, kg_id, COALESCE(session_id, ''), COALESCE(title, ''),
    COALESCE(description, ''), COALESCE(prompt, ''), COALESCE(response, ''), input, output,
    COALESCE(source, ''), COALESCE(source_context, ''), timestamp, timezone_offset,
    seed_memories, retrieval_metadata, conversation_embedding,
    (1 - (conversation_embedding <=> $1::vector))::float8 AS similarity
FROM memory_pairs
WHERE user_id = $2 AND kg_id = $3
  AND conversation_embedding IS NOT NULL
  AND ($4::text = '' OR COALESCE(session_id, 'main') = COALESCE(NULLIF($4::text, ''), 'main'))
  AND ($5::text[] IS NULL OR firestore_pair_id != ALL($5::text[]))
  AND (1 - (conversation_embedding <=> $1::vector)) >= $6
ORDER BY conversation_embedding <=> $1::vector, timestamp DESC
LIMIT $7`, pgvector.NewVector(p.Embedding), p.UserID, p.KgID, p.SessionID, nilIfEmptyStrings(p.ExcludePairIDs), p.MinSimilarity, p.Limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []ScoredMemoryPair
	for rows.Next() {
		mem, sim, err := scanScoredMemoryPair(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, ScoredMemoryPair{MemoryPair: mem, Similarity: sim})
	}
	return out, rows.Err()
}

func (q *Queries) SearchSubjects(ctx context.Context, p SearchSubjectsParams) ([]Subject, error) {
	rows, err := q.db.Query(ctx, `
SELECT s.id, s.user_id, s.kg_id, s.subject_text, COALESCE(s.description_text, ''), s.is_key_subject, s.embedding,
    (1 - (s.embedding <=> $1::vector))::float8 AS similarity,
    COUNT(smpl.pair_id)::bigint AS memory_count
FROM subjects s
LEFT JOIN subject_memory_pair_links smpl ON smpl.subject_id = s.id
WHERE s.user_id = $2 AND s.kg_id = $3
  AND s.embedding IS NOT NULL
  AND (1 - (s.embedding <=> $1::vector)) >= $4
GROUP BY s.id
ORDER BY s.embedding <=> $1::vector, memory_count DESC, s.updated_at DESC
LIMIT $5`, pgvector.NewVector(p.Embedding), p.UserID, p.KgID, p.MinSimilarity, p.Limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Subject
	for rows.Next() {
		subj, err := scanSubject(rows, true)
		if err != nil {
			return nil, err
		}
		out = append(out, subj)
	}
	return out, rows.Err()
}

func (q *Queries) SearchMemoriesBySubject(ctx context.Context, p SearchMemoriesBySubjectParams) ([]ScoredMemoryPair, error) {
	rows, err := q.db.Query(ctx, `
SELECT mp.id, mp.firestore_pair_id, mp.user_id, mp.kg_id, COALESCE(mp.session_id, ''), COALESCE(mp.title, ''),
    COALESCE(mp.description, ''), COALESCE(mp.prompt, ''), COALESCE(mp.response, ''), mp.input, mp.output,
    COALESCE(mp.source, ''), COALESCE(mp.source_context, ''), mp.timestamp, mp.timezone_offset,
    mp.seed_memories, mp.retrieval_metadata, mp.conversation_embedding,
    (1 - (mp.conversation_embedding <=> $1::vector))::float8 AS similarity
FROM memory_pairs mp
JOIN subject_memory_pair_links smpl ON smpl.pair_id = mp.id
WHERE smpl.subject_id = $2 AND mp.user_id = $3
  AND mp.conversation_embedding IS NOT NULL
  AND (1 - (mp.conversation_embedding <=> $1::vector)) >= $4
ORDER BY mp.conversation_embedding <=> $1::vector, mp.timestamp DESC
LIMIT $5`, pgvector.NewVector(p.Embedding), p.SubjectID, p.UserID, p.MinSimilarity, p.Limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []ScoredMemoryPair
	for rows.Next() {
		mem, sim, err := scanScoredMemoryPair(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, ScoredMemoryPair{MemoryPair: mem, Similarity: sim})
	}
	return out, rows.Err()
}

func scanMemoryPair(row pgx.Row) (MemoryPair, error) {
	var mem MemoryPair
	var embedding pgvector.Vector
	err := row.Scan(
		&mem.ID, &mem.FirestorePairID, &mem.UserID, &mem.KgID, &mem.SessionID,
		&mem.Title, &mem.Description, &mem.Prompt, &mem.Response,
		&mem.Input, &mem.Output, &mem.Source, &mem.SourceContext, &mem.Timestamp,
		&mem.TimezoneOffset, &mem.SeedMemories, &mem.RetrievalMetadata, &embedding,
	)
	mem.ConversationEmbedding = embedding.Slice()
	return mem, err
}

func scanScoredMemoryPair(row pgx.Row) (MemoryPair, float64, error) {
	var mem MemoryPair
	var embedding pgvector.Vector
	var similarity float64
	err := row.Scan(
		&mem.ID, &mem.FirestorePairID, &mem.UserID, &mem.KgID, &mem.SessionID,
		&mem.Title, &mem.Description, &mem.Prompt, &mem.Response,
		&mem.Input, &mem.Output, &mem.Source, &mem.SourceContext, &mem.Timestamp,
		&mem.TimezoneOffset, &mem.SeedMemories, &mem.RetrievalMetadata, &embedding, &similarity,
	)
	mem.ConversationEmbedding = embedding.Slice()
	return mem, similarity, err
}

func scanMemoryPairs(rows pgx.Rows) ([]MemoryPair, error) {
	var out []MemoryPair
	for rows.Next() {
		mem, err := scanMemoryPair(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, mem)
	}
	return out, rows.Err()
}

func scanSubject(row pgx.Row, withScore bool) (Subject, error) {
	var subj Subject
	var embedding pgvector.Vector
	args := []any{&subj.ID, &subj.UserID, &subj.KgID, &subj.Text, &subj.Description, &subj.IsKeySubject, &embedding}
	if withScore {
		args = append(args, &subj.Similarity, &subj.MemoryPairCount)
	}
	if err := row.Scan(args...); err != nil {
		return Subject{}, err
	}
	subj.Embedding = embedding.Slice()
	return subj, nil
}

func nullableJSON(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	return raw
}

func vectorOrNil(v []float32) any {
	if len(v) == 0 {
		return nil
	}
	if len(v) != 768 {
		panic(fmt.Sprintf("embedding dimension = %d, want 768", len(v)))
	}
	return pgvector.NewVector(v)
}

func nilIfEmptyStrings(v []string) any {
	if len(v) == 0 {
		return nil
	}
	return v
}
