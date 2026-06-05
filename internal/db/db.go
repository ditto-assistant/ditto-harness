// Package db adapts generated sqlc queries into the stable internal interface
// used by the harness packages.
package db

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/ditto-assistant/ditto-harness/internal/sqlc"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/pgvector/pgvector-go"
)

type DBTX interface {
	Exec(context.Context, string, ...any) (pgconn.CommandTag, error)
	Query(context.Context, string, ...any) (pgx.Rows, error)
	QueryRow(context.Context, string, ...any) pgx.Row
}

type Queries struct {
	raw *sqlc.Queries
	db  DBTX
}

func New(db DBTX) *Queries {
	return &Queries{raw: sqlc.New(db), db: db}
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
	return q.raw.UpsertUser(ctx, uid)
}

func (q *Queries) CreateMemoryPair(ctx context.Context, p CreateMemoryPairParams) (MemoryPair, error) {
	if p.Timestamp.IsZero() {
		p.Timestamp = time.Now().UTC()
	}
	row, err := q.raw.CreateMemoryPair(ctx, sqlc.CreateMemoryPairParams{
		FirestorePairID:       p.FirestorePairID,
		UserID:                p.UserID,
		KgID:                  p.KgID,
		SessionID:             nullStringArg(p.SessionID),
		Title:                 nullStringArg(p.Title),
		Description:           nullStringArg(p.Description),
		Prompt:                nullStringArg(p.Prompt),
		Response:              nullStringArg(p.Response),
		Input:                 nilIfEmptyBytes(p.Input),
		Output:                nilIfEmptyBytes(p.Output),
		Source:                nullStringArg(p.Source),
		SourceContext:         nullStringArg(p.SourceContext),
		Timestamp:             pgtype.Timestamptz{Time: p.Timestamp, Valid: true},
		TimezoneOffset:        int4Value(p.TimezoneOffset),
		SeedMemories:          nilIfEmptyBytes(p.SeedMemories),
		RetrievalMetadata:     nilIfEmptyBytes(p.RetrievalMetadata),
		ConversationEmbedding: vectorValue(p.ConversationEmbedding),
	})
	if err != nil {
		return MemoryPair{}, err
	}
	return memoryPairFromCreate(row)
}

func (q *Queries) UpsertSubject(ctx context.Context, p UpsertSubjectParams) (Subject, error) {
	row, err := q.raw.UpsertSubject(ctx, sqlc.UpsertSubjectParams{
		UserID:          p.UserID,
		KgID:            p.KgID,
		SubjectText:     p.Text,
		DescriptionText: nullStringArg(p.Description),
		IsKeySubject:    p.IsKeySubject,
		Embedding:       vectorValue(p.Embedding),
	})
	if err != nil {
		return Subject{}, err
	}
	return subjectFromUpsert(row), nil
}

func (q *Queries) LinkSubjectMemoryPair(ctx context.Context, p LinkSubjectMemoryPairParams) error {
	return q.raw.LinkSubjectMemoryPair(ctx, sqlc.LinkSubjectMemoryPairParams{
		SubjectID: pgUUID(p.SubjectID),
		PairID:    pgUUID(p.PairID),
		UserID:    p.UserID,
		KgID:      p.KgID,
	})
}

func (q *Queries) FetchMemories(ctx context.Context, userID string, pairIDs []string) ([]MemoryPair, error) {
	rows, err := q.raw.FetchMemories(ctx, sqlc.FetchMemoriesParams{UserID: userID, PairIds: pairIDs})
	if err != nil {
		return nil, err
	}
	out := make([]MemoryPair, 0, len(rows))
	for _, row := range rows {
		out = append(out, memoryPairFromFetch(row))
	}
	return out, nil
}

func (q *Queries) SearchMemories(ctx context.Context, p SearchMemoriesParams) ([]ScoredMemoryPair, error) {
	rows, err := q.raw.SearchMemories(ctx, sqlc.SearchMemoriesParams{
		Embedding:      vectorValue(p.Embedding),
		UserID:         p.UserID,
		KgID:           p.KgID,
		SessionID:      p.SessionID,
		ExcludePairIds: p.ExcludePairIDs,
		MinSimilarity:  p.MinSimilarity,
		LimitCount:     p.Limit,
	})
	if err != nil {
		return nil, err
	}
	out := make([]ScoredMemoryPair, 0, len(rows))
	for _, row := range rows {
		out = append(out, ScoredMemoryPair{MemoryPair: memoryPairFromSearch(row), Similarity: row.Similarity})
	}
	return out, nil
}

func (q *Queries) SearchSubjects(ctx context.Context, p SearchSubjectsParams) ([]Subject, error) {
	rows, err := q.raw.SearchSubjects(ctx, sqlc.SearchSubjectsParams{
		Embedding:     vectorValue(p.Embedding),
		UserID:        p.UserID,
		KgID:          p.KgID,
		MinSimilarity: p.MinSimilarity,
		LimitCount:    p.Limit,
	})
	if err != nil {
		return nil, err
	}
	out := make([]Subject, 0, len(rows))
	for _, row := range rows {
		out = append(out, subjectFromSearch(row))
	}
	return out, nil
}

func (q *Queries) SearchMemoriesBySubject(ctx context.Context, p SearchMemoriesBySubjectParams) ([]ScoredMemoryPair, error) {
	rows, err := q.raw.SearchMemoriesBySubject(ctx, sqlc.SearchMemoriesBySubjectParams{
		Embedding:     vectorValue(p.Embedding),
		SubjectID:     pgUUID(p.SubjectID),
		UserID:        p.UserID,
		MinSimilarity: p.MinSimilarity,
		LimitCount:    p.Limit,
	})
	if err != nil {
		return nil, err
	}
	out := make([]ScoredMemoryPair, 0, len(rows))
	for _, row := range rows {
		out = append(out, ScoredMemoryPair{MemoryPair: memoryPairFromSubjectSearch(row), Similarity: row.Similarity})
	}
	return out, nil
}

func memoryPairFromCreate(row sqlc.CreateMemoryPairRow) (MemoryPair, error) {
	return MemoryPair{
		ID:                    uuidFromPG(row.ID),
		FirestorePairID:       row.FirestorePairID,
		UserID:                row.UserID,
		KgID:                  row.KgID,
		SessionID:             row.SessionID.String,
		Title:                 row.Title.String,
		Description:           row.Description.String,
		Prompt:                row.Prompt.String,
		Response:              row.Response.String,
		Input:                 row.Input,
		Output:                row.Output,
		Source:                row.Source.String,
		SourceContext:         row.SourceContext.String,
		Timestamp:             timeFromPG(row.Timestamp),
		TimezoneOffset:        int32Ptr(row.TimezoneOffset),
		SeedMemories:          row.SeedMemories,
		RetrievalMetadata:     row.RetrievalMetadata,
		ConversationEmbedding: row.ConversationEmbedding.Slice(),
	}, nil
}

func memoryPairFromFetch(row sqlc.FetchMemoriesRow) MemoryPair {
	return MemoryPair{
		ID:                    uuidFromPG(row.ID),
		FirestorePairID:       row.FirestorePairID,
		UserID:                row.UserID,
		KgID:                  row.KgID,
		SessionID:             row.SessionID.String,
		Title:                 row.Title.String,
		Description:           row.Description.String,
		Prompt:                row.Prompt.String,
		Response:              row.Response.String,
		Input:                 row.Input,
		Output:                row.Output,
		Source:                row.Source.String,
		SourceContext:         row.SourceContext.String,
		Timestamp:             timeFromPG(row.Timestamp),
		TimezoneOffset:        int32Ptr(row.TimezoneOffset),
		SeedMemories:          row.SeedMemories,
		RetrievalMetadata:     row.RetrievalMetadata,
		ConversationEmbedding: row.ConversationEmbedding.Slice(),
	}
}

func memoryPairFromSearch(row sqlc.SearchMemoriesRow) MemoryPair {
	return MemoryPair{
		ID:                    uuidFromPG(row.ID),
		FirestorePairID:       row.FirestorePairID,
		UserID:                row.UserID,
		KgID:                  row.KgID,
		SessionID:             row.SessionID.String,
		Title:                 row.Title.String,
		Description:           row.Description.String,
		Prompt:                row.Prompt.String,
		Response:              row.Response.String,
		Input:                 row.Input,
		Output:                row.Output,
		Source:                row.Source.String,
		SourceContext:         row.SourceContext.String,
		Timestamp:             timeFromPG(row.Timestamp),
		TimezoneOffset:        int32Ptr(row.TimezoneOffset),
		SeedMemories:          row.SeedMemories,
		RetrievalMetadata:     row.RetrievalMetadata,
		ConversationEmbedding: row.ConversationEmbedding.Slice(),
	}
}

func memoryPairFromSubjectSearch(row sqlc.SearchMemoriesBySubjectRow) MemoryPair {
	return MemoryPair{
		ID:                    uuidFromPG(row.ID),
		FirestorePairID:       row.FirestorePairID,
		UserID:                row.UserID,
		KgID:                  row.KgID,
		SessionID:             row.SessionID.String,
		Title:                 row.Title.String,
		Description:           row.Description.String,
		Prompt:                row.Prompt.String,
		Response:              row.Response.String,
		Input:                 row.Input,
		Output:                row.Output,
		Source:                row.Source.String,
		SourceContext:         row.SourceContext.String,
		Timestamp:             timeFromPG(row.Timestamp),
		TimezoneOffset:        int32Ptr(row.TimezoneOffset),
		SeedMemories:          row.SeedMemories,
		RetrievalMetadata:     row.RetrievalMetadata,
		ConversationEmbedding: row.ConversationEmbedding.Slice(),
	}
}

func subjectFromUpsert(row sqlc.UpsertSubjectRow) Subject {
	return Subject{
		ID:           uuidFromPG(row.ID),
		UserID:       row.UserID,
		KgID:         row.KgID,
		Text:         row.SubjectText,
		Description:  row.DescriptionText.String,
		IsKeySubject: row.IsKeySubject,
		Embedding:    row.Embedding.Slice(),
	}
}

func subjectFromSearch(row sqlc.SearchSubjectsRow) Subject {
	return Subject{
		ID:              uuidFromPG(row.ID),
		UserID:          row.UserID,
		KgID:            row.KgID,
		Text:            row.SubjectText,
		Description:     row.DescriptionText.String,
		IsKeySubject:    row.IsKeySubject,
		Embedding:       row.Embedding.Slice(),
		Similarity:      row.Similarity,
		MemoryPairCount: row.MemoryCount,
	}
}

func nilIfEmptyBytes(raw []byte) []byte {
	if len(raw) == 0 {
		return nil
	}
	return raw
}

func nullStringArg(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func vectorValue(v []float32) pgvector.Vector {
	if len(v) == 0 {
		return pgvector.Vector{}
	}
	if len(v) != 768 {
		panic(fmt.Sprintf("embedding dimension = %d, want 768", len(v)))
	}
	return pgvector.NewVector(v)
}

func pgUUID(id uuid.UUID) pgtype.UUID {
	return pgtype.UUID{Bytes: id, Valid: true}
}

func uuidFromPG(id pgtype.UUID) uuid.UUID {
	if !id.Valid {
		return uuid.Nil
	}
	return uuid.UUID(id.Bytes)
}

func int4Value(v *int32) pgtype.Int4 {
	if v == nil {
		return pgtype.Int4{}
	}
	return pgtype.Int4{Int32: *v, Valid: true}
}

func int32Ptr(v pgtype.Int4) *int32 {
	if !v.Valid {
		return nil
	}
	out := v.Int32
	return &out
}

func timeFromPG(v pgtype.Timestamptz) time.Time {
	if !v.Valid {
		return time.Time{}
	}
	return v.Time
}
