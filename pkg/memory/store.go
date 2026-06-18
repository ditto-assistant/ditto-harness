// SPDX-License-Identifier: AGPL-3.0-or-later
package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/ditto-assistant/ditto-harness/pkg/db"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/ditto-assistant/ditto-harness/pkg/retrieval"
	"github.com/google/uuid"
	"github.com/omniaura/go-kit/convert/sqlconv/pgconv/pgdecode"
	"github.com/omniaura/go-kit/convert/sqlconv/pgconv/pgencode"
	"github.com/pgvector/pgvector-go"
)

var ErrNoEmbedder = errors.New("memory: embedder is required")

type Store struct {
	q         *db.Queries
	embedder  harness.Embedder
	predictor retrieval.WeightPredictor
}

type Options struct {
	Queries   *db.Queries
	Embedder  harness.Embedder
	Predictor retrieval.WeightPredictor
}

func NewStore(opts Options) *Store {
	return &Store{q: opts.Queries, embedder: opts.Embedder, predictor: opts.Predictor}
}

type SaveMemoryRequest struct {
	UserID            string
	KGID              string
	SessionID         string
	ID                string
	Title             string
	Summary           string
	Prompt            string
	Response          string
	Input             []harness.Content
	Output            []harness.Content
	Source            string
	SourceContext     string
	Timestamp         time.Time
	TimezoneOffset    int
	SeedMemories      []harness.SeedMemoryNode
	RetrievalMetadata *harness.RetrievalMetadata
	Subjects          []SubjectInput
}

type SubjectInput struct {
	Text        string
	Description string
	Key         bool
}

type SearchMemoriesRequest struct {
	UserID         string
	KGID           string
	SessionID      string
	Queries        []string
	Limit          int
	MinSimilarity  float64
	ExcludePairIDs []string
}

type CompositeSearchRequest struct {
	UserID            string
	KGID              string
	SessionID         string
	Query             string
	Limit             int
	CandidatePoolSize int
	ExcludePairIDs    []string
	Variant           retrieval.Variant
	RequestPath       string
	LogEvent          bool
}

type SearchSubjectsRequest struct {
	UserID        string
	KGID          string
	Queries       []string
	Limit         int
	MinSimilarity float64
}

type SubjectMemoryQuery struct {
	SubjectID string `json:"subject_id"`
	Query     string `json:"query"`
}

type SearchMemoriesInSubjectsRequest struct {
	UserID        string
	Queries       []SubjectMemoryQuery
	Limit         int
	MinSimilarity float64
}

type FetchMemoriesRequest struct {
	UserID  string
	PairIDs []string
}

func (s *Store) SaveMemory(ctx context.Context, req SaveMemoryRequest) (harness.Memory, error) {
	if s.q == nil {
		return harness.Memory{}, errors.New("memory: queries are required")
	}
	if strings.TrimSpace(req.UserID) == "" {
		return harness.Memory{}, errors.New("memory: user id is required")
	}
	if req.KGID == "" {
		req.KGID = harness.KGID(req.UserID)
	}
	if req.ID == "" {
		req.ID = uuid.NewString()
	}
	if req.Timestamp.IsZero() {
		req.Timestamp = time.Now().UTC()
	}

	text := strings.TrimSpace(req.Prompt + "\n" + req.Response + "\n" + req.Summary)
	embedResp, err := s.embedTexts(ctx, []string{text})
	if err != nil {
		return harness.Memory{}, err
	}
	embedding := firstEmbedding(embedResp)

	inputJSON, err := json.Marshal(req.Input)
	if err != nil {
		return harness.Memory{}, fmt.Errorf("marshal input: %w", err)
	}
	outputJSON, err := json.Marshal(req.Output)
	if err != nil {
		return harness.Memory{}, fmt.Errorf("marshal output: %w", err)
	}
	seedJSON, err := json.Marshal(req.SeedMemories)
	if err != nil {
		return harness.Memory{}, fmt.Errorf("marshal seed memories: %w", err)
	}
	metadataJSON, err := json.Marshal(req.RetrievalMetadata)
	if err != nil {
		return harness.Memory{}, fmt.Errorf("marshal retrieval metadata: %w", err)
	}

	if err := s.q.UpsertUser(ctx, req.UserID); err != nil {
		return harness.Memory{}, fmt.Errorf("upsert user: %w", err)
	}
	offset := int32(req.TimezoneOffset)
	timezoneOffset, err := pgencode.Int32Ptr(&offset).Int4()
	if err != nil {
		return harness.Memory{}, fmt.Errorf("encode timezone offset: %w", err)
	}
	row, err := s.q.CreateMemoryPair(ctx, db.CreateMemoryPairParams{
		FirestorePairID:       req.ID,
		UserID:                req.UserID,
		KgID:                  req.KGID,
		SessionID:             req.SessionID,
		Title:                 req.Title,
		Description:           req.Summary,
		Prompt:                req.Prompt,
		Response:              req.Response,
		Input:                 inputJSON,
		Output:                outputJSON,
		Source:                req.Source,
		SourceContext:         req.SourceContext,
		Timestamp:             pgencode.Time(req.Timestamp).Timestamptz(),
		TimezoneOffset:        timezoneOffset,
		SeedMemories:          seedJSON,
		RetrievalMetadata:     metadataJSON,
		ConversationEmbedding: vectorValue(embedding),
	})
	if err != nil {
		return harness.Memory{}, fmt.Errorf("create memory pair: %w", err)
	}

	if len(req.Subjects) > 0 {
		subjectTexts := make([]string, len(req.Subjects))
		for i, subj := range req.Subjects {
			subjectTexts[i] = strings.TrimSpace(subj.Text + "\n" + subj.Description)
		}
		subjectEmbeddings, err := s.embedTexts(ctx, subjectTexts)
		if err != nil {
			return harness.Memory{}, fmt.Errorf("embed subjects: %w", err)
		}
		for i, subj := range req.Subjects {
			if strings.TrimSpace(subj.Text) == "" {
				continue
			}
			srow, err := s.q.UpsertSubject(ctx, db.UpsertSubjectParams{
				UserID:          req.UserID,
				KgID:            req.KGID,
				SubjectText:     subj.Text,
				DescriptionText: pgencode.String(subj.Description).EmptyIsNull().Text(),
				IsKeySubject:    subj.Key,
				Embedding:       vectorValue(embeddingAt(subjectEmbeddings, i)),
			})
			if err != nil {
				return harness.Memory{}, fmt.Errorf("upsert subject %q: %w", subj.Text, err)
			}
			if err := s.q.LinkSubjectMemoryPair(ctx, db.LinkSubjectMemoryPairParams{
				SubjectID: srow.ID,
				PairID:    row.ID,
				UserID:    req.UserID,
				KgID:      req.KGID,
			}); err != nil {
				return harness.Memory{}, fmt.Errorf("link subject %q: %w", subj.Text, err)
			}
		}
	}

	return memoryFromCreate(row), nil
}

func (s *Store) SearchMemories(ctx context.Context, req SearchMemoriesRequest) ([]harness.Memory, error) {
	if req.KGID == "" {
		req.KGID = harness.KGID(req.UserID)
	}
	if req.Limit <= 0 {
		req.Limit = 8
	}
	if req.MinSimilarity == 0 {
		req.MinSimilarity = 0.15
	}

	embeddings, err := s.embedTexts(ctx, req.Queries)
	if err != nil {
		return nil, err
	}
	seen := make(map[string]struct{}, len(req.ExcludePairIDs))
	for _, id := range req.ExcludePairIDs {
		seen[id] = struct{}{}
	}
	var out []harness.Memory
	for _, embedding := range embeddings.Embeddings {
		rows, err := s.q.SearchMemories(ctx, db.SearchMemoriesParams{
			Embedding:      vectorValue(embedding),
			UserID:         req.UserID,
			KgID:           req.KGID,
			SessionID:      req.SessionID,
			ExcludePairIds: mapKeys(seen),
			MinSimilarity:  req.MinSimilarity,
			LimitCount:     int32(req.Limit),
		})
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			mem := memoryFromSearch(row)
			mem.Similarity = row.Similarity
			if _, ok := seen[mem.ID]; ok {
				continue
			}
			seen[mem.ID] = struct{}{}
			out = append(out, mem)
		}
	}
	return out, nil
}

func (s *Store) SearchCompositeMemories(ctx context.Context, req CompositeSearchRequest) ([]harness.Memory, *harness.RetrievalMetadata, error) {
	if s.q == nil {
		return nil, nil, errors.New("memory: queries are required")
	}
	if req.KGID == "" {
		req.KGID = harness.KGID(req.UserID)
	}
	if req.Limit <= 0 {
		req.Limit = 8
	}
	if req.CandidatePoolSize <= 0 {
		req.CandidatePoolSize = max(32, req.Limit*4)
	}
	if req.Variant == "" {
		req.Variant = retrieval.VariantLegacy
	}
	embedResp, err := s.embedTexts(ctx, []string{req.Query})
	if err != nil {
		return nil, nil, err
	}
	embedding := firstEmbedding(embedResp)
	weights := retrieval.DefaultWeights()
	intent := "semantic"
	if s.predictor != nil {
		predicted, err := s.predictor.Predict(ctx, retrieval.Features{
			Query:            req.Query,
			Now:              time.Now().UTC(),
			QueryEmbedding:   embedding,
			CurrentSessionID: req.SessionID,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("predict retrieval weights: %w", err)
		}
		weights = predicted
		intent = "learned"
	}
	results, err := retrieval.CompositeRetrieve(ctx, s.q.DB(), retrieval.CompositeParams{
		Embedding:         embedding,
		UserID:            req.UserID,
		KGID:              req.KGID,
		SessionID:         req.SessionID,
		CurrentSessionID:  req.SessionID,
		Limit:             req.Limit,
		CandidatePoolSize: req.CandidatePoolSize,
		ExcludePairIDs:    req.ExcludePairIDs,
		Weights:           weights,
		Variant:           req.Variant,
		RequestPath:       req.RequestPath,
		Query:             req.Query,
		LogEvent:          req.LogEvent,
	})
	if err != nil {
		return nil, nil, err
	}
	pairIDs := make([]string, 0, len(results))
	scoreByID := make(map[string]retrieval.CompositeMemory, len(results))
	for _, result := range results {
		pairIDs = append(pairIDs, result.PairID)
		scoreByID[result.PairID] = result
	}
	memories, err := s.FetchMemories(ctx, FetchMemoriesRequest{UserID: req.UserID, PairIDs: pairIDs})
	if err != nil {
		return nil, nil, err
	}
	for i := range memories {
		score := scoreByID[memories[i].ID]
		memories[i].Similarity = score.CosineSimilarity
		memories[i].RecencyScore = score.RecencyScore
		memories[i].FrequencyScore = score.FrequencyScore
		memories[i].CompositeScore = score.CompositeScore
		memories[i].RecencyExp = score.RecencyExp
		memories[i].SubjectSemMatch = score.SubjectSemMatch
		memories[i].SessionContinuity = score.SessionContinuity
		memories[i].NeighborDensity = score.NeighborDensity
	}
	metadata := &harness.RetrievalMetadata{
		Intent: intent,
		Weights: map[string]float64{
			"cosine":            weights.Cosine,
			"recencyLinear":     weights.RecencyLinear,
			"recencyExp":        weights.RecencyExp,
			"subjectFrequency":  weights.SubjectFrequency,
			"subjectSemMatch":   weights.SubjectSemMatch,
			"sessionContinuity": weights.SessionContinuity,
			"neighborDensity":   weights.NeighborDensity,
			"scale":             weights.Scale,
		},
		Scale:               weights.Scale,
		Variant:             string(req.Variant),
		RetrievedPairIDs:    pairIDs,
		QueryEmbeddingModel: "host",
	}
	return memories, metadata, nil
}

func (s *Store) SearchSubjects(ctx context.Context, req SearchSubjectsRequest) ([]harness.Subject, error) {
	if req.KGID == "" {
		req.KGID = harness.KGID(req.UserID)
	}
	if req.Limit <= 0 {
		req.Limit = 8
	}
	if req.MinSimilarity == 0 {
		req.MinSimilarity = 0.10
	}
	embeddings, err := s.embedTexts(ctx, req.Queries)
	if err != nil {
		return nil, err
	}
	seen := map[string]struct{}{}
	var out []harness.Subject
	for _, embedding := range embeddings.Embeddings {
		rows, err := s.q.SearchSubjects(ctx, db.SearchSubjectsParams{
			Embedding:     vectorValue(embedding),
			UserID:        req.UserID,
			KgID:          req.KGID,
			MinSimilarity: req.MinSimilarity,
			LimitCount:    int32(req.Limit),
		})
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			subj := subjectFromSearch(row)
			if _, ok := seen[subj.ID]; ok {
				continue
			}
			seen[subj.ID] = struct{}{}
			out = append(out, subj)
		}
	}
	return out, nil
}

func (s *Store) SearchMemoriesInSubjects(ctx context.Context, req SearchMemoriesInSubjectsRequest) ([]harness.Memory, error) {
	if req.Limit <= 0 {
		req.Limit = 8
	}
	if req.MinSimilarity == 0 {
		req.MinSimilarity = 0.15
	}
	texts := make([]string, len(req.Queries))
	for i, query := range req.Queries {
		texts[i] = query.Query
	}
	embeddings, err := s.embedTexts(ctx, texts)
	if err != nil {
		return nil, err
	}
	seen := map[string]struct{}{}
	var out []harness.Memory
	for i, query := range req.Queries {
		subjectID, err := uuid.Parse(query.SubjectID)
		if err != nil {
			return nil, fmt.Errorf("parse subject id %q: %w", query.SubjectID, err)
		}
		rows, err := s.q.SearchMemoriesBySubject(ctx, db.SearchMemoriesBySubjectParams{
			Embedding:     vectorValue(embeddingAt(embeddings, i)),
			SubjectID:     pgencode.UUID(subjectID).UUID(),
			UserID:        req.UserID,
			MinSimilarity: req.MinSimilarity,
			LimitCount:    int32(req.Limit),
		})
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			mem := memoryFromSubjectSearch(row)
			mem.Similarity = row.Similarity
			if _, ok := seen[mem.ID]; ok {
				continue
			}
			seen[mem.ID] = struct{}{}
			out = append(out, mem)
		}
	}
	return out, nil
}

func (s *Store) FetchMemories(ctx context.Context, req FetchMemoriesRequest) ([]harness.Memory, error) {
	rows, err := s.q.FetchMemories(ctx, db.FetchMemoriesParams{UserID: req.UserID, PairIds: req.PairIDs})
	if err != nil {
		return nil, err
	}
	out := make([]harness.Memory, 0, len(rows))
	for _, row := range rows {
		out = append(out, memoryFromFetch(row))
	}
	return out, nil
}

func (s *Store) embedTexts(ctx context.Context, texts []string) (harness.EmbedResponse, error) {
	if s.embedder == nil {
		return harness.EmbedResponse{}, ErrNoEmbedder
	}
	clean := make([]string, 0, len(texts))
	for _, text := range texts {
		if strings.TrimSpace(text) != "" {
			clean = append(clean, text)
		}
	}
	if len(clean) == 0 {
		return harness.EmbedResponse{}, errors.New("memory: at least one non-empty query is required")
	}
	return s.embedder.Embed(ctx, harness.EmbedRequest{Texts: clean})
}

type memoryRow struct {
	id                    string
	sourcePairID          string
	userID                string
	kgID                  string
	sessionID             string
	title                 string
	description           string
	prompt                string
	response              string
	input                 []byte
	output                []byte
	source                string
	sourceContext         string
	timestamp             time.Time
	timezoneOffset        int
	seedMemories          []byte
	retrievalMetadata     []byte
	conversationEmbedding []float32
}

func memoryFromCreate(row db.CreateMemoryPairRow) harness.Memory {
	return memoryFromRow(memoryRow{
		id:                    row.FirestorePairID,
		sourcePairID:          pgdecode.UUID(row.ID).String(),
		userID:                row.UserID,
		kgID:                  row.KgID,
		sessionID:             pgdecode.Text(row.SessionID).Value(),
		title:                 pgdecode.Text(row.Title).Value(),
		description:           pgdecode.Text(row.Description).Value(),
		prompt:                pgdecode.Text(row.Prompt).Value(),
		response:              pgdecode.Text(row.Response).Value(),
		input:                 row.Input,
		output:                row.Output,
		source:                pgdecode.Text(row.Source).Value(),
		sourceContext:         pgdecode.Text(row.SourceContext).Value(),
		timestamp:             pgdecode.Timestamptz(row.Timestamp).Value(),
		timezoneOffset:        int(pgdecode.Int4(row.TimezoneOffset).Value()),
		seedMemories:          row.SeedMemories,
		retrievalMetadata:     row.RetrievalMetadata,
		conversationEmbedding: row.ConversationEmbedding.Slice(),
	})
}

func memoryFromFetch(row db.FetchMemoriesRow) harness.Memory {
	return memoryFromRow(memoryRow{
		id:                    row.FirestorePairID,
		sourcePairID:          pgdecode.UUID(row.ID).String(),
		userID:                row.UserID,
		kgID:                  row.KgID,
		sessionID:             pgdecode.Text(row.SessionID).Value(),
		title:                 pgdecode.Text(row.Title).Value(),
		description:           pgdecode.Text(row.Description).Value(),
		prompt:                pgdecode.Text(row.Prompt).Value(),
		response:              pgdecode.Text(row.Response).Value(),
		input:                 row.Input,
		output:                row.Output,
		source:                pgdecode.Text(row.Source).Value(),
		sourceContext:         pgdecode.Text(row.SourceContext).Value(),
		timestamp:             pgdecode.Timestamptz(row.Timestamp).Value(),
		timezoneOffset:        int(pgdecode.Int4(row.TimezoneOffset).Value()),
		seedMemories:          row.SeedMemories,
		retrievalMetadata:     row.RetrievalMetadata,
		conversationEmbedding: row.ConversationEmbedding.Slice(),
	})
}

func memoryFromRecent(row db.ListRecentMemoriesRow) harness.Memory {
	return memoryFromRow(memoryRow{
		id:                    row.FirestorePairID,
		sourcePairID:          pgdecode.UUID(row.ID).String(),
		userID:                row.UserID,
		kgID:                  row.KgID,
		sessionID:             pgdecode.Text(row.SessionID).Value(),
		title:                 pgdecode.Text(row.Title).Value(),
		description:           pgdecode.Text(row.Description).Value(),
		prompt:                pgdecode.Text(row.Prompt).Value(),
		response:              pgdecode.Text(row.Response).Value(),
		input:                 row.Input,
		output:                row.Output,
		source:                pgdecode.Text(row.Source).Value(),
		sourceContext:         pgdecode.Text(row.SourceContext).Value(),
		timestamp:             pgdecode.Timestamptz(row.Timestamp).Value(),
		timezoneOffset:        int(pgdecode.Int4(row.TimezoneOffset).Value()),
		seedMemories:          row.SeedMemories,
		retrievalMetadata:     row.RetrievalMetadata,
		conversationEmbedding: row.ConversationEmbedding.Slice(),
	})
}

func memoryFromSearch(row db.SearchMemoriesRow) harness.Memory {
	return memoryFromRow(memoryRow{
		id:                    row.FirestorePairID,
		sourcePairID:          pgdecode.UUID(row.ID).String(),
		userID:                row.UserID,
		kgID:                  row.KgID,
		sessionID:             pgdecode.Text(row.SessionID).Value(),
		title:                 pgdecode.Text(row.Title).Value(),
		description:           pgdecode.Text(row.Description).Value(),
		prompt:                pgdecode.Text(row.Prompt).Value(),
		response:              pgdecode.Text(row.Response).Value(),
		input:                 row.Input,
		output:                row.Output,
		source:                pgdecode.Text(row.Source).Value(),
		sourceContext:         pgdecode.Text(row.SourceContext).Value(),
		timestamp:             pgdecode.Timestamptz(row.Timestamp).Value(),
		timezoneOffset:        int(pgdecode.Int4(row.TimezoneOffset).Value()),
		seedMemories:          row.SeedMemories,
		retrievalMetadata:     row.RetrievalMetadata,
		conversationEmbedding: row.ConversationEmbedding.Slice(),
	})
}

func memoryFromSubjectSearch(row db.SearchMemoriesBySubjectRow) harness.Memory {
	return memoryFromRow(memoryRow{
		id:                    row.FirestorePairID,
		sourcePairID:          pgdecode.UUID(row.ID).String(),
		userID:                row.UserID,
		kgID:                  row.KgID,
		sessionID:             pgdecode.Text(row.SessionID).Value(),
		title:                 pgdecode.Text(row.Title).Value(),
		description:           pgdecode.Text(row.Description).Value(),
		prompt:                pgdecode.Text(row.Prompt).Value(),
		response:              pgdecode.Text(row.Response).Value(),
		input:                 row.Input,
		output:                row.Output,
		source:                pgdecode.Text(row.Source).Value(),
		sourceContext:         pgdecode.Text(row.SourceContext).Value(),
		timestamp:             pgdecode.Timestamptz(row.Timestamp).Value(),
		timezoneOffset:        int(pgdecode.Int4(row.TimezoneOffset).Value()),
		seedMemories:          row.SeedMemories,
		retrievalMetadata:     row.RetrievalMetadata,
		conversationEmbedding: row.ConversationEmbedding.Slice(),
	})
}

func memoryFromRow(row memoryRow) harness.Memory {
	var input, output []harness.Content
	var seed []harness.SeedMemoryNode
	var metadata *harness.RetrievalMetadata
	_ = json.Unmarshal(row.input, &input)
	_ = json.Unmarshal(row.output, &output)
	_ = json.Unmarshal(row.seedMemories, &seed)
	if len(row.retrievalMetadata) > 0 && string(row.retrievalMetadata) != "null" {
		var md harness.RetrievalMetadata
		if json.Unmarshal(row.retrievalMetadata, &md) == nil {
			metadata = &md
		}
	}
	return harness.Memory{
		ID:                row.id,
		SourcePairID:      row.sourcePairID,
		UserID:            row.userID,
		KGID:              row.kgID,
		SessionID:         row.sessionID,
		Title:             row.title,
		Summary:           row.description,
		Prompt:            row.prompt,
		Response:          row.response,
		Input:             input,
		Output:            output,
		Source:            row.source,
		SourceContext:     row.sourceContext,
		Timestamp:         row.timestamp,
		TimezoneOffset:    row.timezoneOffset,
		SeedMemories:      seed,
		RetrievalMetadata: metadata,
		Embedding:         row.conversationEmbedding,
	}
}

func subjectFromSearch(row db.SearchSubjectsRow) harness.Subject {
	return harness.Subject{
		ID:          pgdecode.UUID(row.ID).String(),
		UserID:      row.UserID,
		KGID:        row.KgID,
		Text:        row.SubjectText,
		Description: pgdecode.Text(row.DescriptionText).Value(),
		Key:         row.IsKeySubject,
		Embedding:   row.Embedding.Slice(),
		Similarity:  row.Similarity,
		MemoryCount: row.MemoryCount,
	}
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

func firstEmbedding(resp harness.EmbedResponse) []float32 {
	return embeddingAt(resp, 0)
}

func embeddingAt(resp harness.EmbedResponse, idx int) []float32 {
	if idx < 0 || idx >= len(resp.Embeddings) {
		return nil
	}
	return resp.Embeddings[idx]
}

func mapKeys(m map[string]struct{}) []string {
	if len(m) == 0 {
		return nil
	}
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
