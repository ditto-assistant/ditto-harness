package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/ditto-assistant/ditto-harness/internal/db"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/google/uuid"
)

var ErrNoEmbedder = errors.New("memory: embedder is required")

type Store struct {
	q        *db.Queries
	embedder harness.Embedder
}

type Options struct {
	Queries  *db.Queries
	Embedder harness.Embedder
}

func NewStore(opts Options) *Store {
	return &Store{q: opts.Queries, embedder: opts.Embedder}
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
		Timestamp:             req.Timestamp,
		TimezoneOffset:        &offset,
		SeedMemories:          seedJSON,
		RetrievalMetadata:     metadataJSON,
		ConversationEmbedding: embedding,
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
				UserID:       req.UserID,
				KgID:         req.KGID,
				Text:         subj.Text,
				Description:  subj.Description,
				IsKeySubject: subj.Key,
				Embedding:    embeddingAt(subjectEmbeddings, i),
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

	return memoryFromDB(row), nil
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
			Embedding:      embedding,
			UserID:         req.UserID,
			KgID:           req.KGID,
			SessionID:      req.SessionID,
			ExcludePairIDs: mapKeys(seen),
			MinSimilarity:  req.MinSimilarity,
			Limit:          int32(req.Limit),
		})
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			mem := memoryFromDB(row.MemoryPair)
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
			Embedding:     embedding,
			UserID:        req.UserID,
			KgID:          req.KGID,
			MinSimilarity: req.MinSimilarity,
			Limit:         int32(req.Limit),
		})
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			subj := subjectFromDB(row)
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
			Embedding:     embeddingAt(embeddings, i),
			SubjectID:     subjectID,
			UserID:        req.UserID,
			MinSimilarity: req.MinSimilarity,
			Limit:         int32(req.Limit),
		})
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			mem := memoryFromDB(row.MemoryPair)
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
	rows, err := s.q.FetchMemories(ctx, req.UserID, req.PairIDs)
	if err != nil {
		return nil, err
	}
	out := make([]harness.Memory, 0, len(rows))
	for _, row := range rows {
		out = append(out, memoryFromDB(row))
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

func memoryFromDB(row db.MemoryPair) harness.Memory {
	var input, output []harness.Content
	var seed []harness.SeedMemoryNode
	var metadata *harness.RetrievalMetadata
	_ = json.Unmarshal(row.Input, &input)
	_ = json.Unmarshal(row.Output, &output)
	_ = json.Unmarshal(row.SeedMemories, &seed)
	if len(row.RetrievalMetadata) > 0 && string(row.RetrievalMetadata) != "null" {
		var md harness.RetrievalMetadata
		if json.Unmarshal(row.RetrievalMetadata, &md) == nil {
			metadata = &md
		}
	}
	offset := 0
	if row.TimezoneOffset != nil {
		offset = int(*row.TimezoneOffset)
	}
	return harness.Memory{
		ID:                row.FirestorePairID,
		SourcePairID:      row.ID.String(),
		UserID:            row.UserID,
		KGID:              row.KgID,
		SessionID:         row.SessionID,
		Title:             row.Title,
		Summary:           row.Description,
		Prompt:            row.Prompt,
		Response:          row.Response,
		Input:             input,
		Output:            output,
		Source:            row.Source,
		SourceContext:     row.SourceContext,
		Timestamp:         row.Timestamp,
		TimezoneOffset:    offset,
		SeedMemories:      seed,
		RetrievalMetadata: metadata,
		Embedding:         row.ConversationEmbedding,
	}
}

func subjectFromDB(row db.Subject) harness.Subject {
	return harness.Subject{
		ID:          row.ID.String(),
		UserID:      row.UserID,
		KGID:        row.KgID,
		Text:        row.Text,
		Description: row.Description,
		Key:         row.IsKeySubject,
		Embedding:   row.Embedding,
		Similarity:  row.Similarity,
		MemoryCount: row.MemoryPairCount,
	}
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
