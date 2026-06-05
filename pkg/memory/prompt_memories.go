package memory

import (
	"context"
	"errors"

	"github.com/ditto-assistant/ditto-harness/internal/db"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/ditto-assistant/ditto-harness/pkg/retrieval"
)

type PromptMemoryRequest struct {
	UserID            string
	KGID              string
	SessionID         string
	Query             string
	LongTermLimit     int
	ShortTermLimit    int
	CandidatePoolSize int
	ExcludePairIDs    []string
	Variant           retrieval.Variant
	RequestPath       string
	LogRetrieval      bool
	UseComposite      bool
}

type PromptMemoryResponse struct {
	LongTerm          []harness.Memory             `json:"longTerm,omitempty"`
	ShortTerm         []harness.Memory             `json:"shortTerm,omitempty"`
	SeedMemoryNodes   []harness.SeedMemoryNode     `json:"seedMemoryNodes,omitempty"`
	RetrievalMetadata *harness.RetrievalMetadata   `json:"retrievalMetadata,omitempty"`
	LongTermJSON      string                       `json:"longTermJSON,omitempty"`
	IDs               []string                     `json:"ids,omitempty"`
	Summary           []PromptMemorySummary        `json:"summary,omitempty"`
	Diagnostics       map[string]PromptMemoryValue `json:"diagnostics,omitempty"`
}

type PromptMemoryValue struct {
	Count int `json:"count"`
}

func (s *Store) GetPromptMemories(ctx context.Context, req PromptMemoryRequest) (PromptMemoryResponse, error) {
	if s.q == nil {
		return PromptMemoryResponse{}, errors.New("memory: queries are required")
	}
	if req.KGID == "" {
		req.KGID = harness.KGID(req.UserID)
	}
	if req.SessionID == "" {
		req.SessionID = harness.MainSessionID
	}
	if req.LongTermLimit <= 0 {
		req.LongTermLimit = 8
	}

	longTerm, metadata, err := s.getPromptLongTerm(ctx, req)
	if err != nil {
		return PromptMemoryResponse{}, err
	}
	exclude := append([]string(nil), req.ExcludePairIDs...)
	exclude = append(exclude, memoryIDs(longTerm)...)

	shortTerm, err := s.ListRecentMemories(ctx, ListRecentMemoriesRequest{
		UserID:         req.UserID,
		KGID:           req.KGID,
		SessionID:      req.SessionID,
		Limit:          req.ShortTermLimit,
		ExcludePairIDs: exclude,
	})
	if err != nil {
		return PromptMemoryResponse{}, err
	}

	ids := append(memoryIDs(longTerm), memoryIDs(shortTerm)...)
	seedNodes := SeedMemoryNodes(longTerm, shortTerm)
	return PromptMemoryResponse{
		LongTerm:          longTerm,
		ShortTerm:         shortTerm,
		SeedMemoryNodes:   seedNodes,
		RetrievalMetadata: metadata,
		LongTermJSON:      BuildPromptLongTermJSON(longTerm),
		IDs:               ids,
		Summary:           SummarizePromptMemories(append(append([]harness.Memory(nil), longTerm...), shortTerm...), len(longTerm)+len(shortTerm)),
		Diagnostics: map[string]PromptMemoryValue{
			"longTerm":  {Count: len(longTerm)},
			"shortTerm": {Count: len(shortTerm)},
		},
	}, nil
}

type ListRecentMemoriesRequest struct {
	UserID         string
	KGID           string
	SessionID      string
	Limit          int
	ExcludePairIDs []string
}

func (s *Store) ListRecentMemories(ctx context.Context, req ListRecentMemoriesRequest) ([]harness.Memory, error) {
	if s.q == nil {
		return nil, errors.New("memory: queries are required")
	}
	if req.KGID == "" {
		req.KGID = harness.KGID(req.UserID)
	}
	if req.SessionID == "" {
		req.SessionID = harness.MainSessionID
	}
	if req.Limit <= 0 {
		return nil, nil
	}
	rows, err := s.q.ListRecentMemories(ctx, db.ListRecentMemoriesParams{
		UserID:         req.UserID,
		KgID:           req.KGID,
		SessionID:      req.SessionID,
		ExcludePairIDs: req.ExcludePairIDs,
		Limit:          int32(req.Limit),
	})
	if err != nil {
		return nil, err
	}
	out := make([]harness.Memory, 0, len(rows))
	for _, row := range rows {
		out = append(out, memoryFromDB(row))
	}
	return out, nil
}

func (s *Store) getPromptLongTerm(ctx context.Context, req PromptMemoryRequest) ([]harness.Memory, *harness.RetrievalMetadata, error) {
	if req.Query == "" {
		return nil, nil, nil
	}
	if req.UseComposite {
		return s.SearchCompositeMemories(ctx, CompositeSearchRequest{
			UserID:            req.UserID,
			KGID:              req.KGID,
			SessionID:         req.SessionID,
			Query:             req.Query,
			Limit:             req.LongTermLimit,
			CandidatePoolSize: req.CandidatePoolSize,
			ExcludePairIDs:    req.ExcludePairIDs,
			Variant:           req.Variant,
			RequestPath:       req.RequestPath,
			LogEvent:          req.LogRetrieval,
		})
	}
	memories, err := s.SearchMemories(ctx, SearchMemoriesRequest{
		UserID:         req.UserID,
		KGID:           req.KGID,
		SessionID:      req.SessionID,
		Queries:        []string{req.Query},
		Limit:          req.LongTermLimit,
		ExcludePairIDs: req.ExcludePairIDs,
	})
	return memories, nil, err
}

func SeedMemoryNodes(longTerm, shortTerm []harness.Memory) []harness.SeedMemoryNode {
	nodes := make([]harness.SeedMemoryNode, 0, len(longTerm)+len(shortTerm))
	for _, mem := range longTerm {
		nodes = append(nodes, seedNode(mem))
	}
	for _, mem := range shortTerm {
		nodes = append(nodes, seedNode(mem))
	}
	return nodes
}

func seedNode(mem harness.Memory) harness.SeedMemoryNode {
	return harness.SeedMemoryNode{
		PairID:   mem.ID,
		Children: append([]harness.SeedMemoryNode(nil), mem.SeedMemories...),
	}
}

func memoryIDs(memories []harness.Memory) []string {
	ids := make([]string, 0, len(memories))
	for _, mem := range memories {
		if mem.ID != "" {
			ids = append(ids, mem.ID)
		}
	}
	return ids
}
