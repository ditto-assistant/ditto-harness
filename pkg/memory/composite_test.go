package memory

import (
	"context"
	"testing"

	"github.com/ditto-assistant/ditto-harness/internal/db"
	"github.com/ditto-assistant/ditto-harness/pkg/retrieval"
	"github.com/ditto-assistant/ditto-harness/pkg/testpg"
)

func TestSearchCompositeMemoriesHydratesScoresAndMetadata(t *testing.T) {
	ctx := context.Background()
	pool := testpg.NewPool(t)
	store := NewStore(Options{
		Queries:   db.New(pool),
		Embedder:  HashEmbedder{},
		Predictor: retrieval.StaticPredictor{Weights: retrieval.DefaultWeights()},
	})

	first, err := store.SaveMemory(ctx, SaveMemoryRequest{
		UserID:   "composite-user",
		Prompt:   "Remember the harness has composite retrieval.",
		Response: "Composite retrieval ranks by similarity, recency, and subject frequency.",
		Summary:  "Composite retrieval exists.",
		Subjects: []SubjectInput{{Text: "Retrieval"}},
	})
	if err != nil {
		t.Fatalf("save first: %v", err)
	}
	_, err = store.SaveMemory(ctx, SaveMemoryRequest{
		UserID:   "composite-user",
		Prompt:   "Remember unrelated settings.",
		Response: "This memory is less relevant to retrieval ranking.",
		Summary:  "Unrelated settings.",
		Subjects: []SubjectInput{{Text: "Settings"}},
	})
	if err != nil {
		t.Fatalf("save second: %v", err)
	}

	memories, metadata, err := store.SearchCompositeMemories(ctx, CompositeSearchRequest{
		UserID:      "composite-user",
		Query:       "composite retrieval ranking",
		Limit:       2,
		RequestPath: "test",
		LogEvent:    true,
	})
	if err != nil {
		t.Fatalf("SearchCompositeMemories: %v", err)
	}
	if len(memories) == 0 {
		t.Fatal("SearchCompositeMemories returned no memories")
	}
	if memories[0].ID != first.ID {
		t.Fatalf("top memory = %q, want %q", memories[0].ID, first.ID)
	}
	if memories[0].CompositeScore == 0 {
		t.Fatalf("top memory missing composite score: %+v", memories[0])
	}
	if metadata == nil || len(metadata.RetrievedPairIDs) == 0 || metadata.Weights["cosine"] == 0 {
		t.Fatalf("metadata incomplete: %+v", metadata)
	}

	var eventCount int
	if err := pool.QueryRow(ctx, `SELECT COUNT(*)::int FROM retrieval_events WHERE user_id = $1`, "composite-user").Scan(&eventCount); err != nil {
		t.Fatalf("count retrieval events: %v", err)
	}
	if eventCount != 1 {
		t.Fatalf("retrieval event count = %d, want 1", eventCount)
	}
}
