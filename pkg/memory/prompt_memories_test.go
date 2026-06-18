// SPDX-License-Identifier: AGPL-3.0-or-later
package memory

import (
	"context"
	"testing"
	"time"

	"github.com/ditto-assistant/ditto-harness/pkg/db"
	"github.com/ditto-assistant/ditto-harness/pkg/testpg"
)

func TestGetPromptMemoriesReturnsCompositeLongTermAndRecentShortTerm(t *testing.T) {
	ctx := context.Background()
	pool := testpg.NewPool(t)
	store := NewStore(Options{Queries: db.New(pool), Embedder: HashEmbedder{}})

	older := time.Date(2026, 1, 1, 10, 0, 0, 0, time.UTC)
	longTerm, err := store.SaveMemory(ctx, SaveMemoryRequest{
		ID:        "long-term",
		UserID:    "prompt-user",
		SessionID: "thread-1",
		Prompt:    "Remember the harness retrieves composite memory context.",
		Response:  "Composite retrieval should seed the prompt.",
		Summary:   "Harness composite retrieval context.",
		Timestamp: older,
	})
	if err != nil {
		t.Fatalf("SaveMemory long term: %v", err)
	}
	_, err = store.SaveMemory(ctx, SaveMemoryRequest{
		ID:        "recent-thread",
		UserID:    "prompt-user",
		SessionID: "thread-1",
		Prompt:    "Recent thread note about packaging.",
		Response:  "Keep the chat harness importable.",
		Summary:   "Recent importable harness note.",
		Timestamp: older.Add(time.Hour),
	})
	if err != nil {
		t.Fatalf("SaveMemory recent: %v", err)
	}

	got, err := store.GetPromptMemories(ctx, PromptMemoryRequest{
		UserID:         "prompt-user",
		SessionID:      "thread-1",
		Query:          "composite memory context",
		LongTermLimit:  1,
		ShortTermLimit: 2,
	})
	if err != nil {
		t.Fatalf("GetPromptMemories: %v", err)
	}
	if len(got.LongTerm) != 1 || got.LongTerm[0].ID != longTerm.ID {
		t.Fatalf("LongTerm = %+v, want %s", got.LongTerm, longTerm.ID)
	}
	if len(got.ShortTerm) != 1 || got.ShortTerm[0].ID != "recent-thread" {
		t.Fatalf("ShortTerm = %+v, want recent-thread only", got.ShortTerm)
	}
	if len(got.SeedMemoryNodes) != 2 || got.LongTermJSON == "" {
		t.Fatalf("incomplete prompt metadata: %+v", got)
	}
}
