// SPDX-License-Identifier: AGPL-3.0-or-later
package memory

import (
	"context"
	"testing"

	"github.com/ditto-assistant/ditto-harness/pkg/db"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/ditto-assistant/ditto-harness/pkg/testpg"
)

func TestStoreSaveSearchFetchAndSubjects(t *testing.T) {
	ctx := context.Background()
	pool := testpg.NewPool(t)
	store := NewStore(Options{Queries: db.New(pool), Embedder: HashEmbedder{}})

	saved, err := store.SaveMemory(ctx, SaveMemoryRequest{
		UserID:   "user-test",
		Prompt:   "Remember that Peyton prefers direct engineering updates.",
		Response: "Use concise status notes and concrete file references.",
		Summary:  "Peyton prefers direct engineering updates.",
		Input:    []harness.Content{{Type: harness.ContentTypeText, Content: "direct updates"}},
		Output:   []harness.Content{{Type: harness.ContentTypeText, Content: "concrete file references"}},
		Subjects: []SubjectInput{{
			Text:        "Engineering communication",
			Description: "Preferences about direct updates",
			Key:         true,
		}},
	})
	if err != nil {
		t.Fatalf("SaveMemory: %v", err)
	}
	if saved.ID == "" {
		t.Fatal("SaveMemory returned empty id")
	}

	memories, err := store.SearchMemories(ctx, SearchMemoriesRequest{
		UserID:  "user-test",
		Queries: []string{"direct engineering updates"},
		Limit:   5,
	})
	if err != nil {
		t.Fatalf("SearchMemories: %v", err)
	}
	if len(memories) != 1 || memories[0].ID != saved.ID {
		t.Fatalf("SearchMemories = %+v, want saved memory %s", memories, saved.ID)
	}

	subjects, err := store.SearchSubjects(ctx, SearchSubjectsRequest{
		UserID:  "user-test",
		Queries: []string{"engineering communication"},
		Limit:   5,
	})
	if err != nil {
		t.Fatalf("SearchSubjects: %v", err)
	}
	if len(subjects) != 1 {
		t.Fatalf("SearchSubjects len = %d, want 1", len(subjects))
	}

	inSubject, err := store.SearchMemoriesInSubjects(ctx, SearchMemoriesInSubjectsRequest{
		UserID: "user-test",
		Queries: []SubjectMemoryQuery{{
			SubjectID: subjects[0].ID,
			Query:     "concrete file references",
		}},
	})
	if err != nil {
		t.Fatalf("SearchMemoriesInSubjects: %v", err)
	}
	if len(inSubject) != 1 || inSubject[0].ID != saved.ID {
		t.Fatalf("SearchMemoriesInSubjects = %+v, want saved memory %s", inSubject, saved.ID)
	}

	fetched, err := store.FetchMemories(ctx, FetchMemoriesRequest{
		UserID:  "user-test",
		PairIDs: []string{saved.ID},
	})
	if err != nil {
		t.Fatalf("FetchMemories: %v", err)
	}
	if len(fetched) != 1 || fetched[0].Prompt == "" || len(fetched[0].Input) != 1 {
		t.Fatalf("FetchMemories returned incomplete memory: %+v", fetched)
	}
}
