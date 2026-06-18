// SPDX-License-Identifier: AGPL-3.0-or-later
package memory

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/ditto-assistant/ditto-harness/pkg/harness"
)

func TestBuildPromptLongTermJSONCompressesAfterFirstTwo(t *testing.T) {
	memories := []harness.Memory{
		{ID: "pair-1", Summary: "First summary", Timestamp: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)},
		{ID: "pair-2", Prompt: "Second prompt", Response: "Second response", Timestamp: time.Date(2026, 1, 2, 12, 0, 0, 0, time.UTC)},
		{ID: "pair-3", Title: "Third title", Prompt: "Third prompt", Response: "Third response", Timestamp: time.Date(2026, 1, 3, 12, 0, 0, 0, time.UTC)},
	}
	raw := BuildPromptLongTermJSON(memories)
	var payload struct {
		Memories []map[string]any `json:"memories"`
	}
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		t.Fatalf("unmarshal prompt json: %v\n%s", err, raw)
	}
	if len(payload.Memories) != 3 {
		t.Fatalf("memory count = %d, want 3", len(payload.Memories))
	}
	if _, ok := payload.Memories[0]["summary"]; !ok {
		t.Fatalf("first memory should be detailed: %+v", payload.Memories[0])
	}
	if _, ok := payload.Memories[1]["user"]; !ok {
		t.Fatalf("second memory should include user text: %+v", payload.Memories[1])
	}
	if _, ok := payload.Memories[2]["title"]; !ok {
		t.Fatalf("third memory should be compressed title-only: %+v", payload.Memories[2])
	}
	if _, ok := payload.Memories[2]["user"]; ok {
		t.Fatalf("third memory should not include full user text: %+v", payload.Memories[2])
	}
}

func TestSummarizePromptMemories(t *testing.T) {
	memories := []harness.Memory{{
		ID:             "pair-1",
		SessionID:      "",
		Source:         "agent",
		Title:          "Title",
		Timestamp:      time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC),
		Similarity:     0.8,
		CompositeScore: 0.7,
	}}
	got := SummarizePromptMemories(memories, 10)
	if len(got) != 1 {
		t.Fatalf("summary len = %d, want 1", len(got))
	}
	if got[0].SessionID != harness.MainSessionID {
		t.Fatalf("session = %q, want main", got[0].SessionID)
	}
	if got[0].Title != "Title" || got[0].CosineSim != 0.8 || got[0].CompositeScore != 0.7 {
		t.Fatalf("unexpected summary: %+v", got[0])
	}
}
