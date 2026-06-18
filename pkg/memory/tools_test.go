// SPDX-License-Identifier: AGPL-3.0-or-later
package memory

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/ditto-assistant/ditto-harness/pkg/db"
	"github.com/ditto-assistant/ditto-harness/pkg/testpg"
)

func TestMemoryToolsExposeExpectedDefinitions(t *testing.T) {
	tools := Tools(ToolOptions{})
	names := make([]string, len(tools))
	for i, tool := range tools {
		def := tool.Definition()
		names[i] = def.Name
		if len(def.InputSchema) == 0 || !json.Valid(def.InputSchema) {
			t.Fatalf("%s has invalid schema %s", def.Name, def.InputSchema)
		}
	}
	want := []string{"save_memory", "search_memories", "search_subjects", "search_memories_in_subjects", "fetch_memories"}
	if got := names; len(got) != len(want) {
		t.Fatalf("tool count = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Fatalf("tool[%d] = %q, want %q", i, names[i], want[i])
		}
	}
}

func TestMemoryToolsCallThroughStore(t *testing.T) {
	ctx := context.Background()
	pool := testpg.NewPool(t)
	store := NewStore(Options{Queries: db.New(pool), Embedder: HashEmbedder{}})

	tools := Tools(ToolOptions{UserID: "tool-user", Store: store})
	save := tools[0]
	resp, err := save.Call(ctx, json.RawMessage(`{"prompt":"Remember the importable tool adapter","response":"It should call through the store","summary":"Tool adapter works","subjects":[{"text":"Harness tools"}]}`))
	if err != nil {
		t.Fatalf("save tool: %v", err)
	}
	if len(resp.Output) == 0 {
		t.Fatal("save tool returned empty output")
	}

	search := tools[1]
	resp, err = search.Call(ctx, json.RawMessage(`{"queries":["importable tool adapter"],"topK":5}`))
	if err != nil {
		t.Fatalf("search tool: %v", err)
	}
	var payload struct {
		Memories []struct {
			ID      string `json:"id"`
			Preview string `json:"preview"`
			User    string `json:"user"`
			Ditto   string `json:"ditto"`
		} `json:"memories"`
	}
	if err := json.Unmarshal(resp.Output, &payload); err != nil {
		t.Fatalf("unmarshal search output: %v", err)
	}
	if len(payload.Memories) != 1 {
		t.Fatalf("search returned %d memories, want 1", len(payload.Memories))
	}
	if payload.Memories[0].Preview == "" || payload.Memories[0].User != "" || payload.Memories[0].Ditto != "" {
		t.Fatalf("search should return slim previews only: %+v", payload.Memories[0])
	}

	fetch := tools[4]
	resp, err = fetch.Call(ctx, json.RawMessage(`{"pairIds":["`+payload.Memories[0].ID+`"]}`))
	if err != nil {
		t.Fatalf("fetch tool: %v", err)
	}
	var fetched struct {
		Memories []struct {
			ID      string `json:"id"`
			Preview string `json:"preview"`
			User    string `json:"user"`
			Ditto   string `json:"ditto"`
		} `json:"memories"`
	}
	if err := json.Unmarshal(resp.Output, &fetched); err != nil {
		t.Fatalf("unmarshal fetch output: %v", err)
	}
	if len(fetched.Memories) != 1 || fetched.Memories[0].User == "" || fetched.Memories[0].Ditto == "" || fetched.Memories[0].Preview != "" {
		t.Fatalf("fetch should return truncated full slim memory: %+v", fetched.Memories)
	}
}
