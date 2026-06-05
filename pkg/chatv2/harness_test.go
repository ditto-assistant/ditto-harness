package chatv2

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ditto-assistant/ditto-harness/internal/db"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/ditto-assistant/ditto-harness/pkg/memory"
	"github.com/ditto-assistant/ditto-harness/pkg/testpg"
)

type scriptedModel struct {
	chunks       []harness.ChatChunk
	seenTools    []string
	seenMessages []harness.ChatMessage
}

func (m *scriptedModel) Next(_ context.Context, messages []harness.ChatMessage, tools []harness.ToolDefinition) (harness.ChatChunk, error) {
	m.seenMessages = append([]harness.ChatMessage(nil), messages...)
	m.seenTools = m.seenTools[:0]
	for _, tool := range tools {
		m.seenTools = append(m.seenTools, tool.Name)
	}
	chunk := m.chunks[0]
	m.chunks = m.chunks[1:]
	return chunk, nil
}

type markerTool struct{}

func (markerTool) Definition() harness.ToolDefinition {
	return harness.ToolDefinition{Name: "marker"}
}

func (markerTool) Call(_ context.Context, raw json.RawMessage) (harness.ToolCallResponse, error) {
	return harness.ToolCallResponse{Name: "marker", Output: raw}, nil
}

func TestHarnessPrepareRunAndSave(t *testing.T) {
	ctx := context.Background()
	pool := testpg.NewPool(t)
	store := memory.NewStore(memory.Options{Queries: db.New(pool), Embedder: hashEmbedder{}})

	_, err := store.SaveMemory(ctx, memory.SaveMemoryRequest{
		ID:        "seed-memory",
		UserID:    "chat-user",
		SessionID: "thread-a",
		Prompt:    "Remember that chatv2 should import the harness.",
		Response:  "The harness prepares memory context before running the agent loop.",
		Summary:   "chatv2 imports the harness for memory context.",
		Timestamp: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatalf("SaveMemory seed: %v", err)
	}

	model := &scriptedModel{chunks: []harness.ChatChunk{{
		Text: "final answer",
		Cost: &harness.CostedUsage{
			Usage: harness.Usage{Model: "test-model", TotalTokens: 12},
			Cost:  harness.Cost{Currency: "USD", Amount: 0.01},
		},
	}}}
	h := New(Options{
		Model:              model,
		Memory:             store,
		Tools:              []harness.Tool{markerTool{}},
		IncludeMemoryTools: true,
	})

	result, err := h.Run(ctx, RunRequest{
		PrepareRequest: PrepareRequest{
			UserID:         "chat-user",
			SessionID:      "thread-a",
			UserInput:      "How should chatv2 use memory context?",
			SystemPrompt:   "You are a harness test.",
			LongTermLimit:  1,
			ShortTermLimit: 0,
		},
		SaveMemory: true,
		Source:     "chatv2_test",
	}, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Text != "final answer" || len(result.Costs) != 1 {
		t.Fatalf("unexpected run result: %+v", result.RunResult)
	}
	if result.SavedMemory == nil || result.SavedMemory.Source != "chatv2_test" {
		t.Fatalf("saved memory missing or wrong source: %+v", result.SavedMemory)
	}
	if len(result.SavedMemory.SeedMemories) != 1 || result.SavedMemory.SeedMemories[0].PairID != "seed-memory" {
		t.Fatalf("seed metadata not persisted: %+v", result.SavedMemory.SeedMemories)
	}
	if !slices.Contains(model.seenTools, "marker") || !slices.Contains(model.seenTools, "save_memory") {
		t.Fatalf("model tools = %v, want injected and memory tools", model.seenTools)
	}
	if !containsMemoryContext(model.seenMessages) {
		t.Fatalf("model did not receive memory context: %+v", model.seenMessages)
	}
}

func containsMemoryContext(messages []harness.ChatMessage) bool {
	for _, msg := range messages {
		if msg.Role != "system" {
			continue
		}
		for _, part := range msg.Content {
			if strings.Contains(part.Content, "Relevant memory context") && strings.Contains(part.Content, "seed-memory") {
				return true
			}
		}
	}
	return false
}

type hashEmbedder struct{}

func (hashEmbedder) Embed(_ context.Context, req harness.EmbedRequest) (harness.EmbedResponse, error) {
	out := make([][]float32, len(req.Texts))
	for i, text := range req.Texts {
		out[i] = hashEmbedding(text)
	}
	return harness.EmbedResponse{Embeddings: out}, nil
}

func hashEmbedding(text string) []float32 {
	vec := make([]float32, 768)
	for _, token := range strings.Fields(strings.ToLower(text)) {
		sum := sha256.Sum256([]byte(token))
		idx := int(binary.BigEndian.Uint16(sum[:2]) % uint16(len(vec)))
		vec[idx] += 1
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v * v)
	}
	if norm == 0 {
		vec[0] = 1
		return vec
	}
	scale := float32(1 / math.Sqrt(norm))
	for i := range vec {
		vec[i] *= scale
	}
	return vec
}
