package mcpserver

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/ditto-assistant/ditto-harness/pkg/db"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/ditto-assistant/ditto-harness/pkg/memory"
	"github.com/ditto-assistant/ditto-harness/pkg/testpg"
	"github.com/mark3labs/mcp-go/mcp"
)

func TestServerHandlersSupportSubjectsAndSlimMemoryPayloads(t *testing.T) {
	ctx := context.Background()
	pool := testpg.NewPool(t)
	store := memory.NewStore(memory.Options{Queries: db.New(pool), Embedder: hashEmbedder{}})
	srv := New(Options{Store: store, UserID: "mcp-user"})

	saveResult, err := srv.handleSaveMemory(ctx, callRequest("save_memory", map[string]any{
		"prompt":   "Remember that MCP save can attach subjects.",
		"response": "Subjects should be searchable later.",
		"summary":  "MCP subject save works.",
		"subjects": []map[string]any{{
			"text":        "MCP harness parity",
			"description": "Subject saved through MCP",
			"key":         true,
		}},
	}))
	if err != nil {
		t.Fatalf("handleSaveMemory: %v", err)
	}
	if toolText(saveResult) == "" {
		t.Fatal("save result should contain JSON text")
	}

	subjectResult, err := srv.handleSearchSubjects(ctx, callRequest("search_subjects", map[string]any{
		"queries": []any{"MCP harness parity"},
		"topK":    5,
	}))
	if err != nil {
		t.Fatalf("handleSearchSubjects: %v", err)
	}
	var subjectsPayload struct {
		Subjects []struct {
			ID   string `json:"id"`
			Text string `json:"text"`
		} `json:"subjects"`
	}
	if err := json.Unmarshal([]byte(toolText(subjectResult)), &subjectsPayload); err != nil {
		t.Fatalf("unmarshal subject payload: %v", err)
	}
	if len(subjectsPayload.Subjects) != 1 || subjectsPayload.Subjects[0].Text != "MCP harness parity" {
		t.Fatalf("unexpected subjects payload: %+v", subjectsPayload)
	}

	memoriesResult, err := srv.handleSearchMemoriesInSubjects(ctx, callRequest("search_memories_in_subjects", map[string]any{
		"subject_id": subjectsPayload.Subjects[0].ID,
		"queries":    []any{"subjects searchable later"},
		"topK":       5,
	}))
	if err != nil {
		t.Fatalf("handleSearchMemoriesInSubjects: %v", err)
	}
	var memoriesPayload struct {
		Memories []struct {
			ID      string `json:"id"`
			Preview string `json:"preview"`
			User    string `json:"user"`
		} `json:"memories"`
	}
	if err := json.Unmarshal([]byte(toolText(memoriesResult)), &memoriesPayload); err != nil {
		t.Fatalf("unmarshal memories payload: %v", err)
	}
	if len(memoriesPayload.Memories) != 1 || memoriesPayload.Memories[0].Preview == "" || memoriesPayload.Memories[0].User != "" {
		t.Fatalf("MCP subject memory search should return slim previews: %+v", memoriesPayload)
	}
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

func callRequest(name string, args map[string]any) mcp.CallToolRequest {
	return mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      name,
			Arguments: args,
		},
	}
}

func toolText(result *mcp.CallToolResult) string {
	if result == nil {
		return ""
	}
	raw, _ := json.Marshal(result.Content)
	var texts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	_ = json.Unmarshal(raw, &texts)
	var out strings.Builder
	for _, item := range texts {
		if item.Type == "text" {
			out.WriteString(item.Text)
		}
	}
	return out.String()
}
