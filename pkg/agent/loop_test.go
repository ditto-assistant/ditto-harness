package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/ditto-assistant/ditto-harness/pkg/harness"
)

type scriptedModel struct {
	chunks []harness.ChatChunk
}

func (m *scriptedModel) Next(context.Context, []harness.ChatMessage, []harness.ToolDefinition) (harness.ChatChunk, error) {
	chunk := m.chunks[0]
	m.chunks = m.chunks[1:]
	return chunk, nil
}

type echoTool struct{}

func (echoTool) Definition() harness.ToolDefinition {
	return harness.ToolDefinition{Name: "echo"}
}

func (echoTool) Call(_ context.Context, raw json.RawMessage) (harness.ToolCallResponse, error) {
	return harness.ToolCallResponse{ID: "call_1", Name: "echo", Output: raw}, nil
}

func TestLoopExecutesToolThenReturnsFinalText(t *testing.T) {
	loop := NewLoop(Options{
		Model: &scriptedModel{chunks: []harness.ChatChunk{
			{ToolCall: &harness.ToolCall{ID: "call_1", Name: "echo", Args: json.RawMessage(`{"ok":true}`)}},
			{Text: "done"},
		}},
		Tools: []harness.Tool{echoTool{}},
	})

	result, err := loop.Run(context.Background(), RunRequest{
		Messages: []harness.ChatMessage{{Role: "user", Content: []harness.Content{{Content: "hello"}}}},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Text != "done" {
		t.Fatalf("Text = %q, want done", result.Text)
	}
	if len(result.Messages) != 4 {
		t.Fatalf("messages len = %d, want 4", len(result.Messages))
	}
}
