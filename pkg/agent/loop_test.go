// SPDX-License-Identifier: AGPL-3.0-or-later
package agent

import (
	"context"
	"encoding/json"
	"slices"
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

type recordingHandler struct {
	chatContent      []string
	toolProgressIDs  []string
	toolCompletedIDs []string
	toolResultIDs    []string
	errors           []error
}

func (h *recordingHandler) SendChatContent(text string) {
	h.chatContent = append(h.chatContent, text)
}

func (h *recordingHandler) SendToolCallProgress(toolCallID string, _ map[string]any) {
	h.toolProgressIDs = append(h.toolProgressIDs, toolCallID)
}

func (h *recordingHandler) SendToolCallCompleted(toolCallID, _ string) {
	h.toolCompletedIDs = append(h.toolCompletedIDs, toolCallID)
}

func (h *recordingHandler) SendToolResult(result *harness.ToolCallResponse) {
	h.toolResultIDs = append(h.toolResultIDs, result.ID)
}

func (h *recordingHandler) SendError(err error) {
	h.errors = append(h.errors, err)
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

func TestLoopStreamingEmitsToolAndContentEvents(t *testing.T) {
	handler := &recordingHandler{}
	loop := NewLoop(Options{
		Model: &scriptedModel{chunks: []harness.ChatChunk{
			{ToolCall: &harness.ToolCall{ID: "call_1", Name: "echo", Args: json.RawMessage(`{"ok":true}`)}},
			{Text: "done"},
		}},
		Tools: []harness.Tool{echoTool{}},
	})
	result, err := loop.RunStreaming(context.Background(), RunRequest{
		Messages: []harness.ChatMessage{{Role: "user", Content: []harness.Content{{Content: "hello"}}}},
	}, handler)
	if err != nil {
		t.Fatalf("RunStreaming: %v", err)
	}
	if result.Text != "done" {
		t.Fatalf("Text = %q, want done", result.Text)
	}
	if !slices.Equal(handler.toolProgressIDs, []string{"call_1"}) {
		t.Fatalf("progress IDs = %v", handler.toolProgressIDs)
	}
	if !slices.Equal(handler.toolCompletedIDs, []string{"call_1"}) {
		t.Fatalf("completed IDs = %v", handler.toolCompletedIDs)
	}
	if !slices.Equal(handler.toolResultIDs, []string{"call_1"}) {
		t.Fatalf("result IDs = %v", handler.toolResultIDs)
	}
	if !slices.Equal(handler.chatContent, []string{"done"}) {
		t.Fatalf("chat content = %v", handler.chatContent)
	}
}

func TestLoopBreaksRepeatedToolCallAndSynthesizesFinalAnswer(t *testing.T) {
	loop := NewLoop(Options{
		Model: &scriptedModel{chunks: []harness.ChatChunk{
			{ToolCall: &harness.ToolCall{ID: "call_1", Name: "echo", Args: json.RawMessage(`{"q":"same"}`)}},
			{ToolCall: &harness.ToolCall{ID: "call_2", Name: "echo", Args: json.RawMessage(`{"q":"same"}`)}},
			{ToolCall: &harness.ToolCall{ID: "call_3", Name: "echo", Args: json.RawMessage(`{"q":"same"}`)}},
			{Text: "final after loop break"},
		}},
		Tools: []harness.Tool{echoTool{}},
	})
	result, err := loop.Run(context.Background(), RunRequest{
		Messages: []harness.ChatMessage{{Role: "user", Content: []harness.Content{{Content: "hello"}}}},
		MaxTurns: 5,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Text != "final after loop break" {
		t.Fatalf("Text = %q", result.Text)
	}
	var sawPrompt bool
	for _, msg := range result.Messages {
		if msg.Role != "user" {
			continue
		}
		for _, content := range msg.Content {
			if content.Content == LoopBreakSynthesisPrompt {
				sawPrompt = true
			}
		}
	}
	if !sawPrompt {
		t.Fatal("loop-break synthesis prompt was not appended")
	}
}

func TestExecuteToolCallsPreservesInputOrder(t *testing.T) {
	loop := NewLoop(Options{Tools: []harness.Tool{echoTool{}}})
	results := loop.ExecuteToolCalls(context.Background(), []harness.ToolCall{
		{ID: "call_a", Name: "echo", Args: json.RawMessage(`{"a":true}`)},
		{ID: "call_b", Name: "missing", Args: json.RawMessage(`{}`)},
	})
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2", len(results))
	}
	if results[0].ID != "call_a" || results[0].Error != "" {
		t.Fatalf("unexpected first result: %+v", results[0])
	}
	if results[1].ID != "call_b" || results[1].Error == "" {
		t.Fatalf("unexpected second result: %+v", results[1])
	}
}
