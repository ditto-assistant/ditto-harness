package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/ditto-assistant/ditto-harness/pkg/cost"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/ditto-assistant/ditto-harness/pkg/memory"
)

type Loop struct {
	model  harness.Model
	memory *memory.Store
	tools  []harness.Tool
}

type Options struct {
	Model  harness.Model
	Memory *memory.Store
	Tools  []harness.Tool
}

func NewLoop(opts Options) *Loop {
	return &Loop{model: opts.Model, memory: opts.Memory, tools: opts.Tools}
}

type RunRequest struct {
	UserID     string
	KGID       string
	SessionID  string
	Messages   []harness.ChatMessage
	MaxTurns   int
	SaveMemory bool
}

type RunResult struct {
	Messages []harness.ChatMessage `json:"messages"`
	Text     string                `json:"text,omitempty"`
	Costs    []harness.CostedUsage `json:"costs,omitempty"`
	Metadata map[string]any        `json:"metadata,omitempty"`
}

func (l *Loop) Run(ctx context.Context, req RunRequest) (RunResult, error) {
	if l.model == nil {
		return RunResult{}, errors.New("agent: model is required")
	}
	if req.MaxTurns <= 0 {
		req.MaxTurns = 8
	}
	messages := append([]harness.ChatMessage(nil), req.Messages...)
	defs := make([]harness.ToolDefinition, 0, len(l.tools))
	toolsByName := make(map[string]harness.Tool, len(l.tools))
	for _, tool := range l.tools {
		def := tool.Definition()
		defs = append(defs, def)
		toolsByName[def.Name] = tool
	}

	var costs cost.Collector
	var final string
	for turn := 0; turn < req.MaxTurns; turn++ {
		chunk, err := l.model.Next(ctx, messages, defs)
		if err != nil {
			return RunResult{}, err
		}
		costs.Add(chunk.Cost)
		if chunk.ToolCall == nil {
			final = chunk.Text
			messages = append(messages, harness.ChatMessage{
				Role:    "assistant",
				Content: []harness.Content{{Type: harness.ContentTypeText, Content: chunk.Text}},
			})
			break
		}

		tc := *chunk.ToolCall
		messages = append(messages, harness.ChatMessage{
			Role:      "assistant",
			ToolCalls: []harness.ToolCall{tc},
		})
		tool, ok := toolsByName[tc.Name]
		if !ok {
			return RunResult{}, fmt.Errorf("agent: unknown tool %q", tc.Name)
		}
		resp, err := tool.Call(ctx, tc.Args)
		if err != nil {
			resp = harness.ToolCallResponse{ID: tc.ID, Name: tc.Name, Error: err.Error()}
		}
		raw, _ := json.Marshal(resp)
		messages = append(messages, harness.ChatMessage{
			Role:       "tool",
			ToolCallID: tc.ID,
			Content: []harness.Content{{
				Type: harness.ContentTypeToolResult,
				ToolCallResponse: &harness.ToolCallResponse{
					ID:     tc.ID,
					Name:   tc.Name,
					Output: raw,
					Error:  resp.Error,
				},
			}},
		})
	}

	if req.SaveMemory && l.memory != nil && final != "" {
		_, err := l.memory.SaveMemory(ctx, memory.SaveMemoryRequest{
			UserID:    req.UserID,
			KGID:      req.KGID,
			SessionID: req.SessionID,
			Prompt:    lastUserText(req.Messages),
			Response:  final,
			Output:    []harness.Content{{Type: harness.ContentTypeText, Content: final}},
			Source:    "agent_loop",
		})
		if err != nil {
			return RunResult{}, fmt.Errorf("save agent memory: %w", err)
		}
	}

	return RunResult{Messages: messages, Text: final, Costs: costs.Items()}, nil
}

func (l *Loop) Tools() []harness.ToolDefinition {
	out := make([]harness.ToolDefinition, 0, len(l.tools))
	for _, tool := range l.tools {
		out = append(out, tool.Definition())
	}
	return out
}

func lastUserText(messages []harness.ChatMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "user" {
			continue
		}
		var out string
		for _, part := range messages[i].Content {
			out += part.Content
		}
		return out
	}
	return ""
}
