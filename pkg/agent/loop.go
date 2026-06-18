// SPDX-License-Identifier: AGPL-3.0-or-later
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/ditto-assistant/ditto-harness/pkg/agent/loopdetect"
	"github.com/ditto-assistant/ditto-harness/pkg/cost"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/ditto-assistant/ditto-harness/pkg/memory"
)

const LoopBreakSynthesisPrompt = "You repeated the same tool call several times without producing a final answer. Stop calling tools and answer the user's request now using the conversation context and the tool results you already have. If the available results are incomplete, say that briefly and answer with what you have."

var loopBreakToolResult = json.RawMessage(`{"status":"loop_detected","message":"This repeated tool call was stopped. Do not call tools again. Write the final answer using the tool results already available in this conversation."}`)

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

type EventHandler interface {
	SendChatContent(text string)
	SendToolCallProgress(toolCallID string, data map[string]any)
	SendToolCallCompleted(toolCallID, toolName string)
	SendToolResult(result *harness.ToolCallResponse)
	SendError(err error)
}

type noopHandler struct{}

func (noopHandler) SendChatContent(string)                      {}
func (noopHandler) SendToolCallProgress(string, map[string]any) {}
func (noopHandler) SendToolCallCompleted(string, string)        {}
func (noopHandler) SendToolResult(*harness.ToolCallResponse)    {}
func (noopHandler) SendError(error)                             {}

func (l *Loop) Run(ctx context.Context, req RunRequest) (RunResult, error) {
	return l.RunStreaming(ctx, req, nil)
}

func (l *Loop) RunStreaming(ctx context.Context, req RunRequest, handler EventHandler) (RunResult, error) {
	if l.model == nil {
		return RunResult{}, errors.New("agent: model is required")
	}
	if handler == nil {
		handler = noopHandler{}
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
	var detector loopdetect.Detector
	for turn := 0; turn < req.MaxTurns; turn++ {
		chunk, err := l.model.Next(ctx, messages, defs)
		if err != nil {
			handler.SendError(err)
			return RunResult{}, err
		}
		costs.Add(chunk.Cost)
		if chunk.ToolCall == nil {
			final = chunk.Text
			if chunk.Text != "" {
				handler.SendChatContent(chunk.Text)
			}
			messages = append(messages, harness.ChatMessage{
				Role:    "assistant",
				Content: []harness.Content{{Type: harness.ContentTypeText, Content: chunk.Text}},
			})
			break
		}

		tc := *chunk.ToolCall
		handler.SendToolCallProgress(tc.ID, map[string]any{"name": tc.Name, "arguments": string(tc.Args)})
		handler.SendToolCallCompleted(tc.ID, tc.Name)
		messages = append(messages, harness.ChatMessage{
			Role:      "assistant",
			ToolCalls: []harness.ToolCall{tc},
		})

		if _, ok := detector.RecordTurn(turn, []loopdetect.ToolCall{{Name: tc.Name, Args: string(tc.Args)}}); ok {
			resp := harness.ToolCallResponse{ID: tc.ID, Name: tc.Name, Output: loopBreakToolResult}
			handler.SendToolResult(&resp)
			messages = append(messages, toolMessage(resp))
			messages = append(messages, harness.ChatMessage{
				Role:    "user",
				Content: []harness.Content{{Type: harness.ContentTypeText, Content: LoopBreakSynthesisPrompt}},
			})
			defs = nil
			toolsByName = nil
			continue
		}

		resp := l.executeTool(ctx, toolsByName, tc)
		handler.SendToolResult(&resp)
		messages = append(messages, toolMessage(resp))
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

func (l *Loop) ExecuteToolCalls(ctx context.Context, calls []harness.ToolCall) []harness.ToolCallResponse {
	toolsByName := make(map[string]harness.Tool, len(l.tools))
	for _, tool := range l.tools {
		toolsByName[tool.Definition().Name] = tool
	}
	return executeToolCalls(ctx, toolsByName, calls)
}

func (l *Loop) executeTool(ctx context.Context, toolsByName map[string]harness.Tool, tc harness.ToolCall) harness.ToolCallResponse {
	results := executeToolCalls(ctx, toolsByName, []harness.ToolCall{tc})
	if len(results) == 0 {
		return harness.ToolCallResponse{ID: tc.ID, Name: tc.Name, Error: "tool produced no result"}
	}
	return results[0]
}

func executeToolCalls(ctx context.Context, toolsByName map[string]harness.Tool, calls []harness.ToolCall) []harness.ToolCallResponse {
	if len(calls) == 0 {
		return nil
	}
	results := make([]harness.ToolCallResponse, len(calls))
	var wg sync.WaitGroup
	wg.Add(len(calls))
	for i, call := range calls {
		go func(i int, call harness.ToolCall) {
			defer wg.Done()
			tool, ok := toolsByName[call.Name]
			if !ok {
				results[i] = harness.ToolCallResponse{ID: call.ID, Name: call.Name, Error: fmt.Sprintf("unknown tool %q", call.Name)}
				return
			}
			resp, err := tool.Call(ctx, call.Args)
			resp.ID = call.ID
			if resp.Name == "" {
				resp.Name = call.Name
			}
			if err != nil && resp.Error == "" {
				resp.Error = err.Error()
			}
			results[i] = resp
		}(i, call)
	}
	wg.Wait()
	return results
}

func toolMessage(resp harness.ToolCallResponse) harness.ChatMessage {
	raw, _ := json.Marshal(resp)
	result := resp
	result.Output = raw
	return harness.ChatMessage{
		Role:       "tool",
		ToolCallID: resp.ID,
		Content: []harness.Content{{
			Type:             harness.ContentTypeToolResult,
			ToolCallResponse: &result,
		}},
	}
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
