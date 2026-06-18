// SPDX-License-Identifier: AGPL-3.0-or-later
package chatv2_test

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ditto-assistant/ditto-harness/pkg/chatv2"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
)

type exampleModel struct{}

func (exampleModel) Next(context.Context, []harness.ChatMessage, []harness.ToolDefinition) (harness.ChatChunk, error) {
	return harness.ChatChunk{
		Text: "Use the retrieved memory context and injected tools.",
		Cost: &harness.CostedUsage{
			Usage: harness.Usage{Provider: "host", Model: "example-model", TotalTokens: 42},
			Cost:  harness.Cost{Currency: "USD", Amount: 0.001},
		},
	}, nil
}

type exampleTool struct{}

func (exampleTool) Definition() harness.ToolDefinition {
	return harness.ToolDefinition{
		Name:        "host_tool",
		Description: "An application-specific tool supplied by the importing service.",
		InputSchema: json.RawMessage(`{"type":"object"}`),
	}
}

func (exampleTool) Call(context.Context, json.RawMessage) (harness.ToolCallResponse, error) {
	return harness.ToolCallResponse{Name: "host_tool", Output: json.RawMessage(`{"ok":true}`)}, nil
}

func ExampleHarness_Run() {
	h := chatv2.New(chatv2.Options{
		Model: exampleModel{},
		Tools: []harness.Tool{exampleTool{}},
	})

	result, err := h.Run(context.Background(), chatv2.RunRequest{
		PrepareRequest: chatv2.PrepareRequest{
			UserID:       "user_123",
			UserInput:    "How should backend import ditto-harness?",
			SystemPrompt: "You are a concise assistant.",
		},
	}, nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Text)
	fmt.Println(result.Costs[0].Usage.TotalTokens)
	// Output:
	// Use the retrieved memory context and injected tools.
	// 42
}
