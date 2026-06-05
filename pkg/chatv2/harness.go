package chatv2

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/ditto-assistant/ditto-harness/pkg/agent"
	"github.com/ditto-assistant/ditto-harness/pkg/harness"
	"github.com/ditto-assistant/ditto-harness/pkg/memory"
	"github.com/ditto-assistant/ditto-harness/pkg/retrieval"
)

type Harness struct {
	model              harness.Model
	memory             *memory.Store
	tools              []harness.Tool
	includeMemoryTools bool
}

type Options struct {
	Model              harness.Model
	Memory             *memory.Store
	Tools              []harness.Tool
	IncludeMemoryTools bool
}

func New(opts Options) *Harness {
	return &Harness{
		model:              opts.Model,
		memory:             opts.Memory,
		tools:              append([]harness.Tool(nil), opts.Tools...),
		includeMemoryTools: opts.IncludeMemoryTools,
	}
}

type PrepareRequest struct {
	UserID            string
	KGID              string
	SessionID         string
	UserInput         string
	SystemPrompt      string
	Messages          []harness.ChatMessage
	LongTermLimit     int
	ShortTermLimit    int
	CandidatePoolSize int
	ExcludePairIDs    []string
	Variant           retrieval.Variant
	RequestPath       string
	LogRetrieval      bool
	UseComposite      bool
}

type PrepareResult struct {
	Messages            []harness.ChatMessage       `json:"messages"`
	Tools               []harness.ToolDefinition    `json:"tools,omitempty"`
	Memories            memory.PromptMemoryResponse `json:"memories,omitempty"`
	AlreadyFoundPairIDs []string                    `json:"alreadyFoundPairIds,omitempty"`
}

type RunRequest struct {
	PrepareRequest
	MaxTurns       int
	SaveMemory     bool
	Source         string
	SourceContext  string
	Timestamp      time.Time
	TimezoneOffset int
	Subjects       []memory.SubjectInput
}

type RunResult struct {
	agent.RunResult
	Preparation memory.PromptMemoryResponse `json:"preparation,omitempty"`
	SavedMemory *harness.Memory             `json:"savedMemory,omitempty"`
}

func (h *Harness) Prepare(ctx context.Context, req PrepareRequest) (PrepareResult, error) {
	if req.UserID == "" {
		return PrepareResult{}, errors.New("chatv2: user id is required")
	}
	if req.KGID == "" {
		req.KGID = harness.KGID(req.UserID)
	}
	if req.SessionID == "" {
		req.SessionID = harness.MainSessionID
	}
	messages := normalizeMessages(req.Messages, req.UserInput, req.SystemPrompt)

	var memories memory.PromptMemoryResponse
	if h.memory != nil && req.UserInput != "" {
		var err error
		memories, err = h.memory.GetPromptMemories(ctx, memory.PromptMemoryRequest{
			UserID:            req.UserID,
			KGID:              req.KGID,
			SessionID:         req.SessionID,
			Query:             req.UserInput,
			LongTermLimit:     req.LongTermLimit,
			ShortTermLimit:    req.ShortTermLimit,
			CandidatePoolSize: req.CandidatePoolSize,
			ExcludePairIDs:    req.ExcludePairIDs,
			Variant:           req.Variant,
			RequestPath:       req.RequestPath,
			LogRetrieval:      req.LogRetrieval,
			UseComposite:      req.UseComposite,
		})
		if err != nil {
			return PrepareResult{}, err
		}
		if msg := MemoryContextMessage(memories); len(msg.Content) > 0 {
			messages = insertAfterSystem(messages, msg)
		}
	}

	tools := h.toolsFor(req.UserID, req.KGID)
	defs := make([]harness.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		defs = append(defs, tool.Definition())
	}
	return PrepareResult{
		Messages:            messages,
		Tools:               defs,
		Memories:            memories,
		AlreadyFoundPairIDs: append([]string(nil), memories.IDs...),
	}, nil
}

func (h *Harness) Run(ctx context.Context, req RunRequest, handler agent.EventHandler) (RunResult, error) {
	if h.model == nil {
		return RunResult{}, errors.New("chatv2: model is required")
	}
	prepared, err := h.Prepare(ctx, req.PrepareRequest)
	if err != nil {
		return RunResult{}, err
	}
	loop := agent.NewLoop(agent.Options{
		Model: h.model,
		Tools: h.toolsFor(req.UserID, req.KGID),
	})
	result, err := loop.RunStreaming(ctx, agent.RunRequest{
		UserID:    req.UserID,
		KGID:      req.KGID,
		SessionID: req.SessionID,
		Messages:  prepared.Messages,
		MaxTurns:  req.MaxTurns,
	}, handler)
	if err != nil {
		return RunResult{}, err
	}

	var saved *harness.Memory
	if req.SaveMemory && h.memory != nil && result.Text != "" {
		mem, err := h.memory.SaveMemory(ctx, memory.SaveMemoryRequest{
			UserID:            req.UserID,
			KGID:              firstNonEmpty(req.KGID, harness.KGID(req.UserID)),
			SessionID:         firstNonEmpty(req.SessionID, harness.MainSessionID),
			Prompt:            req.UserInput,
			Response:          result.Text,
			Input:             lastUserInput(prepared.Messages, req.UserInput),
			Output:            []harness.Content{{Type: harness.ContentTypeText, Content: result.Text}},
			Source:            firstNonEmpty(req.Source, "chatv2"),
			SourceContext:     req.SourceContext,
			Timestamp:         req.Timestamp,
			TimezoneOffset:    req.TimezoneOffset,
			SeedMemories:      prepared.Memories.SeedMemoryNodes,
			RetrievalMetadata: prepared.Memories.RetrievalMetadata,
			Subjects:          req.Subjects,
		})
		if err != nil {
			return RunResult{}, err
		}
		saved = &mem
	}
	return RunResult{RunResult: result, Preparation: prepared.Memories, SavedMemory: saved}, nil
}

func (h *Harness) Loop(req PrepareRequest) *agent.Loop {
	return agent.NewLoop(agent.Options{
		Model: h.model,
		Tools: h.toolsFor(req.UserID, req.KGID),
	})
}

func (h *Harness) toolsFor(userID, kgID string) []harness.Tool {
	tools := append([]harness.Tool(nil), h.tools...)
	if h.includeMemoryTools && h.memory != nil {
		tools = append(tools, memory.Tools(memory.ToolOptions{Store: h.memory, UserID: userID, KGID: kgID})...)
	}
	return tools
}

func MemoryContextMessage(memories memory.PromptMemoryResponse) harness.ChatMessage {
	if len(memories.LongTerm) == 0 && len(memories.ShortTerm) == 0 {
		return harness.ChatMessage{}
	}
	payload := map[string]any{}
	if len(memories.LongTerm) > 0 {
		payload["longTerm"] = json.RawMessage(memories.LongTermJSON)
	}
	if len(memories.ShortTerm) > 0 {
		payload["shortTerm"] = compactMemories(memories.ShortTerm)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return harness.ChatMessage{}
	}
	return harness.ChatMessage{
		Role: "system",
		Content: []harness.Content{{
			Type:    harness.ContentTypeText,
			Content: "Relevant memory context for this turn:\n" + string(raw),
		}},
	}
}

func normalizeMessages(messages []harness.ChatMessage, userInput, systemPrompt string) []harness.ChatMessage {
	out := append([]harness.ChatMessage(nil), messages...)
	if strings.TrimSpace(systemPrompt) != "" {
		out = append([]harness.ChatMessage{{
			Role:    "system",
			Content: []harness.Content{{Type: harness.ContentTypeText, Content: systemPrompt}},
		}}, out...)
	}
	if len(out) == 0 && strings.TrimSpace(userInput) != "" {
		out = append(out, harness.ChatMessage{
			Role:    "user",
			Content: []harness.Content{{Type: harness.ContentTypeText, Content: userInput}},
		})
	}
	return out
}

func insertAfterSystem(messages []harness.ChatMessage, msg harness.ChatMessage) []harness.ChatMessage {
	out := make([]harness.ChatMessage, 0, len(messages)+1)
	inserted := false
	for i, existing := range messages {
		if !inserted && i > 0 && existing.Role != "system" {
			out = append(out, msg)
			inserted = true
		}
		out = append(out, existing)
	}
	if !inserted {
		out = append(out, msg)
	}
	return out
}

func compactMemories(memories []harness.Memory) []map[string]any {
	out := make([]map[string]any, 0, len(memories))
	for _, mem := range memories {
		item := map[string]any{
			"pairID":    mem.ID,
			"timestamp": mem.Timestamp.UTC().Format(time.RFC3339),
			"title":     memory.PromptMemoryTitle(mem),
		}
		if mem.Summary != "" {
			item["summary"] = mem.Summary
		}
		out = append(out, item)
	}
	return out
}

func lastUserInput(messages []harness.ChatMessage, fallback string) []harness.Content {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" && len(messages[i].Content) > 0 {
			return append([]harness.Content(nil), messages[i].Content...)
		}
	}
	if fallback == "" {
		return nil
	}
	return []harness.Content{{Type: harness.ContentTypeText, Content: fallback}}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
