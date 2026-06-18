// SPDX-License-Identifier: AGPL-3.0-or-later
package harness

import (
	"context"
	"encoding/json"
	"time"
)

const (
	DefaultKGPrefix = "user_memories_"
	MainSessionID   = "main"
)

type ContentType string

const (
	ContentTypeText       ContentType = "text"
	ContentTypeMarkdown   ContentType = "text/markdown"
	ContentTypeToolCall   ContentType = "tool_call"
	ContentTypeToolResult ContentType = "tool_result"
)

type ToolCall struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type ToolCallResponse struct {
	ID     string          `json:"id"`
	Name   string          `json:"name,omitempty"`
	Output json.RawMessage `json:"output,omitempty"`
	Error  string          `json:"error,omitempty"`
}

type Content struct {
	Type             ContentType       `json:"type,omitempty"`
	Content          string            `json:"content,omitempty"`
	ToolCall         *ToolCall         `json:"toolCall,omitempty"`
	ToolCallResponse *ToolCallResponse `json:"toolCallResponse,omitempty"`
	Metadata         map[string]any    `json:"metadata,omitempty"`
}

type SeedMemoryNode struct {
	PairID   string           `json:"pairId"`
	Children []SeedMemoryNode `json:"children,omitempty"`
}

type RetrievalMetadata struct {
	Intent              string             `json:"intent,omitempty"`
	Weights             map[string]float64 `json:"weights,omitempty"`
	Scale               float64            `json:"scale,omitempty"`
	Variant             string             `json:"variant,omitempty"`
	RetrievedPairIDs    []string           `json:"retrievedPairIds,omitempty"`
	QueryEmbeddingModel string             `json:"queryEmbeddingModel,omitempty"`
}

type Memory struct {
	ID                string             `json:"id"`
	SourcePairID      string             `json:"sourcePairId,omitempty"`
	UserID            string             `json:"userId,omitempty"`
	KGID              string             `json:"kgId,omitempty"`
	SessionID         string             `json:"sessionId,omitempty"`
	Title             string             `json:"title,omitempty"`
	Summary           string             `json:"summary,omitempty"`
	Prompt            string             `json:"prompt,omitempty"`
	Response          string             `json:"response,omitempty"`
	Input             []Content          `json:"input,omitempty"`
	Output            []Content          `json:"output,omitempty"`
	Source            string             `json:"source,omitempty"`
	SourceContext     string             `json:"sourceContext,omitempty"`
	Timestamp         time.Time          `json:"timestamp"`
	TimezoneOffset    int                `json:"timezoneOffset,omitempty"`
	SeedMemories      []SeedMemoryNode   `json:"seedMemories,omitempty"`
	RetrievalMetadata *RetrievalMetadata `json:"retrievalMetadata,omitempty"`
	Embedding         []float32          `json:"-"`
	Similarity        float64            `json:"similarity,omitempty"`
	RecencyScore      float64            `json:"recencyScore,omitempty"`
	FrequencyScore    float64            `json:"frequencyScore,omitempty"`
	CompositeScore    float64            `json:"compositeScore,omitempty"`
	RecencyExp        float64            `json:"recencyExp,omitempty"`
	SubjectSemMatch   float64            `json:"subjectSemMatch,omitempty"`
	SessionContinuity float64            `json:"sessionContinuity,omitempty"`
	NeighborDensity   float64            `json:"neighborDensity,omitempty"`
}

type Subject struct {
	ID          string    `json:"id"`
	UserID      string    `json:"userId,omitempty"`
	KGID        string    `json:"kgId,omitempty"`
	Text        string    `json:"text"`
	Description string    `json:"description,omitempty"`
	Key         bool      `json:"key,omitempty"`
	Embedding   []float32 `json:"-"`
	Similarity  float64   `json:"similarity,omitempty"`
	MemoryCount int64     `json:"memoryCount,omitempty"`
}

type Usage struct {
	Provider     string `json:"provider,omitempty"`
	Model        string `json:"model,omitempty"`
	InputTokens  int64  `json:"inputTokens,omitempty"`
	OutputTokens int64  `json:"outputTokens,omitempty"`
	TotalTokens  int64  `json:"totalTokens,omitempty"`
}

type Cost struct {
	Currency string  `json:"currency,omitempty"`
	Amount   float64 `json:"amount,omitempty"`
}

type CostedUsage struct {
	Usage Usage `json:"usage"`
	Cost  Cost  `json:"cost"`
}

type EmbedRequest struct {
	Texts []string `json:"texts"`
}

type EmbedResponse struct {
	Embeddings [][]float32    `json:"embeddings"`
	Cost       *CostedUsage   `json:"cost,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

type Embedder interface {
	Embed(ctx context.Context, req EmbedRequest) (EmbedResponse, error)
}

type ChatMessage struct {
	Role       string     `json:"role"`
	Content    []Content  `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"toolCalls,omitempty"`
	ToolCallID string     `json:"toolCallId,omitempty"`
}

type ChatChunk struct {
	Text     string         `json:"text,omitempty"`
	ToolCall *ToolCall      `json:"toolCall,omitempty"`
	Cost     *CostedUsage   `json:"cost,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type Model interface {
	Next(ctx context.Context, messages []ChatMessage, tools []ToolDefinition) (ChatChunk, error)
}

type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

type Tool interface {
	Definition() ToolDefinition
	Call(ctx context.Context, raw json.RawMessage) (ToolCallResponse, error)
}

func KGID(userID string) string {
	return DefaultKGPrefix + userID
}
