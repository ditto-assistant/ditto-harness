// SPDX-License-Identifier: AGPL-3.0-or-later
package memory

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ditto-assistant/ditto-harness/pkg/harness"
)

type ToolOptions struct {
	UserID        string
	KGID          string
	Store         *Store
	PreviewLen    int
	FetchMaxBytes int
}

func Tools(opts ToolOptions) []harness.Tool {
	return []harness.Tool{
		SaveMemoryTool(opts),
		SearchMemoriesTool(opts),
		SearchSubjectsTool(opts),
		SearchMemoriesInSubjectsTool(opts),
		FetchMemoriesTool(opts),
	}
}

func SaveMemoryTool(opts ToolOptions) harness.Tool {
	return tool{
		name:        "save_memory",
		description: "Save a durable memory for the current user.",
		schema:      rawSchema(`{"type":"object","properties":{"prompt":{"type":"string"},"response":{"type":"string"},"summary":{"type":"string"},"sessionId":{"type":"string"},"subjects":{"type":"array","items":{"type":"object","properties":{"text":{"type":"string"},"description":{"type":"string"},"key":{"type":"boolean"}}}}}}`),
		call: func(ctx context.Context, raw json.RawMessage) (any, error) {
			var args struct {
				Prompt    string         `json:"prompt"`
				Response  string         `json:"response"`
				Summary   string         `json:"summary"`
				SessionID string         `json:"sessionId"`
				Subjects  []SubjectInput `json:"subjects"`
			}
			if err := json.Unmarshal(raw, &args); err != nil {
				return nil, err
			}
			mem, err := opts.Store.SaveMemory(ctx, SaveMemoryRequest{
				UserID:    opts.UserID,
				KGID:      opts.KGID,
				SessionID: args.SessionID,
				Prompt:    args.Prompt,
				Response:  args.Response,
				Summary:   args.Summary,
				Subjects:  args.Subjects,
				Source:    "agent_tool",
			})
			if err != nil {
				return nil, err
			}
			return map[string]any{"memory": mem}, nil
		},
	}
}

func SearchMemoriesTool(opts ToolOptions) harness.Tool {
	return tool{
		name:        "search_memories",
		description: "Search past memories and return compact memory objects. Use fetch_memories for selected IDs that need full text.",
		schema:      rawSchema(`{"type":"object","required":["queries"],"properties":{"queries":{"type":"array","items":{"type":"string"}},"topK":{"type":"integer"},"sessionId":{"type":"string"}}}`),
		call: func(ctx context.Context, raw json.RawMessage) (any, error) {
			var args struct {
				Queries   []string `json:"queries"`
				TopK      int      `json:"topK"`
				SessionID string   `json:"sessionId"`
			}
			if err := json.Unmarshal(raw, &args); err != nil {
				return nil, err
			}
			memories, err := opts.Store.SearchMemories(ctx, SearchMemoriesRequest{
				UserID:    opts.UserID,
				KGID:      opts.KGID,
				SessionID: args.SessionID,
				Queries:   args.Queries,
				Limit:     args.TopK,
			})
			if err != nil {
				return nil, err
			}
			return map[string]any{"memories": SlimPreviews(memories, opts.previewLen())}, nil
		},
	}
}

func SearchSubjectsTool(opts ToolOptions) harness.Tool {
	return tool{
		name:        "search_subjects",
		description: "Search the user's subject graph and return subject objects with ids for subject-scoped memory search.",
		schema:      rawSchema(`{"type":"object","required":["queries"],"properties":{"queries":{"type":"array","items":{"type":"string"}},"topK":{"type":"integer"}}}`),
		call: func(ctx context.Context, raw json.RawMessage) (any, error) {
			var args struct {
				Queries []string `json:"queries"`
				TopK    int      `json:"topK"`
			}
			if err := json.Unmarshal(raw, &args); err != nil {
				return nil, err
			}
			subjects, err := opts.Store.SearchSubjects(ctx, SearchSubjectsRequest{
				UserID:  opts.UserID,
				KGID:    opts.KGID,
				Queries: args.Queries,
				Limit:   args.TopK,
			})
			if err != nil {
				return nil, err
			}
			return map[string]any{"subjects": subjects}, nil
		},
	}
}

func SearchMemoriesInSubjectsTool(opts ToolOptions) harness.Tool {
	return tool{
		name:        "search_memories_in_subjects",
		description: "Search memories inside one or more subject IDs.",
		schema:      rawSchema(`{"type":"object","required":["queries"],"properties":{"subject_id":{"type":"string"},"queries":{"type":"array","items":{"type":"string"}},"topK":{"type":"integer"}}}`),
		call: func(ctx context.Context, raw json.RawMessage) (any, error) {
			var args struct {
				SubjectID string   `json:"subject_id"`
				Queries   []string `json:"queries"`
				TopK      int      `json:"topK"`
			}
			if err := json.Unmarshal(raw, &args); err != nil {
				return nil, err
			}
			queries := make([]SubjectMemoryQuery, len(args.Queries))
			for i, query := range args.Queries {
				queries[i] = SubjectMemoryQuery{SubjectID: args.SubjectID, Query: query}
			}
			memories, err := opts.Store.SearchMemoriesInSubjects(ctx, SearchMemoriesInSubjectsRequest{
				UserID:  opts.UserID,
				Queries: queries,
				Limit:   args.TopK,
			})
			if err != nil {
				return nil, err
			}
			return map[string]any{"memories": SlimPreviews(memories, opts.previewLen())}, nil
		},
	}
}

func FetchMemoriesTool(opts ToolOptions) harness.Tool {
	return tool{
		name:        "fetch_memories",
		description: "Fetch full memory content for selected memory pair IDs.",
		schema:      rawSchema(`{"type":"object","required":["pairIds"],"properties":{"pairIds":{"type":"array","items":{"type":"string"}}}}`),
		call: func(ctx context.Context, raw json.RawMessage) (any, error) {
			var args struct {
				PairIDs []string `json:"pairIds"`
			}
			if err := json.Unmarshal(raw, &args); err != nil {
				return nil, err
			}
			memories, err := opts.Store.FetchMemories(ctx, FetchMemoriesRequest{UserID: opts.UserID, PairIDs: args.PairIDs})
			if err != nil {
				return nil, err
			}
			return map[string]any{"memories": SlimTruncated(memories, opts.fetchMaxBytes())}, nil
		},
	}
}

func (opts ToolOptions) previewLen() int {
	if opts.PreviewLen > 0 {
		return opts.PreviewLen
	}
	return DefaultPreviewLen
}

func (opts ToolOptions) fetchMaxBytes() int {
	if opts.FetchMaxBytes > 0 {
		return opts.FetchMaxBytes
	}
	return DefaultFetchMaxBytes
}

type tool struct {
	name        string
	description string
	schema      json.RawMessage
	call        func(context.Context, json.RawMessage) (any, error)
}

func (t tool) Definition() harness.ToolDefinition {
	return harness.ToolDefinition{Name: t.name, Description: t.description, InputSchema: t.schema}
}

func (t tool) Call(ctx context.Context, raw json.RawMessage) (harness.ToolCallResponse, error) {
	result, err := t.call(ctx, raw)
	if err != nil {
		return harness.ToolCallResponse{Name: t.name, Error: err.Error()}, err
	}
	output, err := json.Marshal(result)
	if err != nil {
		return harness.ToolCallResponse{Name: t.name, Error: err.Error()}, err
	}
	return harness.ToolCallResponse{Name: t.name, Output: output}, nil
}

func rawSchema(s string) json.RawMessage {
	if !json.Valid([]byte(s)) {
		panic(fmt.Sprintf("invalid tool schema: %s", s))
	}
	return json.RawMessage(s)
}
