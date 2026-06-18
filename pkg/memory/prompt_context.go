// SPDX-License-Identifier: AGPL-3.0-or-later
package memory

import (
	"encoding/json"
	"time"

	"github.com/ditto-assistant/ditto-harness/pkg/harness"
)

const promptDetailedSeedRootCount = 2

type PromptMemorySummary struct {
	PairID         string    `json:"pairID"`
	SessionID      string    `json:"sessionID,omitempty"`
	Source         string    `json:"source,omitempty"`
	Timestamp      time.Time `json:"timestamp"`
	Title          string    `json:"title,omitempty"`
	Children       int       `json:"children"`
	CosineSim      float64   `json:"cosineSim,omitempty"`
	RecencyScore   float64   `json:"recencyScore,omitempty"`
	FrequencyScore float64   `json:"frequencyScore,omitempty"`
	CompositeScore float64   `json:"compositeScore,omitempty"`
}

func ResolvePromptSessionID(sessionID string) string {
	if sessionID == "" {
		return harness.MainSessionID
	}
	return sessionID
}

func SummarizePromptMemories(memories []harness.Memory, limit int) []PromptMemorySummary {
	if limit <= 0 || len(memories) == 0 {
		return nil
	}
	if len(memories) < limit {
		limit = len(memories)
	}
	summary := make([]PromptMemorySummary, 0, limit)
	for _, mem := range memories[:limit] {
		summary = append(summary, PromptMemorySummary{
			PairID:         mem.ID,
			SessionID:      ResolvePromptSessionID(mem.SessionID),
			Source:         mem.Source,
			Timestamp:      mem.Timestamp,
			Title:          PromptMemoryTitle(mem),
			Children:       len(mem.SeedMemories),
			CosineSim:      mem.Similarity,
			CompositeScore: mem.CompositeScore,
		})
	}
	return summary
}

func BuildPromptLongTermJSON(memories []harness.Memory) string {
	if len(memories) == 0 {
		return `{"memories":[]}`
	}
	items := make([]map[string]any, 0, len(memories)*2)
	for i, mem := range memories {
		detailed := i < promptDetailedSeedRootCount
		addPromptMemoryJSON(&items, mem, "", detailed)
	}
	raw, err := json.Marshal(map[string]any{"memories": items})
	if err != nil {
		return `{"memories":[]}`
	}
	return string(raw)
}

func PromptMemoryTitle(mem harness.Memory) string {
	switch {
	case mem.Title != "":
		return mem.Title
	case mem.Summary != "":
		return mem.Summary
	case mem.Prompt != "":
		return mem.Prompt
	default:
		for _, part := range mem.Input {
			if part.Content != "" {
				return part.Content
			}
		}
		return ""
	}
}

func addPromptMemoryJSON(items *[]map[string]any, mem harness.Memory, parentID string, detailed bool) {
	item := map[string]any{
		"pairID":    mem.ID,
		"timestamp": mem.Timestamp.UTC().Format(time.RFC3339),
	}
	if parentID != "" {
		item["parent"] = parentID
	}
	if detailed {
		if mem.Summary != "" {
			item["summary"] = mem.Summary
		} else {
			item["user"] = contentText(mem.Input, mem.Prompt)
			item["ditto"] = contentText(mem.Output, mem.Response)
		}
		*items = append(*items, item)
		for _, child := range mem.SeedMemories {
			(*items) = append(*items, map[string]any{
				"pairID": child.PairID,
				"parent": mem.ID,
			})
		}
		return
	}
	item["title"] = PromptMemoryTitle(mem)
	*items = append(*items, item)
}

func contentText(parts []harness.Content, fallback string) string {
	if fallback != "" {
		return fallback
	}
	var out string
	for _, part := range parts {
		if part.Content != "" {
			out += part.Content
		}
	}
	return out
}
