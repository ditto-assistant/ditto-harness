// SPDX-License-Identifier: AGPL-3.0-or-later
package memory

import (
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/ditto-assistant/ditto-harness/pkg/harness"
)

const (
	DefaultPreviewLen    = 500
	DefaultFetchMaxBytes = 8000
)

type SlimMemory struct {
	ID            string `json:"id"`
	Timestamp     string `json:"timestamp"`
	Source        string `json:"source,omitempty"`
	SourceContext string `json:"sourceContext,omitempty"`
	User          string `json:"user,omitempty"`
	Ditto         string `json:"ditto,omitempty"`
	Summary       string `json:"summary,omitempty"`
	Preview       string `json:"preview,omitempty"`
	CharLen       int    `json:"charLen,omitempty"`
}

func ToSlimMemory(mem harness.Memory) SlimMemory {
	return SlimMemory{
		ID:            mem.ID,
		Timestamp:     FormatMemoryTimestamp(mem),
		Source:        mem.Source,
		SourceContext: mem.SourceContext,
		User:          UserTextPrompt(mem),
		Ditto:         AssistantTextResponse(mem),
		Summary:       mem.Summary,
		CharLen:       ComputeCharLength(mem),
	}
}

func ToSlimMemoryTruncated(mem harness.Memory, maxBytes int) SlimMemory {
	if maxBytes <= 0 {
		maxBytes = DefaultFetchMaxBytes
	}
	return SlimMemory{
		ID:            mem.ID,
		Timestamp:     FormatMemoryTimestamp(mem),
		Source:        mem.Source,
		SourceContext: mem.SourceContext,
		User:          MiddleTruncateUTF8(UserTextPrompt(mem), maxBytes),
		Ditto:         MiddleTruncateUTF8(AssistantTextResponse(mem), maxBytes),
		CharLen:       ComputeCharLength(mem),
	}
}

func ToSlimMemoryPreview(mem harness.Memory, previewLen int) SlimMemory {
	if previewLen <= 0 {
		previewLen = DefaultPreviewLen
	}
	out := SlimMemory{
		ID:            mem.ID,
		Timestamp:     FormatMemoryTimestamp(mem),
		Source:        mem.Source,
		SourceContext: mem.SourceContext,
		CharLen:       ComputeCharLength(mem),
	}
	if mem.Summary != "" {
		out.Preview = TruncateUTF8(mem.Summary, previewLen, "...")
		return out
	}
	out.Preview = ConversationTextPreview(mem, previewLen)
	return out
}

func SlimPreviews(memories []harness.Memory, previewLen int) []SlimMemory {
	out := make([]SlimMemory, 0, len(memories))
	for _, mem := range memories {
		out = append(out, ToSlimMemoryPreview(mem, previewLen))
	}
	return out
}

func SlimTruncated(memories []harness.Memory, maxBytes int) []SlimMemory {
	out := make([]SlimMemory, 0, len(memories))
	for _, mem := range memories {
		out = append(out, ToSlimMemoryTruncated(mem, maxBytes))
	}
	return out
}

func UserTextPrompt(mem harness.Memory) string {
	for _, content := range mem.Input {
		if content.Type == harness.ContentTypeText {
			return content.Content
		}
	}
	return mem.Prompt
}

func AssistantTextResponse(mem harness.Memory) string {
	textParts := make([]string, 0, len(mem.Output))
	for _, content := range mem.Output {
		if content.Type == harness.ContentTypeText {
			textParts = append(textParts, content.Content)
		}
	}
	if len(textParts) > 0 {
		return strings.Join(textParts, "\n")
	}
	return mem.Response
}

func FullTextContent(mem harness.Memory) string {
	userText := UserTextPrompt(mem)
	assistantText := AssistantTextResponse(mem)
	switch {
	case userText != "" && assistantText != "":
		return fmt.Sprintf("User: %s\n\nDitto: %s", userText, assistantText)
	case userText != "":
		return fmt.Sprintf("User: %s", userText)
	case assistantText != "":
		return fmt.Sprintf("Ditto: %s", assistantText)
	default:
		return ""
	}
}

func ConversationTextPreview(mem harness.Memory, maxLen int) string {
	if maxLen <= 0 {
		return ""
	}
	userText := MiddleTruncateUTF8(UserTextPrompt(mem), maxLen/2)
	assistantText := MiddleTruncateUTF8(AssistantTextResponse(mem), maxLen/2)
	switch {
	case userText != "" && assistantText != "":
		return fmt.Sprintf("User: %s\n\nDitto: %s", userText, assistantText)
	case userText != "":
		return fmt.Sprintf("User: %s", userText)
	case assistantText != "":
		return fmt.Sprintf("Ditto: %s", assistantText)
	default:
		return ""
	}
}

func ComputeCharLength(mem harness.Memory) int {
	const formattingOverhead = len("User: \n\nDitto: ")
	userLen := 0
	for _, content := range mem.Input {
		if content.Type == harness.ContentTypeText {
			userLen += len(content.Content)
		}
	}
	if len(mem.Input) == 0 {
		userLen = len(mem.Prompt)
	}
	assistantLen := 0
	for _, content := range mem.Output {
		if content.Type == harness.ContentTypeText {
			assistantLen += len(content.Content)
		}
	}
	if len(mem.Output) == 0 {
		assistantLen = len(mem.Response)
	}
	return userLen + assistantLen + formattingOverhead
}

func FormatMemoryTimestamp(mem harness.Memory) string {
	if mem.Timestamp.IsZero() {
		return ""
	}
	return mem.Timestamp.UTC().Format(time.RFC3339)
}

func TruncateUTF8(s string, maxBytes int, suffix string) string {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s
	}
	limit := maxBytes - len(suffix)
	if limit <= 0 {
		return suffix[:min(len(suffix), maxBytes)]
	}
	return trimToValidUTF8(s[:limit]) + suffix
}

func MiddleTruncateUTF8(s string, maxBytes int) string {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s
	}
	const marker = "\n...[truncated]...\n"
	if maxBytes <= len(marker) {
		return TruncateUTF8(s, maxBytes, "")
	}
	remaining := maxBytes - len(marker)
	prefixLen := remaining / 2
	suffixLen := remaining - prefixLen
	prefix := trimToValidUTF8(s[:prefixLen])
	suffix := trimLeadingToValidUTF8(s[len(s)-suffixLen:])
	return prefix + marker + suffix
}

func trimToValidUTF8(s string) string {
	for len(s) > 0 && !utf8.ValidString(s) {
		s = s[:len(s)-1]
	}
	return s
}

func trimLeadingToValidUTF8(s string) string {
	for len(s) > 0 && !utf8.ValidString(s) {
		_, size := utf8.DecodeRuneInString(s)
		if size <= 0 {
			return ""
		}
		s = s[size:]
	}
	return s
}
