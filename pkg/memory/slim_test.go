package memory

import (
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/ditto-assistant/ditto-harness/pkg/harness"
)

func TestSlimMemoryPreviewPrefersSummary(t *testing.T) {
	mem := harness.Memory{
		ID:        "pair-1",
		Summary:   strings.Repeat("summary ", 20),
		Prompt:    "full prompt should not be included",
		Response:  "full response should not be included",
		Timestamp: time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC),
	}
	got := ToSlimMemoryPreview(mem, 32)
	if got.ID != "pair-1" || got.Timestamp == "" || got.CharLen == 0 {
		t.Fatalf("incomplete preview: %+v", got)
	}
	if got.User != "" || got.Ditto != "" {
		t.Fatalf("preview should not include full text: %+v", got)
	}
	if !strings.HasSuffix(got.Preview, "...") || strings.Contains(got.Preview, "full prompt") {
		t.Fatalf("unexpected preview text: %q", got.Preview)
	}
}

func TestSlimMemoryTruncatedKeepsValidUTF8(t *testing.T) {
	mem := harness.Memory{
		ID:       "pair-2",
		Prompt:   strings.Repeat("hello🙂", 200),
		Response: strings.Repeat("world🙂", 200),
	}
	got := ToSlimMemoryTruncated(mem, 80)
	if got.Summary != "" || got.Preview != "" {
		t.Fatalf("truncated fetch should not include summary/preview: %+v", got)
	}
	if len(got.User) > 80 || len(got.Ditto) > 80 {
		t.Fatalf("truncated fields too large: user=%d ditto=%d", len(got.User), len(got.Ditto))
	}
	if !utf8.ValidString(got.User) || !utf8.ValidString(got.Ditto) {
		t.Fatalf("truncated strings must remain valid UTF-8: %+v", got)
	}
	if !strings.Contains(got.User, "truncated") || !strings.Contains(got.Ditto, "truncated") {
		t.Fatalf("missing truncation marker: %+v", got)
	}
}
