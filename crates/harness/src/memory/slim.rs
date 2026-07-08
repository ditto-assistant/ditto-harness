// SPDX-License-Identifier: MIT
//! Slim, token-efficient memory payloads for tool results.

use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};

use crate::types::{ContentType, Memory};

/// Default preview length in bytes.
pub const DEFAULT_PREVIEW_LEN: usize = 500;
/// Default per-field byte budget for fetched memories.
pub const DEFAULT_FETCH_MAX_BYTES: usize = 8000;

/// Truncation marker inserted by [`middle_truncate_utf8`].
const MIDDLE_TRUNCATION_MARKER: &str = "\n...[truncated]...\n";

/// Compact memory representation returned by memory tools. JSON field names
/// are camelCase on the wire.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlimMemory {
    #[serde(default)]
    pub id: String,
    /// RFC3339 (UTC) or empty for a zero timestamp; always serialized.
    #[serde(default)]
    pub timestamp: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source_context: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub user: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub ditto: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub summary: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub preview: String,
    #[serde(default, skip_serializing_if = "crate::types::is_zero_i64")]
    pub char_len: i64,
}

/// Full slim view: user/ditto text untruncated, plus summary.
pub fn to_slim_memory(mem: &Memory) -> SlimMemory {
    SlimMemory {
        id: mem.id.clone(),
        timestamp: format_memory_timestamp(mem),
        source: mem.source.clone(),
        source_context: mem.source_context.clone(),
        user: user_text_prompt(mem),
        ditto: assistant_text_response(mem),
        summary: mem.summary.clone(),
        char_len: compute_char_length(mem),
        ..SlimMemory::default()
    }
}

/// Slim view with middle-truncated user/ditto text; `max_bytes == 0` uses
/// [`DEFAULT_FETCH_MAX_BYTES`].
pub fn to_slim_memory_truncated(mem: &Memory, max_bytes: usize) -> SlimMemory {
    let max_bytes = if max_bytes == 0 {
        DEFAULT_FETCH_MAX_BYTES
    } else {
        max_bytes
    };
    SlimMemory {
        id: mem.id.clone(),
        timestamp: format_memory_timestamp(mem),
        source: mem.source.clone(),
        source_context: mem.source_context.clone(),
        user: middle_truncate_utf8(&user_text_prompt(mem), max_bytes),
        ditto: middle_truncate_utf8(&assistant_text_response(mem), max_bytes),
        char_len: compute_char_length(mem),
        ..SlimMemory::default()
    }
}

/// Preview-only slim view: `preview` is the truncated summary when present,
/// else a "User: ...\n\nDitto: ..." conversation preview; `preview_len == 0`
/// uses [`DEFAULT_PREVIEW_LEN`].
pub fn to_slim_memory_preview(mem: &Memory, preview_len: usize) -> SlimMemory {
    let preview_len = if preview_len == 0 {
        DEFAULT_PREVIEW_LEN
    } else {
        preview_len
    };
    let mut out = SlimMemory {
        id: mem.id.clone(),
        timestamp: format_memory_timestamp(mem),
        source: mem.source.clone(),
        source_context: mem.source_context.clone(),
        char_len: compute_char_length(mem),
        ..SlimMemory::default()
    };
    if !mem.summary.is_empty() {
        out.preview = truncate_utf8(&mem.summary, preview_len, "...");
        return out;
    }
    out.preview = conversation_text_preview(mem, preview_len);
    out
}

/// Maps memories to previews.
pub fn slim_previews(memories: &[Memory], preview_len: usize) -> Vec<SlimMemory> {
    memories
        .iter()
        .map(|mem| to_slim_memory_preview(mem, preview_len))
        .collect()
}

/// Maps memories to truncated slim views.
pub fn slim_truncated(memories: &[Memory], max_bytes: usize) -> Vec<SlimMemory> {
    memories
        .iter()
        .map(|mem| to_slim_memory_truncated(mem, max_bytes))
        .collect()
}

/// First text content part of `input`, falling back to `prompt`.
pub fn user_text_prompt(mem: &Memory) -> String {
    for content in &mem.input {
        if content.content_type == Some(ContentType::Text) {
            return content.content.clone();
        }
    }
    mem.prompt.clone()
}

/// All text content parts of `output` joined with newlines, falling back to
/// `response`.
pub fn assistant_text_response(mem: &Memory) -> String {
    let parts: Vec<&str> = mem
        .output
        .iter()
        .filter(|content| content.content_type == Some(ContentType::Text))
        .map(|content| content.content.as_str())
        .collect();
    if !parts.is_empty() {
        return parts.join("\n");
    }
    mem.response.clone()
}

/// "User: ...\n\nDitto: ..." rendering of the full conversation text;
/// degrades gracefully when one side is empty.
pub fn full_text_content(mem: &Memory) -> String {
    format_conversation(&user_text_prompt(mem), &assistant_text_response(mem))
}

/// Conversation preview with each side middle-truncated to `max_len / 2`
/// bytes; empty when `max_len == 0`.
pub fn conversation_text_preview(mem: &Memory, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }
    let user_text = middle_truncate_utf8(&user_text_prompt(mem), max_len / 2);
    let assistant_text = middle_truncate_utf8(&assistant_text_response(mem), max_len / 2);
    format_conversation(&user_text, &assistant_text)
}

fn format_conversation(user_text: &str, assistant_text: &str) -> String {
    match (user_text.is_empty(), assistant_text.is_empty()) {
        (false, false) => format!("User: {user_text}\n\nDitto: {assistant_text}"),
        (false, true) => format!("User: {user_text}"),
        (true, false) => format!("Ditto: {assistant_text}"),
        (true, true) => String::new(),
    }
}

/// Approximate rendered character length: user text bytes + assistant text
/// bytes + `len("User: \n\nDitto: ")` formatting overhead (falls back to
/// prompt/response lengths when input/output are empty).
pub fn compute_char_length(mem: &Memory) -> i64 {
    const FORMATTING_OVERHEAD: usize = "User: \n\nDitto: ".len();
    let mut user_len: usize = mem
        .input
        .iter()
        .filter(|content| content.content_type == Some(ContentType::Text))
        .map(|content| content.content.len())
        .sum();
    if mem.input.is_empty() {
        user_len = mem.prompt.len();
    }
    let mut assistant_len: usize = mem
        .output
        .iter()
        .filter(|content| content.content_type == Some(ContentType::Text))
        .map(|content| content.content.len())
        .sum();
    if mem.output.is_empty() {
        assistant_len = mem.response.len();
    }
    (user_len + assistant_len + FORMATTING_OVERHEAD) as i64
}

/// RFC3339 UTC timestamp, or empty string for the zero/unix-epoch timestamp
/// (`DateTime::<Utc>::UNIX_EPOCH`, the `Memory` default).
pub fn format_memory_timestamp(mem: &Memory) -> String {
    if mem.timestamp == DateTime::<Utc>::UNIX_EPOCH {
        return String::new();
    }
    mem.timestamp.to_rfc3339_opts(SecondsFormat::Secs, true)
}

/// Truncates to at most `max_bytes` bytes on a valid UTF-8 boundary,
/// appending `suffix` when truncation occurs.
/// `max_bytes == 0` disables truncation.
pub fn truncate_utf8(s: &str, max_bytes: usize, suffix: &str) -> String {
    if max_bytes == 0 || s.len() <= max_bytes {
        return s.to_string();
    }
    if max_bytes <= suffix.len() {
        return slice_to_boundary(suffix, suffix.len().min(max_bytes)).to_string();
    }
    let limit = max_bytes - suffix.len();
    let mut out = slice_to_boundary(s, limit).to_string();
    out.push_str(suffix);
    out
}

/// Keeps the head and tail of `s` within `max_bytes` bytes, joining with
/// `"\n...[truncated]...\n"`. `max_bytes == 0` disables truncation.
pub fn middle_truncate_utf8(s: &str, max_bytes: usize) -> String {
    if max_bytes == 0 || s.len() <= max_bytes {
        return s.to_string();
    }
    if max_bytes <= MIDDLE_TRUNCATION_MARKER.len() {
        return truncate_utf8(s, max_bytes, "");
    }
    let remaining = max_bytes - MIDDLE_TRUNCATION_MARKER.len();
    let prefix_len = remaining / 2;
    let suffix_len = remaining - prefix_len;
    let prefix = slice_to_boundary(s, prefix_len);
    let suffix = slice_from_boundary(s, s.len() - suffix_len);
    format!("{prefix}{MIDDLE_TRUNCATION_MARKER}{suffix}")
}

/// Largest prefix of `s` at most `max_bytes` long ending on a char boundary.
fn slice_to_boundary(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Suffix of `s` starting at or after byte `start`, on a char boundary.
fn slice_from_boundary(s: &str, start: usize) -> &str {
    if start >= s.len() {
        return "";
    }
    let mut begin = start;
    while begin < s.len() && !s.is_char_boundary(begin) {
        begin += 1;
    }
    &s[begin..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Content;
    use chrono::TimeZone;

    #[test]
    fn slim_memory_preview_prefers_summary() {
        let mem = Memory {
            id: "pair-1".to_string(),
            summary: "summary ".repeat(20),
            prompt: "full prompt should not be included".to_string(),
            response: "full response should not be included".to_string(),
            timestamp: Utc
                .with_ymd_and_hms(2026, 1, 1, 12, 0, 0)
                .single()
                .expect("timestamp"),
            ..Memory::default()
        };
        let got = to_slim_memory_preview(&mem, 32);
        assert_eq!(got.id, "pair-1");
        assert!(!got.timestamp.is_empty());
        assert_ne!(got.char_len, 0);
        assert!(
            got.user.is_empty() && got.ditto.is_empty(),
            "preview should not include full text: {got:?}"
        );
        assert!(
            got.preview.ends_with("...") && !got.preview.contains("full prompt"),
            "unexpected preview text: {:?}",
            got.preview
        );
    }

    #[test]
    fn slim_memory_truncated_keeps_valid_utf8() {
        let mem = Memory {
            id: "pair-2".to_string(),
            prompt: "hello🙂".repeat(200),
            response: "world🙂".repeat(200),
            ..Memory::default()
        };
        let got = to_slim_memory_truncated(&mem, 80);
        assert!(
            got.summary.is_empty() && got.preview.is_empty(),
            "truncated fetch should not include summary/preview: {got:?}"
        );
        assert!(
            got.user.len() <= 80 && got.ditto.len() <= 80,
            "truncated fields too large: user={} ditto={}",
            got.user.len(),
            got.ditto.len()
        );
        // Owned Strings are valid UTF-8 by construction; assert the marker.
        assert!(
            got.user.contains("truncated") && got.ditto.contains("truncated"),
            "missing truncation marker: {got:?}"
        );
        assert_eq!(got.timestamp, "", "epoch sentinel renders empty");
    }

    #[test]
    fn user_and_assistant_text_prefer_content_parts() {
        let mem = Memory {
            prompt: "fallback prompt".to_string(),
            response: "fallback response".to_string(),
            input: vec![Content::text("typed input")],
            output: vec![Content::text("part one"), Content::text("part two")],
            ..Memory::default()
        };
        assert_eq!(user_text_prompt(&mem), "typed input");
        assert_eq!(assistant_text_response(&mem), "part one\npart two");
        assert_eq!(
            full_text_content(&mem),
            "User: typed input\n\nDitto: part one\npart two"
        );
    }

    #[test]
    fn compute_char_length_falls_back_to_prompt_response() {
        let mem = Memory {
            prompt: "abcd".to_string(),
            response: "ef".to_string(),
            ..Memory::default()
        };
        assert_eq!(compute_char_length(&mem), 4 + 2 + 15);
    }

    #[test]
    fn truncate_utf8_respects_byte_budget_and_boundaries() {
        assert_eq!(truncate_utf8("short", 32, "..."), "short");
        assert_eq!(truncate_utf8("abcdefgh", 7, "..."), "abcd...");
        // 🙂 is 4 bytes; cutting mid-char must back up to a boundary.
        let truncated = truncate_utf8("ab🙂cd", 5, "...");
        assert!(truncated.len() <= 5);
        assert_eq!(truncated, "ab...");
        assert_eq!(truncate_utf8("abcdef", 2, "..."), "..".to_string());
    }

    #[test]
    fn conversation_preview_handles_one_sided_memories() {
        let user_only = Memory {
            prompt: "hello".to_string(),
            ..Memory::default()
        };
        assert_eq!(conversation_text_preview(&user_only, 100), "User: hello");
        let assistant_only = Memory {
            response: "hi".to_string(),
            ..Memory::default()
        };
        assert_eq!(conversation_text_preview(&assistant_only, 100), "Ditto: hi");
        assert_eq!(conversation_text_preview(&Memory::default(), 0), "");
    }
}
