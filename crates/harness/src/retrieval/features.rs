// SPDX-License-Identifier: AGPL-3.0-or-later
//! Auxiliary feature extraction for the learned-weight predictor.
//! Port of Go `pkg/retrieval/features.go`.

use chrono::{DateTime, Timelike, Utc};

/// Dimension of the auxiliary feature vector (Go: `AuxFeatureDim`).
pub const AUX_FEATURE_DIM: usize = 17;
/// Dimension used by legacy model artifacts (Go: `LegacyAuxFeatureDim`).
pub const LEGACY_AUX_FEATURE_DIM: usize = 6;

/// Feature vector layout (indexes 0..17), matching Go exactly:
/// 0 normalized word count (words/15, capped at 1)
/// 1 has temporal keyword (0/1)
/// 2 has frequency keyword (0/1)
/// 3 temporal keyword count / 3, capped at 1
/// 4 frequency keyword count / 3, capped at 1
/// 5 has specificity keyword OR a mid-sentence Capitalized word (0/1)
/// 6..=10 question-type one-hot (info extraction, multi-session, knowledge
///        update, temporal reasoning, abstention)
/// 11 sin(2*pi*hourOfDay/24), 12 cos(...) — from `now` (UTC)
/// 13 ln(1 + seconds since last query)
/// 14 ln(1 + num pairs)
/// 15 ln(1 + days since signup)
/// 16 cosine(query embedding, user corpus centroid)
pub const QTYPE_INFO_EXTRACTION_IDX: usize = 6;
pub const QTYPE_MULTI_SESSION_IDX: usize = 7;
pub const QTYPE_KNOWLEDGE_UPDATE_IDX: usize = 8;
pub const QTYPE_TEMPORAL_REASONING_IDX: usize = 9;
pub const QTYPE_ABSTENTION_IDX: usize = 10;
pub const AUX_HOUR_SIN_IDX: usize = 11;
pub const AUX_HOUR_COS_IDX: usize = 12;
pub const AUX_LOG_SECS_SINCE_LAST_QUERY_IDX: usize = 13;
pub const AUX_LOG_NUM_PAIRS_IDX: usize = 14;
pub const AUX_LOG_DAYS_SINCE_SIGNUP_IDX: usize = 15;
pub const AUX_QUERY_CORPUS_DRIFT_IDX: usize = 16;

/// Context for richer feature extraction (Go: `AuxFeatureContext`).
/// `None`/empty fields zero out the corresponding features.
#[derive(Debug, Clone, Default)]
pub struct AuxFeatureContext {
    /// Free-form question type; mapped to a one-hot index (see
    /// `question_type_one_hot_idx` rules in the Go source: suffix `_abs` or
    /// "abstention" -> abstention; contains "multi" -> multi-session;
    /// "knowledge" -> knowledge update; "temporal" -> temporal reasoning;
    /// "single"/"extraction"/"preference"/"user"/"assistant" -> info
    /// extraction; otherwise none).
    pub question_type: String,
    pub now: Option<DateTime<Utc>>,
    pub last_query_at: Option<DateTime<Utc>>,
    pub num_pairs: usize,
    pub signup_at: Option<DateTime<Utc>>,
    pub query_embedding: Vec<f32>,
    pub user_corpus_centroid: Vec<f32>,
}

/// Keyword lists copied verbatim from Go (substring matched on the lowercased
/// query; each matching keyword counts once).
pub(crate) const TEMPORAL_KEYWORDS: &[&str] = &[
    "yesterday",
    "today",
    "tonight",
    "this morning",
    "this afternoon",
    "last night",
    "last week",
    "last month",
    "last year",
    "recently",
    "lately",
    "just now",
    "earlier",
    "before",
    "ago",
    "previous",
    "prior",
    "past",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "hour",
    "minute",
    "day",
    "week",
    "month",
];

pub(crate) const FREQUENCY_KEYWORDS: &[&str] = &[
    "often",
    "always",
    "usually",
    "frequently",
    "regularly",
    "keep talking",
    "keep discussing",
    "keep mentioning",
    "repeatedly",
    "constantly",
    "continuously",
    "common",
    "typical",
    "normal",
    "standard",
    "pattern",
    "habit",
    "routine",
    "recurring",
    "again and again",
    "over and over",
    "all the time",
    "every time",
    "each time",
];

pub(crate) const SPECIFICITY_KEYWORDS: &[&str] = &[
    "about",
    "regarding",
    "concerning",
    "related to",
    "specifically",
    "exactly",
    "precisely",
    "what is",
    "what are",
    "how does",
    "how do",
    "explain",
    "describe",
    "tell me about",
];

/// Extracts auxiliary features with an empty context
/// (Go: `ExtractAuxiliaryFeatures`).
pub fn extract_auxiliary_features(query: &str) -> [f32; AUX_FEATURE_DIM] {
    extract_auxiliary_features_context(query, &AuxFeatureContext::default())
}

/// Extracts auxiliary features with context
/// (Go: `ExtractAuxiliaryFeaturesContext`). See the layout doc on
/// [`AUX_FEATURE_DIM`]'s sibling constants above. Hour-of-day features use
/// the UTC hour/minute of `ctx.now`.
pub fn extract_auxiliary_features_context(
    query: &str,
    ctx: &AuxFeatureContext,
) -> [f32; AUX_FEATURE_DIM] {
    let mut out = [0f32; AUX_FEATURE_DIM];
    let words: Vec<&str> = query.split_whitespace().collect();
    out[0] = (words.len() as f32 / 15.0).min(1.0);

    let lower = query.to_lowercase();
    let temporal_count = count_keyword_matches(&lower, TEMPORAL_KEYWORDS);
    out[1] = bool_to_float(temporal_count > 0);
    out[3] = (temporal_count as f32 / 3.0).min(1.0);

    let freq_count = count_keyword_matches(&lower, FREQUENCY_KEYWORDS);
    out[2] = bool_to_float(freq_count > 0);
    out[4] = (freq_count as f32 / 3.0).min(1.0);

    let spec_count = count_keyword_matches(&lower, SPECIFICITY_KEYWORDS);
    out[5] = bool_to_float(spec_count > 0 || has_named_entity_pattern(&words));

    if let Some(idx) = question_type_one_hot_idx(&ctx.question_type) {
        out[idx] = 1.0;
    }
    if let Some(now) = ctx.now {
        let hour = f64::from(now.hour()) + f64::from(now.minute()) / 60.0;
        let angle = 2.0 * std::f64::consts::PI * hour / 24.0;
        out[AUX_HOUR_SIN_IDX] = angle.sin() as f32;
        out[AUX_HOUR_COS_IDX] = angle.cos() as f32;
        if let Some(last) = ctx.last_query_at {
            let delta = duration_seconds(now - last).max(0.0);
            out[AUX_LOG_SECS_SINCE_LAST_QUERY_IDX] = delta.ln_1p() as f32;
        }
    }
    if ctx.num_pairs > 0 {
        out[AUX_LOG_NUM_PAIRS_IDX] = (ctx.num_pairs as f64).ln_1p() as f32;
    }
    if let (Some(signup), Some(now)) = (ctx.signup_at, ctx.now) {
        let days = (duration_seconds(now - signup) / (24.0 * 3600.0)).max(0.0);
        out[AUX_LOG_DAYS_SINCE_SIGNUP_IDX] = days.ln_1p() as f32;
    }
    if let Some(drift) =
        crate::db::cosine_similarity(&ctx.query_embedding, &ctx.user_corpus_centroid)
    {
        out[AUX_QUERY_CORPUS_DRIFT_IDX] = drift;
    }
    out
}

/// Fractional seconds in a chrono duration (Go: `time.Duration.Seconds()`).
pub(crate) fn duration_seconds(d: chrono::Duration) -> f64 {
    match d.num_microseconds() {
        Some(us) => us as f64 / 1e6,
        None => d.num_milliseconds() as f64 / 1e3,
    }
}

/// Maps a free-form question type onto its one-hot index
/// (Go: `questionTypeOneHotIdx`; `None` leaves the one-hot all zero).
fn question_type_one_hot_idx(qtype: &str) -> Option<usize> {
    if qtype.is_empty() {
        return None;
    }
    let q = qtype.trim().to_lowercase();
    if q.ends_with("_abs") || q == "abstention" {
        return Some(QTYPE_ABSTENTION_IDX);
    }
    if q.contains("multi") {
        return Some(QTYPE_MULTI_SESSION_IDX);
    }
    if q.contains("knowledge") {
        return Some(QTYPE_KNOWLEDGE_UPDATE_IDX);
    }
    if q.contains("temporal") {
        return Some(QTYPE_TEMPORAL_REASONING_IDX);
    }
    if q.contains("single")
        || q.contains("extraction")
        || q.contains("preference")
        || q.contains("user")
        || q.contains("assistant")
    {
        return Some(QTYPE_INFO_EXTRACTION_IDX);
    }
    None
}

/// Number of keywords appearing as substrings of `text`
/// (Go: `countKeywordMatches`).
fn count_keyword_matches(text: &str, keywords: &[&str]) -> usize {
    keywords.iter().filter(|kw| text.contains(*kw)).count()
}

/// True when a word after the first starts a Capitalized-lowercase pattern
/// mid-sentence (Go: `hasNamedEntityPattern`). Words following sentence
/// punctuation (`.`, `!`, `?`) are skipped.
fn has_named_entity_pattern(words: &[&str]) -> bool {
    for i in 1..words.len() {
        if let Some(last) = words[i - 1].chars().last() {
            if last == '.' || last == '!' || last == '?' {
                continue;
            }
        }
        let mut chars = words[i].chars();
        if let (Some(first), Some(second)) = (chars.next(), chars.next()) {
            if first.is_uppercase() && second.is_lowercase() {
                return true;
            }
        }
    }
    false
}

fn bool_to_float(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// Port of Go `TestExtractAuxiliaryFeaturesContext`.
    #[test]
    fn extract_auxiliary_features_context_populates_features() {
        let now = Utc
            .with_ymd_and_hms(2026, 1, 1, 12, 0, 0)
            .single()
            .expect("timestamp");
        let got = extract_auxiliary_features_context(
            "what did Peyton mention recently",
            &AuxFeatureContext {
                question_type: "temporal_reasoning".to_string(),
                now: Some(now),
                last_query_at: Some(now - chrono::Duration::hours(1)),
                num_pairs: 10,
                signup_at: Some(now - chrono::Duration::hours(24)),
                query_embedding: vec![1.0, 0.0],
                user_corpus_centroid: vec![1.0, 0.0],
            },
        );
        assert!(
            got[0] != 0.0 && got[1] != 0.0 && got[QTYPE_TEMPORAL_REASONING_IDX] == 1.0,
            "expected query length, temporal, and qtype features: {got:?}"
        );
        assert_eq!(
            got[AUX_QUERY_CORPUS_DRIFT_IDX], 1.0,
            "query corpus drift should be 1"
        );
    }

    #[test]
    fn empty_context_zeroes_contextual_features() {
        let got =
            extract_auxiliary_features("describe the standard routine I keep mentioning every day");
        // Specificity ("describe"), temporal ("day"), frequency keywords hit.
        assert_eq!(got[5], 1.0);
        assert!(got[1] == 1.0 && got[2] == 1.0);
        // Contextual features stay zero without context.
        for idx in [
            AUX_HOUR_SIN_IDX,
            AUX_HOUR_COS_IDX,
            AUX_LOG_SECS_SINCE_LAST_QUERY_IDX,
            AUX_LOG_NUM_PAIRS_IDX,
            AUX_LOG_DAYS_SINCE_SIGNUP_IDX,
            AUX_QUERY_CORPUS_DRIFT_IDX,
        ] {
            assert_eq!(got[idx], 0.0, "feature {idx} should be zero");
        }
    }

    #[test]
    fn named_entity_pattern_skips_sentence_starts() {
        // Mid-sentence Capitalized word counts...
        assert_eq!(extract_auxiliary_features("what did Peyton say")[5], 1.0);
        // ...but a capital right after sentence punctuation does not.
        assert_eq!(extract_auxiliary_features("ok. Peyton")[5], 0.0);
        // Leading capital (first word) alone does not count.
        assert_eq!(extract_auxiliary_features("Peyton said hi")[5], 0.0);
    }

    #[test]
    fn question_type_mapping_matches_go() {
        assert_eq!(
            question_type_one_hot_idx("multi_session"),
            Some(QTYPE_MULTI_SESSION_IDX)
        );
        assert_eq!(
            question_type_one_hot_idx("knowledge_update"),
            Some(QTYPE_KNOWLEDGE_UPDATE_IDX)
        );
        assert_eq!(
            question_type_one_hot_idx("single_session_abs"),
            Some(QTYPE_ABSTENTION_IDX)
        );
        assert_eq!(
            question_type_one_hot_idx("abstention"),
            Some(QTYPE_ABSTENTION_IDX)
        );
        assert_eq!(
            question_type_one_hot_idx("user_preference"),
            Some(QTYPE_INFO_EXTRACTION_IDX)
        );
        assert_eq!(question_type_one_hot_idx(""), None);
        assert_eq!(question_type_one_hot_idx("unknown-kind"), None);
    }
}
