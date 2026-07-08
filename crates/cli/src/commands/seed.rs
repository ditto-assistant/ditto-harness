// SPDX-License-Identifier: MIT
//! `seed` subcommand: ingest memories from a JSON file (array of
//! `{id?, prompt, response, summary?, sessionId?, daysAgo?, subjects?}`
//! objects) or built-in sample data for the fictional engineer "Quinn",
//! embedding each via the configured Ollama embedder.

use std::path::Path;

use anyhow::Context;
use chrono::{Duration, Utc};
use serde::Deserialize;

use ditto_harness::memory::{SaveMemoryRequest, SubjectInput};

use crate::commands::util;
use crate::Common;

/// One seedable memory. Stable `id`s make re-seeding idempotent (the store
/// upserts on `(user_id, firestore_pair_id)`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SeedEntry {
    #[serde(default)]
    id: String,
    prompt: String,
    response: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    session_id: String,
    /// Backdates the memory so recency scoring has signal.
    #[serde(default)]
    days_ago: Option<i64>,
    #[serde(default)]
    subjects: Vec<SubjectInput>,
}

pub async fn run(common: &Common, file: Option<&Path>) -> anyhow::Result<()> {
    let entries = match file {
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("read seed file {}", path.display()))?;
            serde_json::from_str::<Vec<SeedEntry>>(&raw).with_context(|| {
                format!(
                    "parse seed file {} as a JSON array of memories",
                    path.display()
                )
            })?
        }
        None => builtin_entries(),
    };
    if entries.is_empty() {
        anyhow::bail!("seed file contains no memories");
    }

    let store = util::open_store(common).await?;
    eprintln!(
        "seeding {} memories for user {:?} into {}",
        entries.len(),
        common.user,
        common.db
    );

    let now = Utc::now();
    let total = entries.len();
    for (idx, entry) in entries.into_iter().enumerate() {
        let label = if entry.summary.is_empty() {
            entry.prompt.clone()
        } else {
            entry.summary.clone()
        };
        let memory = store
            .save_memory(SaveMemoryRequest {
                user_id: common.user.clone(),
                session_id: entry.session_id,
                id: entry.id,
                summary: entry.summary,
                prompt: entry.prompt,
                response: entry.response,
                source: "seed".to_string(),
                timestamp: entry.days_ago.map(|days| now - Duration::days(days)),
                subjects: entry.subjects,
                ..SaveMemoryRequest::default()
            })
            .await
            .map_err(|err| util::ollama_hint(err, "save seed memory"))?;
        println!(
            "[{:>2}/{total}] {}  {}",
            idx + 1,
            memory.id,
            util::snippet(&label, 72)
        );
    }
    println!("seeded {total} memories for user {:?}", common.user);
    Ok(())
}

/// Built-in sample dataset: a coherent test user, "Quinn", a software
/// engineer with recurring topics (a Rust rewrite at work, a Mount Rainier
/// hiking trip, a sourdough hobby) that cross-reference each other so the
/// dream pipeline can extract real subjects. Stored as JSON and parsed
/// through the same `Vec<SeedEntry>` serde path as `--file`, so the built-in
/// data also exercises the file parser.
const QUINN_DATASET_JSON: &str = include_str!("seed_quinn.json");

fn builtin_entries() -> Vec<SeedEntry> {
    serde_json::from_str(QUINN_DATASET_JSON)
        .expect("embedded seed_quinn.json parses as Vec<SeedEntry>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_entries_are_coherent() {
        let entries = builtin_entries();
        assert_eq!(entries.len(), 12);
        let mut ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            12,
            "ids must be unique for idempotent re-seeding"
        );
        for entry in &entries {
            assert!(!entry.id.is_empty());
            assert!(!entry.prompt.is_empty());
            assert!(!entry.response.is_empty());
            assert!(!entry.summary.is_empty());
            assert!(entry.days_ago.is_some());
            assert!(!entry.subjects.is_empty());
        }
        // Recurring topics must repeat so dreaming/frequency scoring has signal.
        let count = |text: &str| {
            entries
                .iter()
                .filter(|e| e.subjects.iter().any(|s| s.text == text))
                .count()
        };
        assert!(count("Rust rewrite") >= 4);
        assert!(count("sourdough baking") >= 4);
        assert!(count("Mount Rainier trip") >= 3);
    }

    #[test]
    fn seed_file_format_parses() {
        let raw = r#"[
            {
                "prompt": "hello",
                "response": "hi there",
                "summary": "greeting",
                "sessionId": "s1",
                "daysAgo": 3,
                "subjects": [{"text": "greetings", "description": "salutations", "key": true}]
            },
            {"prompt": "minimal", "response": "entry"}
        ]"#;
        let entries: Vec<SeedEntry> = serde_json::from_str(raw).expect("parse seed JSON");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].session_id, "s1");
        assert_eq!(entries[0].days_ago, Some(3));
        assert!(entries[0].subjects[0].key);
        assert!(entries[1].id.is_empty());
        assert!(entries[1].subjects.is_empty());
    }
}
