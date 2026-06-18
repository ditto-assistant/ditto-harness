// SPDX-License-Identifier: AGPL-3.0-or-later
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

fn subject(text: &str, description: &str, key: bool) -> SubjectInput {
    SubjectInput {
        text: text.to_string(),
        description: description.to_string(),
        key,
    }
}

/// Built-in sample data: a coherent test user, "Quinn", a software engineer
/// with recurring topics (a Rust rewrite at work, a Mount Rainier hiking
/// trip, a sourdough hobby) that cross-reference each other so the dream
/// pipeline can extract real subjects.
fn builtin_entries() -> Vec<SeedEntry> {
    vec![
        SeedEntry {
            id: "seed-quinn-001".into(),
            prompt: "I'm Quinn, a software engineer. My team just approved my proposal to \
                     rewrite our Go memory service in Rust. Where should I start?"
                .into(),
            response: "Congrats! Start by treating the Go code as the spec: freeze its \
                       behavior with integration tests, then port the storage layer first \
                       since everything else depends on it. Keep the wire format identical \
                       so you can run both side by side."
                .into(),
            summary: "Quinn kicked off the Rust rewrite of the Go memory service; plan is \
                      to port the storage layer first behind identical wire formats."
                .into(),
            session_id: "work".into(),
            days_ago: Some(28),
            subjects: vec![
                subject(
                    "Rust rewrite",
                    "Quinn's project rewriting the Go memory service in Rust",
                    true,
                ),
                subject(
                    "Go memory service",
                    "The legacy Go service being replaced",
                    false,
                ),
            ],
        },
        SeedEntry {
            id: "seed-quinn-002".into(),
            prompt: "For the Rust rewrite, should we keep Postgres or try an embedded \
                     SQLite-family database with vector support?"
                .into(),
            response: "For a local-first harness, an embedded engine is compelling: zero \
                       ops, file-backed databases, and native F32 vector columns cover \
                       your cosine-similarity search. Spike it behind your storage trait \
                       and benchmark against Postgres before committing."
                .into(),
            summary: "Discussed database choice for the Rust rewrite: embedded \
                      SQLite-family engine with native vector columns vs Postgres."
                .into(),
            session_id: "work".into(),
            days_ago: Some(24),
            subjects: vec![
                subject(
                    "Rust rewrite",
                    "Quinn's project rewriting the Go memory service in Rust",
                    true,
                ),
                subject(
                    "vector search",
                    "Embedding similarity search design decisions",
                    false,
                ),
            ],
        },
        SeedEntry {
            id: "seed-quinn-003".into(),
            prompt: "The borrow checker is destroying me on the Rust rewrite. I keep \
                     fighting lifetimes in the async storage layer."
                .into(),
            response: "Classic week-three Rust. In async code, prefer owned data over \
                       references across await points: clone cheap handles, wrap shared \
                       state in Arc, and let connections be cloneable. Most lifetime \
                       fights in async traits disappear once you stop borrowing across \
                       awaits."
                .into(),
            summary: "Quinn fought the borrow checker in the async storage layer; advice \
                      was to use owned data and Arc instead of borrowing across await \
                      points."
                .into(),
            session_id: "work".into(),
            days_ago: Some(21),
            subjects: vec![
                subject(
                    "Rust rewrite",
                    "Quinn's project rewriting the Go memory service in Rust",
                    true,
                ),
                subject(
                    "borrow checker",
                    "Rust ownership and lifetime struggles",
                    false,
                ),
            ],
        },
        SeedEntry {
            id: "seed-quinn-004".into(),
            prompt: "Planning a three-day hiking trip around Mount Rainier in late \
                     September with my sister Ava. Which loop should we do?"
                .into(),
            response: "Late September is perfect: golden larches and fewer crowds. The \
                       Spray Park to Mowich Lake loop fits three days well. Book the \
                       Mowich Lake campsites early, and pack layers — afternoon storms \
                       roll in fast on the northwest side."
                .into(),
            summary: "Quinn planned a 3-day late-September Mount Rainier hiking trip with \
                      sister Ava; Spray Park - Mowich Lake loop recommended."
                .into(),
            session_id: "personal".into(),
            days_ago: Some(18),
            subjects: vec![
                subject(
                    "Mount Rainier trip",
                    "Quinn's 3-day September hiking trip with sister Ava",
                    true,
                ),
                subject("hiking", "Quinn's hiking hobby and trips", false),
            ],
        },
        SeedEntry {
            id: "seed-quinn-005".into(),
            prompt: "My old boots gave me blisters on the last trail. What should I look \
                     for in new ones before the Rainier trip?"
                .into(),
            response: "Fit beats brand: shop in the afternoon when feet are swollen, leave \
                       a thumb's width at the toe, and break them in for at least 30 trail \
                       miles before Rainier. A stiffer midsole helps with Spray Park's \
                       rocky stretches."
                .into(),
            summary: "Quinn shopped for new hiking boots ahead of the Mount Rainier trip; \
                      fit and break-in advice."
                .into(),
            session_id: "personal".into(),
            days_ago: Some(15),
            subjects: vec![
                subject(
                    "Mount Rainier trip",
                    "Quinn's 3-day September hiking trip with sister Ava",
                    true,
                ),
                subject(
                    "hiking gear",
                    "Boots and equipment for Quinn's hikes",
                    false,
                ),
            ],
        },
        SeedEntry {
            id: "seed-quinn-006".into(),
            prompt: "I started a sourdough starter! Named it Ferris. Day one: flour, \
                     water, hope. What's the feeding schedule?"
                .into(),
            response: "Welcome to the cult, and excellent name. Feed Ferris 1:1:1 \
                       (starter:flour:water by weight) once a day at room temperature. \
                       Expect sluggish bubbles by day three and real activity around day \
                       five to seven. The hooch smell is normal — just stir it in."
                .into(),
            summary: "Quinn started a sourdough starter named Ferris; got a 1:1:1 daily \
                      feeding schedule."
                .into(),
            session_id: "personal".into(),
            days_ago: Some(14),
            subjects: vec![subject(
                "sourdough baking",
                "Quinn's sourdough hobby; starter named Ferris",
                true,
            )],
        },
        SeedEntry {
            id: "seed-quinn-007".into(),
            prompt: "First sourdough loaf came out like a brick. Dense, gummy crumb. \
                     Ferris betrayed me. What went wrong?"
                .into(),
            response: "Ferris is innocent — a dense gummy crumb usually means underproofed \
                       dough or a starter that wasn't at peak. Use the float test before \
                       mixing, push bulk fermentation until the dough is jiggly and \
                       roughly 50% larger, and bake by internal temp (96-99C) rather than \
                       the clock."
                .into(),
            summary: "Quinn's first sourdough loaf was dense and gummy; diagnosed as \
                      underproofing, advised float test and longer bulk fermentation."
                .into(),
            session_id: "personal".into(),
            days_ago: Some(10),
            subjects: vec![subject(
                "sourdough baking",
                "Quinn's sourdough hobby; starter named Ferris",
                true,
            )],
        },
        SeedEntry {
            id: "seed-quinn-008".into(),
            prompt: "Update: bumped hydration to 75% and stretched bulk fermentation like \
                     you said. The new loaf has an open crumb and a crackly crust!"
                .into(),
            response: "That's the loaf! 75% hydration plus patient bulk fermentation is a \
                       great house recipe. Note today's timings and temperatures — \
                       repeatability is the real skill. Next experiment: 10% whole wheat \
                       for deeper flavor."
                .into(),
            summary: "Quinn's sourdough succeeded at 75% hydration with longer bulk \
                      fermentation; next step is adding whole wheat."
                .into(),
            session_id: "personal".into(),
            days_ago: Some(7),
            subjects: vec![subject(
                "sourdough baking",
                "Quinn's sourdough hobby; starter named Ferris",
                true,
            )],
        },
        SeedEntry {
            id: "seed-quinn-009".into(),
            prompt: "Preferences for our pairing sessions: I use Helix, dark mode \
                     everything, spaces over tabs, and I think best before 10am with \
                     black coffee."
                .into(),
            response: "Noted: Helix keybindings in examples, dark-mode-friendly \
                       screenshots, spaces in snippets, and we'll schedule deep work — \
                       like the Rust rewrite's storage layer — for your pre-10am coffee \
                       window."
                .into(),
            summary: "Quinn's working preferences: Helix editor, dark mode, spaces over \
                      tabs, deep work before 10am with black coffee."
                .into(),
            session_id: "work".into(),
            days_ago: Some(12),
            subjects: vec![
                subject(
                    "preferences",
                    "Quinn's tooling and working-style preferences",
                    true,
                ),
                subject(
                    "Rust rewrite",
                    "Quinn's project rewriting the Go memory service in Rust",
                    false,
                ),
            ],
        },
        SeedEntry {
            id: "seed-quinn-010".into(),
            prompt: "Vector search in the Rust rewrite is returning weird neighbors. \
                     Similarities cluster around 0.3 even for unrelated text."
                .into(),
            response: "Check three things: that you're converting distance to similarity \
                       (similarity = 1 - cosine distance), that query and document \
                       vectors come from the same embedding model, and that you're not \
                       truncating the 768-dim vectors when encoding them as little-endian \
                       blobs."
                .into(),
            summary: "Debugged vector search in the Rust rewrite: distance vs similarity \
                      confusion and embedding-model mismatch were the suspects."
                .into(),
            session_id: "work".into(),
            days_ago: Some(5),
            subjects: vec![
                subject(
                    "Rust rewrite",
                    "Quinn's project rewriting the Go memory service in Rust",
                    true,
                ),
                subject(
                    "vector search",
                    "Embedding similarity search design decisions",
                    false,
                ),
            ],
        },
        SeedEntry {
            id: "seed-quinn-011".into(),
            prompt: "Back from Rainier! Ava and I did the Spray Park loop. I baked a \
                     sourdough loaf as trail bread and spent an evening at camp \
                     explaining the Rust rewrite to her."
                .into(),
            response: "Sounds like a perfect trip: golden larches, Ferris-powered trail \
                       bread, and you know you've internalized the rewrite when you can \
                       explain ownership semantics around a campfire. How did the new \
                       boots hold up?"
                .into(),
            summary: "Quinn returned from the Mount Rainier trip with Ava; brought \
                      homemade sourdough as trail bread and talked through the Rust \
                      rewrite at camp."
                .into(),
            session_id: "personal".into(),
            days_ago: Some(2),
            subjects: vec![
                subject(
                    "Mount Rainier trip",
                    "Quinn's 3-day September hiking trip with sister Ava",
                    true,
                ),
                subject(
                    "sourdough baking",
                    "Quinn's sourdough hobby; starter named Ferris",
                    false,
                ),
                subject(
                    "Rust rewrite",
                    "Quinn's project rewriting the Go memory service in Rust",
                    false,
                ),
            ],
        },
        SeedEntry {
            id: "seed-quinn-012".into(),
            prompt: "Demo day: the Rust rewrite passed the full Go test suite and memory \
                     search latency dropped 40%. Team wants to ship it next sprint."
                .into(),
            response: "Huge milestone — passing the legacy suite is the real finish line, \
                       the latency win is the bonus. Ship it behind a flag, mirror \
                       traffic for a week, and celebrate properly: this calls for a fresh \
                       sourdough bake."
                .into(),
            summary: "The Rust rewrite passed the full Go test suite with 40% faster \
                      memory search; shipping next sprint behind a flag."
                .into(),
            session_id: "work".into(),
            days_ago: Some(1),
            subjects: vec![
                subject(
                    "Rust rewrite",
                    "Quinn's project rewriting the Go memory service in Rust",
                    true,
                ),
                subject(
                    "vector search",
                    "Embedding similarity search design decisions",
                    false,
                ),
            ],
        },
    ]
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
