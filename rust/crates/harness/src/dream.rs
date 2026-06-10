//! Dream pipeline: offline subject extraction/consolidation over a user's
//! stored memories, locally verifiable with Ollama (gemma3:4b +
//! embeddinggemma). New module — a scaled-down local port of Ditto's
//! production "dreaming pipeline".
//!
//! Stages:
//! 1. **EXTRACT** (the only LLM stage at write time): for each recent memory
//!    pair the model emits strict JSON
//!    `{"summary": "...", "subjects": [{"name","description","type"}]}` with
//!    0-5 durable subjects. Parsing is defensive (code fences stripped,
//!    malformed entries skipped, never panics). Pairs run with bounded
//!    concurrency.
//! 2. **DEDUP+CREATE**: extracted subjects are embedded and cosine-matched
//!    against the user's existing subject embeddings; best similarity >=
//!    [`SUBJECT_MERGE_THRESHOLD`] merges into the closest existing subject
//!    (keeping its id and canonical name, appending the new description with
//!    `" | "`), otherwise a new row is inserted. Exact-string duplicates are
//!    backstopped by the `UNIQUE(user_id, kg_id, subject_text)` index.
//! 3. **LINK**: one `subject_memory_pair_links` row per (subject, pair),
//!    `ON CONFLICT DO NOTHING` (safe retries). A subject that gains a linked
//!    pair flips `is_key_subject` on.
//! 4. **REFINE**: subjects whose description accumulated `" | "` merges get
//!    one LLM call to synthesize a canonical name (2-5 words) and a rolling
//!    narrative summary, then are re-embedded so future dedup matches the
//!    consolidated vector.
//!
//! Storage note: dream needs append/rename/re-embed semantics that exceed the
//! `Db::upsert_subject` contract (which only COALESCEs), so it runs its own
//! SQL through [`Db::connection`], like the retrieval module does.

use std::sync::Arc;

use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::db::{self, Db};
use crate::memory::Store;
use crate::types::{
    kg_id as derive_kg_id, ChatChunk, ChatMessage, Content, CostCollector, CostedUsage,
    EmbedRequest, Embedder, Error, Model, Result,
};

/// Default cap on memories considered per run (newest first).
pub const DEFAULT_DREAM_MAX_MEMORIES: usize = 128;
/// Default bound on concurrent LLM calls (extract + refine stages).
pub const DEFAULT_DREAM_CONCURRENCY: usize = 2;
/// Cosine similarity at or above which an extracted subject merges into the
/// closest existing subject. (Prod rationale: >0.85 fragments the graph,
/// <0.65 over-collapses.)
pub const SUBJECT_MERGE_THRESHOLD: f64 = 0.75;
/// Separator used when merge appends a description (deliberately ugly;
/// the refine stage cleans it).
pub const MERGE_SEPARATOR: &str = " | ";
/// Max subjects accepted from one extraction.
const MAX_SUBJECTS_PER_MEMORY: usize = 5;
/// Byte budget for the conversation text shown to the extraction model.
const MAX_CONVERSATION_BYTES: usize = 6000;

/// Runs dreams over a user's memories.
pub struct Dreamer {
    store: Arc<Store>,
    model: Arc<dyn Model>,
    embedder: Arc<dyn Embedder>,
}

/// Options for one dream run. Zero values pick the documented defaults.
#[derive(Debug, Clone, Default)]
pub struct DreamOptions {
    /// Empty -> `kg_id(user_id)`.
    pub kg_id: String,
    /// Max memories considered, newest first. `0` -> 128.
    pub max_memories: usize,
    /// Bound on concurrent LLM calls. `0` -> 2.
    pub concurrency: usize,
    /// Run the refine stage. `None` -> true.
    pub refine: Option<bool>,
}

/// Outcome of a dream run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DreamReport {
    pub memories_processed: usize,
    pub subjects_created: usize,
    pub subjects_merged: usize,
    pub links_created: usize,
    pub subjects_refined: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub usage: Vec<CostedUsage>,
}

/// A memory pair loaded for extraction.
struct PairLite {
    /// Internal row uuid (`memory_pairs.id`) — link target.
    id: String,
    conversation: String,
    has_description: bool,
}

/// In-memory view of a subject row used for cosine dedup.
struct SubjectEntry {
    id: String,
    description: String,
    embedding: Vec<f32>,
}

/// Parsed extraction payload.
#[derive(Debug, Clone, Default, PartialEq)]
struct Extraction {
    summary: String,
    subjects: Vec<ExtractedSubject>,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct ExtractedSubject {
    name: String,
    description: String,
}

impl Dreamer {
    /// Creates a dreamer over a store, chat model, and embedder.
    pub fn new(store: Arc<Store>, model: Arc<dyn Model>, embedder: Arc<dyn Embedder>) -> Dreamer {
        Dreamer {
            store,
            model,
            embedder,
        }
    }

    /// Dreams over `user_id`'s memories: extracts durable subjects per recent
    /// memory pair, dedups them against the existing subject graph (merging
    /// at >= [`SUBJECT_MERGE_THRESHOLD`] cosine similarity), links subjects
    /// to their source pairs, and optionally refines merge-accumulated
    /// subjects. Idempotent: re-running merges instead of duplicating and
    /// re-links are conflict-ignored.
    pub async fn dream(&self, user_id: &str, opts: DreamOptions) -> Result<DreamReport> {
        if user_id.trim().is_empty() {
            return Err(Error::InvalidArgument("dream: user_id is required".into()));
        }
        let kg_id = if opts.kg_id.is_empty() {
            derive_kg_id(user_id)
        } else {
            opts.kg_id.clone()
        };
        let max_memories = if opts.max_memories == 0 {
            DEFAULT_DREAM_MAX_MEMORIES
        } else {
            opts.max_memories
        };
        let concurrency = if opts.concurrency == 0 {
            DEFAULT_DREAM_CONCURRENCY
        } else {
            opts.concurrency
        };
        let refine = opts.refine.unwrap_or(true);

        let db = self.store.db();
        let mut report = DreamReport::default();
        let mut costs = CostCollector::default();

        // Stage EXTRACT.
        let pairs = load_recent_pairs(db, user_id, &kg_id, max_memories).await?;
        let extractions = self
            .run_model_stage(
                pairs
                    .iter()
                    .map(|pair| extraction_prompt(&pair.conversation))
                    .collect(),
                concurrency,
            )
            .await;

        let mut entries = load_subject_entries(db, user_id, &kg_id).await?;
        let mut first_model_error: Option<Error> = None;
        let mut any_model_ok = false;

        for (pair, outcome) in pairs.iter().zip(extractions) {
            let chunk = match outcome {
                Ok(chunk) => {
                    any_model_ok = true;
                    chunk
                }
                Err(err) => {
                    tracing::warn!(pair_id = %pair.id, error = %err, "dream: extraction call failed");
                    if first_model_error.is_none() {
                        first_model_error = Some(err);
                    }
                    continue;
                }
            };
            costs.add(chunk.cost.as_ref());
            report.memories_processed += 1;
            let Some(extraction) = parse_extraction(&chunk.text) else {
                tracing::warn!(pair_id = %pair.id, "dream: unparseable extraction output, skipping");
                continue;
            };

            if !pair.has_description && !extraction.summary.is_empty() {
                write_pair_summary(db, &pair.id, &extraction.summary).await?;
            }
            if extraction.subjects.is_empty() {
                continue;
            }

            // Stage DEDUP+CREATE: embed this pair's subjects in one call.
            let texts: Vec<String> = extraction
                .subjects
                .iter()
                .map(|s| format!("{}\n{}", s.name, s.description).trim().to_string())
                .collect();
            let embedded = self.embedder.embed(EmbedRequest { texts }).await?;
            costs.add(embedded.cost.as_ref());
            if embedded.embeddings.len() != extraction.subjects.len() {
                return Err(Error::Embedding(format!(
                    "dream: embedder returned {} embeddings for {} subjects",
                    embedded.embeddings.len(),
                    extraction.subjects.len()
                )));
            }

            for (subject, embedding) in extraction.subjects.iter().zip(embedded.embeddings) {
                let subject_id = self
                    .dedup_or_create_subject(
                        db,
                        user_id,
                        &kg_id,
                        subject,
                        embedding,
                        &mut entries,
                        &mut report,
                    )
                    .await?;

                // Stage LINK.
                let linked = link_subject_pair(db, &subject_id, &pair.id, user_id, &kg_id).await?;
                if linked {
                    report.links_created += 1;
                    mark_key_subject(db, &subject_id).await?;
                }
            }
        }

        // All model calls failed and there was work to do: surface the error.
        if !pairs.is_empty() && !any_model_ok {
            if let Some(err) = first_model_error {
                return Err(err);
            }
        }

        // Stage REFINE.
        if refine {
            report.subjects_refined = self
                .refine_subjects(db, user_id, &kg_id, concurrency, &mut costs)
                .await?;
        }

        report.usage = costs.into_items();
        Ok(report)
    }

    /// Runs one prompt per item through the model with bounded concurrency,
    /// preserving input order.
    async fn run_model_stage(
        &self,
        prompts: Vec<String>,
        concurrency: usize,
    ) -> Vec<Result<ChatChunk>> {
        stream::iter(prompts)
            .map(|prompt| {
                let model = Arc::clone(&self.model);
                async move {
                    let messages = [ChatMessage {
                        role: "user".to_string(),
                        content: vec![Content::text(prompt)],
                        ..ChatMessage::default()
                    }];
                    model.next(&messages, &[]).await
                }
            })
            .buffered(concurrency.max(1))
            .collect()
            .await
    }

    /// Merges `subject` into the closest existing subject (cosine >=
    /// threshold) or inserts a new row, updating `entries` and `report`.
    /// Returns the subject row id to link.
    #[allow(clippy::too_many_arguments)]
    async fn dedup_or_create_subject(
        &self,
        db: &Db,
        user_id: &str,
        kg_id: &str,
        subject: &ExtractedSubject,
        embedding: Vec<f32>,
        entries: &mut Vec<SubjectEntry>,
        report: &mut DreamReport,
    ) -> Result<String> {
        // Top candidate by cosine similarity over the in-memory graph view.
        let best = entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| {
                db::cosine_similarity(&embedding, &entry.embedding).map(|sim| (idx, f64::from(sim)))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));

        if let Some((idx, sim)) = best {
            if sim >= SUBJECT_MERGE_THRESHOLD {
                let entry = &mut entries[idx];
                let desc = subject.description.trim();
                if !desc.is_empty() && !entry.description.contains(desc) {
                    entry.description = if entry.description.is_empty() {
                        desc.to_string()
                    } else {
                        format!("{}{MERGE_SEPARATOR}{}", entry.description, desc)
                    };
                    db.connection()
                        .execute(
                            "UPDATE subjects SET description_text = ?, updated_at = ? WHERE id = ?",
                            (
                                turso::Value::Text(entry.description.clone()),
                                turso::Value::Text(now_text()),
                                turso::Value::Text(entry.id.clone()),
                            ),
                        )
                        .await?;
                }
                report.subjects_merged += 1;
                return Ok(entry.id.clone());
            }
        }

        // Insert; UNIQUE(user_id, kg_id, subject_text) backstops exact dups.
        let new_id = db::new_row_id();
        let mut rows = db
            .connection()
            .query(
                "INSERT INTO subjects (id, user_id, kg_id, subject_text, description_text, is_key_subject, embedding, updated_at)
                 VALUES (?, ?, ?, ?, NULLIF(?, ''), 0, ?, ?)
                 ON CONFLICT (user_id, kg_id, subject_text) DO UPDATE SET
                    description_text = COALESCE(NULLIF(subjects.description_text, ''), excluded.description_text),
                    embedding = COALESCE(subjects.embedding, excluded.embedding),
                    updated_at = excluded.updated_at
                 RETURNING id",
                vec![
                    turso::Value::Text(new_id.clone()),
                    turso::Value::Text(user_id.to_string()),
                    turso::Value::Text(kg_id.to_string()),
                    turso::Value::Text(subject.name.clone()),
                    turso::Value::Text(subject.description.clone()),
                    turso::Value::Blob(db::encode_f32_blob(&embedding)),
                    turso::Value::Text(now_text()),
                ],
            )
            .await?;
        let row = rows
            .next()
            .await?
            .ok_or_else(|| Error::Other("dream: subject upsert returned no row".into()))?;
        let stored_id = value_text(row.get_value(0)?);

        if stored_id == new_id {
            report.subjects_created += 1;
            entries.push(SubjectEntry {
                id: stored_id.clone(),
                description: subject.description.clone(),
                embedding,
            });
        } else {
            // Same name already existed (e.g. its stored embedding drifted
            // below the merge threshold) — count as a merge.
            report.subjects_merged += 1;
        }
        Ok(stored_id)
    }

    /// Stage REFINE: consolidates subjects whose descriptions accumulated
    /// `" | "` merges. Returns the number of subjects refined.
    async fn refine_subjects(
        &self,
        db: &Db,
        user_id: &str,
        kg_id: &str,
        concurrency: usize,
        costs: &mut CostCollector,
    ) -> Result<usize> {
        let mut candidates: Vec<(String, String, String)> = Vec::new();
        let mut rows = db
            .connection()
            .query(
                "SELECT id, subject_text, description_text FROM subjects
                 WHERE user_id = ? AND kg_id = ? AND description_text LIKE '%' || ? || '%'",
                (
                    turso::Value::Text(user_id.to_string()),
                    turso::Value::Text(kg_id.to_string()),
                    turso::Value::Text(MERGE_SEPARATOR.to_string()),
                ),
            )
            .await?;
        while let Some(row) = rows.next().await? {
            candidates.push((
                value_text(row.get_value(0)?),
                value_text(row.get_value(1)?),
                value_text(row.get_value(2)?),
            ));
        }
        if candidates.is_empty() {
            return Ok(0);
        }

        let outcomes = self
            .run_model_stage(
                candidates
                    .iter()
                    .map(|(_, name, desc)| refine_prompt(name, desc))
                    .collect(),
                concurrency,
            )
            .await;

        let mut refined = 0usize;
        for ((id, old_name, _), outcome) in candidates.iter().zip(outcomes) {
            let chunk = match outcome {
                Ok(chunk) => chunk,
                Err(err) => {
                    tracing::warn!(subject_id = %id, error = %err, "dream: refine call failed");
                    continue;
                }
            };
            costs.add(chunk.cost.as_ref());
            let Some((mut name, summary)) = parse_refinement(&chunk.text) else {
                tracing::warn!(subject_id = %id, "dream: unparseable refine output, skipping");
                continue;
            };

            // Renaming must not collide with UNIQUE(user_id, kg_id,
            // subject_text); on collision keep the old name.
            if name != *old_name {
                let mut taken = db
                    .connection()
                    .query(
                        "SELECT id FROM subjects WHERE user_id = ? AND kg_id = ? AND subject_text = ? AND id != ?",
                        (
                            turso::Value::Text(user_id.to_string()),
                            turso::Value::Text(kg_id.to_string()),
                            turso::Value::Text(name.clone()),
                            turso::Value::Text(id.clone()),
                        ),
                    )
                    .await?;
                if taken.next().await?.is_some() {
                    name = old_name.clone();
                }
            }

            let embedded = self
                .embedder
                .embed(EmbedRequest {
                    texts: vec![format!("{name}\n{summary}")],
                })
                .await?;
            costs.add(embedded.cost.as_ref());
            let Some(embedding) = embedded.embeddings.first() else {
                continue;
            };

            db.connection()
                .execute(
                    "UPDATE subjects SET subject_text = ?, description_text = ?, embedding = ?, updated_at = ? WHERE id = ?",
                    vec![
                        turso::Value::Text(name),
                        turso::Value::Text(summary),
                        turso::Value::Blob(db::encode_f32_blob(embedding)),
                        turso::Value::Text(now_text()),
                        turso::Value::Text(id.clone()),
                    ],
                )
                .await?;
            refined += 1;
        }
        Ok(refined)
    }
}

/// Loads up to `limit` newest memory pairs with renderable conversation text.
async fn load_recent_pairs(
    db: &Db,
    user_id: &str,
    kg_id: &str,
    limit: usize,
) -> Result<Vec<PairLite>> {
    let mut rows = db
        .connection()
        .query(
            "SELECT id, prompt, response, input, output, description FROM memory_pairs
             WHERE user_id = ? AND kg_id = ?
             ORDER BY timestamp DESC LIMIT ?",
            (
                turso::Value::Text(user_id.to_string()),
                turso::Value::Text(kg_id.to_string()),
                turso::Value::Integer(limit as i64),
            ),
        )
        .await?;
    let mut pairs = Vec::new();
    while let Some(row) = rows.next().await? {
        let prompt = value_text(row.get_value(1)?);
        let response = value_text(row.get_value(2)?);
        let input_json = value_text(row.get_value(3)?);
        let output_json = value_text(row.get_value(4)?);
        let user_text = if prompt.is_empty() {
            content_text(&input_json)
        } else {
            prompt
        };
        let assistant_text = if response.is_empty() {
            content_text(&output_json)
        } else {
            response
        };
        let conversation = conversation_text(&user_text, &assistant_text);
        if conversation.is_empty() {
            continue;
        }
        pairs.push(PairLite {
            id: value_text(row.get_value(0)?),
            conversation,
            has_description: !value_text(row.get_value(5)?).is_empty(),
        });
    }
    Ok(pairs)
}

/// Loads the user's existing subjects for cosine dedup.
async fn load_subject_entries(db: &Db, user_id: &str, kg_id: &str) -> Result<Vec<SubjectEntry>> {
    let mut rows = db
        .connection()
        .query(
            "SELECT id, subject_text, description_text, embedding FROM subjects
             WHERE user_id = ? AND kg_id = ?",
            (
                turso::Value::Text(user_id.to_string()),
                turso::Value::Text(kg_id.to_string()),
            ),
        )
        .await?;
    let mut entries = Vec::new();
    while let Some(row) = rows.next().await? {
        let embedding = match row.get_value(3)? {
            turso::Value::Blob(blob) => db::decode_f32_blob(&blob),
            _ => Vec::new(),
        };
        entries.push(SubjectEntry {
            id: value_text(row.get_value(0)?),
            description: value_text(row.get_value(2)?),
            embedding,
        });
    }
    Ok(entries)
}

/// Writes the extraction summary onto a pair that has no description yet.
async fn write_pair_summary(db: &Db, pair_id: &str, summary: &str) -> Result<()> {
    db.connection()
        .execute(
            "UPDATE memory_pairs SET description = ?, updated_at = ?
             WHERE id = ? AND (description IS NULL OR description = '')",
            (
                turso::Value::Text(summary.to_string()),
                turso::Value::Text(now_text()),
                turso::Value::Text(pair_id.to_string()),
            ),
        )
        .await?;
    Ok(())
}

/// Inserts a subject↔pair link, ignoring conflicts. Returns whether a new
/// row was inserted.
async fn link_subject_pair(
    db: &Db,
    subject_id: &str,
    pair_id: &str,
    user_id: &str,
    kg_id: &str,
) -> Result<bool> {
    let affected = db
        .connection()
        .execute(
            "INSERT INTO subject_memory_pair_links (subject_id, pair_id, user_id, kg_id)
             VALUES (?, ?, ?, ?)
             ON CONFLICT (subject_id, pair_id) DO NOTHING",
            (
                turso::Value::Text(subject_id.to_string()),
                turso::Value::Text(pair_id.to_string()),
                turso::Value::Text(user_id.to_string()),
                turso::Value::Text(kg_id.to_string()),
            ),
        )
        .await?;
    Ok(affected > 0)
}

/// Flips `is_key_subject` on once a subject has a linked pair.
async fn mark_key_subject(db: &Db, subject_id: &str) -> Result<()> {
    db.connection()
        .execute(
            "UPDATE subjects SET is_key_subject = 1 WHERE id = ? AND is_key_subject = 0",
            (turso::Value::Text(subject_id.to_string()),),
        )
        .await?;
    Ok(())
}

fn now_text() -> String {
    db::format_timestamp(chrono::Utc::now())
}

fn value_text(v: turso::Value) -> String {
    match v {
        turso::Value::Text(s) => s,
        _ => String::new(),
    }
}

/// Joins the text parts of a stored `input`/`output` JSON array.
fn content_text(json_text: &str) -> String {
    if json_text.is_empty() {
        return String::new();
    }
    let Ok(parts) = serde_json::from_str::<Vec<Content>>(json_text) else {
        return String::new();
    };
    parts
        .iter()
        .filter(|p| !p.content.is_empty())
        .map(|p| p.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Renders the conversation shown to the extraction model, byte-capped on a
/// char boundary.
fn conversation_text(user_text: &str, assistant_text: &str) -> String {
    let text = match (user_text.is_empty(), assistant_text.is_empty()) {
        (true, true) => String::new(),
        (false, true) => format!("User: {user_text}"),
        (true, false) => format!("Assistant: {assistant_text}"),
        (false, false) => format!("User: {user_text}\n\nAssistant: {assistant_text}"),
    };
    truncate_chars(&text, MAX_CONVERSATION_BYTES)
}

/// Truncates to at most `max_bytes` bytes on a valid char boundary.
fn truncate_chars(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// Prompt for the EXTRACT stage. Written so a small local model reliably
/// emits parseable JSON.
fn extraction_prompt(conversation: &str) -> String {
    format!(
        "You are a memory analyst. Read the conversation below and extract durable subjects worth remembering long-term.\n\
         \n\
         Durable subjects are people, projects, preferences, and recurring topics that will matter in future conversations. Ignore one-off trivia, pleasantries, and anything ephemeral.\n\
         \n\
         Respond with STRICT JSON only — no prose, no code fences, exactly this shape:\n\
         {{\"summary\": \"<one sentence summary of the conversation>\", \"subjects\": [{{\"name\": \"<short subject name>\", \"description\": \"<one sentence describing the subject>\", \"type\": \"<person|project|preference|topic>\"}}]}}\n\
         \n\
         Rules:\n\
         - 1 to 5 subjects when the conversation contains something durable; an EMPTY subjects array is a valid answer — not every exchange merits memory.\n\
         - Keep names short (1-4 words) and descriptions to one sentence.\n\
         \n\
         Conversation:\n\
         {conversation}"
    )
}

/// Prompt for the REFINE stage.
fn refine_prompt(name: &str, merged_description: &str) -> String {
    format!(
        "You maintain a knowledge graph. The subject below has accumulated several merged descriptions separated by \" | \". Consolidate them.\n\
         \n\
         Respond with STRICT JSON only — no prose, no code fences, exactly this shape:\n\
         {{\"name\": \"<concise canonical name, 2-5 words>\", \"summary\": \"<rolling narrative summary, 1-3 sentences>\"}}\n\
         \n\
         Subject name: {name}\n\
         Merged descriptions: {merged_description}"
    )
}

/// Returns the substring spanning the first `{{` to the last `}}`, which also
/// strips surrounding prose and markdown code fences.
fn first_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    Some(&text[start..=end])
}

/// Defensively parses EXTRACT output. Returns `None` only when no JSON
/// object can be recovered at all; malformed subject entries are skipped.
fn parse_extraction(raw: &str) -> Option<Extraction> {
    let json = first_json_object(raw)?;
    let value: Value = serde_json::from_str(json).ok()?;
    let obj = value.as_object()?;
    let summary = obj
        .get("summary")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut subjects = Vec::new();
    if let Some(arr) = obj.get("subjects").and_then(Value::as_array) {
        for entry in arr {
            let Some(entry) = entry.as_object() else {
                continue;
            };
            let name = entry
                .get("name")
                .or_else(|| entry.get("text"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            if name.is_empty() {
                continue;
            }
            let description = entry
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            subjects.push(ExtractedSubject {
                name: name.to_string(),
                description: description.to_string(),
            });
            if subjects.len() == MAX_SUBJECTS_PER_MEMORY {
                break;
            }
        }
    }
    Some(Extraction { summary, subjects })
}

/// Defensively parses REFINE output into (name, summary).
fn parse_refinement(raw: &str) -> Option<(String, String)> {
    let json = first_json_object(raw)?;
    let value: Value = serde_json::from_str(json).ok()?;
    let obj = value.as_object()?;
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let summary = obj
        .get("summary")
        .or_else(|| obj.get("description"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if name.is_empty() || summary.is_empty() {
        return None;
    }
    Some((name, summary))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Store, StoreOptions};
    use crate::types::{ChatChunk, Cost, CostedUsage, EmbedResponse, ToolDefinition, Usage};
    use async_trait::async_trait;

    /// Model whose reply is the first scripted response whose marker appears
    /// in the prompt; unmatched prompts get `fallback`.
    struct ScriptedModel {
        script: Vec<(&'static str, &'static str)>,
        fallback: &'static str,
    }

    #[async_trait]
    impl Model for ScriptedModel {
        async fn next(
            &self,
            messages: &[ChatMessage],
            _tools: &[ToolDefinition],
        ) -> crate::types::Result<ChatChunk> {
            let prompt = messages
                .last()
                .and_then(|m| m.content.first())
                .map(|c| c.content.clone())
                .unwrap_or_default();
            let text = self
                .script
                .iter()
                .find(|(marker, _)| prompt.contains(marker))
                .map(|(_, response)| (*response).to_string())
                .unwrap_or_else(|| self.fallback.to_string());
            Ok(ChatChunk {
                text,
                cost: Some(CostedUsage {
                    usage: Usage {
                        provider: "fake".into(),
                        model: "scripted".into(),
                        input_tokens: 10,
                        output_tokens: 5,
                        total_tokens: 15,
                    },
                    cost: Cost::default(),
                }),
                ..ChatChunk::default()
            })
        }
    }

    /// Deterministic embedder: a one-hot 768-dim vector keyed by the first
    /// word, so texts sharing a first word are identical (cosine 1.0) and
    /// others are orthogonal (cosine 0.0).
    struct FirstWordEmbedder;

    #[async_trait]
    impl Embedder for FirstWordEmbedder {
        async fn embed(&self, req: EmbedRequest) -> crate::types::Result<EmbedResponse> {
            let embeddings = req
                .texts
                .iter()
                .map(|text| {
                    let word = text
                        .split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .to_lowercase();
                    let mut hash = 0usize;
                    for b in word.bytes() {
                        hash = hash.wrapping_mul(31).wrapping_add(b as usize);
                    }
                    let mut v = vec![0.0f32; crate::db::EMBEDDING_DIMS];
                    v[hash % crate::db::EMBEDDING_DIMS] = 1.0;
                    v
                })
                .collect();
            Ok(EmbedResponse {
                embeddings,
                ..EmbedResponse::default()
            })
        }
    }

    async fn seed_pair(db: &Db, user_id: &str, kg: &str, n: i64, prompt: &str, response: &str) {
        db.connection()
            .execute(
                "INSERT OR IGNORE INTO harness_users (uid) VALUES (?)",
                (turso::Value::Text(user_id.to_string()),),
            )
            .await
            .expect("seed user");
        db.connection()
            .execute(
                "INSERT INTO memory_pairs (id, firestore_pair_id, user_id, kg_id, prompt, response, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?)",
                vec![
                    turso::Value::Text(format!("row-{n}")),
                    turso::Value::Text(format!("pair-{n}")),
                    turso::Value::Text(user_id.to_string()),
                    turso::Value::Text(kg.to_string()),
                    turso::Value::Text(prompt.to_string()),
                    turso::Value::Text(response.to_string()),
                    turso::Value::Text(db::format_timestamp(
                        chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                            .expect("ts")
                            .with_timezone(&chrono::Utc)
                            + chrono::Duration::seconds(n),
                    )),
                ],
            )
            .await
            .expect("seed pair");
    }

    async fn dreamer_with(
        script: Vec<(&'static str, &'static str)>,
        fallback: &'static str,
    ) -> (Dreamer, Arc<Db>) {
        let db = Arc::new(Db::open_memory().await.expect("open db"));
        let embedder: Arc<dyn Embedder> = Arc::new(FirstWordEmbedder);
        let store = Arc::new(Store::new(StoreOptions {
            db: Arc::clone(&db),
            embedder: Arc::clone(&embedder),
            predictor: None,
        }));
        let model: Arc<dyn Model> = Arc::new(ScriptedModel { script, fallback });
        (Dreamer::new(store, model, embedder), db)
    }

    async fn count(db: &Db, sql: &str) -> i64 {
        let mut rows = db.connection().query(sql, ()).await.expect("count query");
        match rows
            .next()
            .await
            .expect("row")
            .expect("one row")
            .get_value(0)
            .expect("value")
        {
            turso::Value::Integer(n) => n,
            other => panic!("expected integer, got {other:?}"),
        }
    }

    #[test]
    fn parse_extraction_is_defensive() {
        // Code fences + prose around the object.
        let fenced = "Sure! Here you go:\n```json\n{\"summary\": \"s\", \"subjects\": [{\"name\": \"Hiking\", \"description\": \"d\", \"type\": \"topic\"}]}\n```";
        let parsed = parse_extraction(fenced).expect("parse fenced");
        assert_eq!(parsed.summary, "s");
        assert_eq!(parsed.subjects.len(), 1);
        assert_eq!(parsed.subjects[0].name, "Hiking");

        // Garbage entries skipped, names via "text" alias accepted.
        let messy = r#"{"summary": 7, "subjects": [42, {"description": "no name"}, {"text": "Rust", "description": "lang"}, {"name": "  "}]}"#;
        let parsed = parse_extraction(messy).expect("parse messy");
        assert_eq!(parsed.summary, "");
        assert_eq!(parsed.subjects.len(), 1);
        assert_eq!(parsed.subjects[0].name, "Rust");

        // Subject cap.
        let many = format!(
            r#"{{"summary": "s", "subjects": [{}]}}"#,
            (0..9)
                .map(|i| format!(r#"{{"name": "S{i}", "description": "d"}}"#))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert_eq!(
            parse_extraction(&many).expect("parse many").subjects.len(),
            MAX_SUBJECTS_PER_MEMORY
        );

        // Unrecoverable outputs.
        assert!(parse_extraction("no json here at all").is_none());
        assert!(parse_extraction("{not valid json}").is_none());
        assert!(parse_refinement("{\"name\": \"x\"}").is_none());
        assert!(parse_refinement("```json\n{\"name\": \"N\", \"summary\": \"S\"}\n```").is_some());
    }

    #[tokio::test]
    async fn dream_extracts_links_and_is_idempotent() {
        let user = "u1";
        let kg = derive_kg_id(user);
        let (dreamer, db) = dreamer_with(
            vec![
                (
                    "hiking",
                    r#"{"summary": "User loves hiking.", "subjects": [{"name": "Hiking", "description": "User enjoys hiking in Colorado", "type": "preference"}]}"#,
                ),
                (
                    "sourdough",
                    r#"{"summary": "Talked sourdough.", "subjects": [{"name": "Sourdough baking", "description": "User is learning sourdough", "type": "topic"}]}"#,
                ),
            ],
            r#"{"summary": "n/a", "subjects": []}"#,
        )
        .await;
        seed_pair(&db, user, &kg, 1, "I love hiking in Colorado", "Noted!").await;
        seed_pair(
            &db,
            user,
            &kg,
            2,
            "Teach me sourdough baking",
            "Start with a starter.",
        )
        .await;

        let report = dreamer
            .dream(
                user,
                DreamOptions {
                    refine: Some(false),
                    ..DreamOptions::default()
                },
            )
            .await
            .expect("dream");
        assert_eq!(report.memories_processed, 2);
        assert_eq!(report.subjects_created, 2);
        assert_eq!(report.subjects_merged, 0);
        assert_eq!(report.links_created, 2);
        assert_eq!(report.subjects_refined, 0);
        assert!(!report.usage.is_empty());

        assert_eq!(count(&db, "SELECT COUNT(*) FROM subjects").await, 2);
        assert_eq!(
            count(
                &db,
                "SELECT COUNT(*) FROM subjects WHERE is_key_subject = 1"
            )
            .await,
            2
        );
        assert_eq!(
            count(&db, "SELECT COUNT(*) FROM subject_memory_pair_links").await,
            2
        );
        // Extraction summaries land on pairs without descriptions.
        assert_eq!(
            count(
                &db,
                "SELECT COUNT(*) FROM memory_pairs WHERE description IS NOT NULL AND description != ''"
            )
            .await,
            2
        );

        // Re-running merges into existing subjects and re-links nothing.
        let rerun = dreamer
            .dream(
                user,
                DreamOptions {
                    refine: Some(false),
                    ..DreamOptions::default()
                },
            )
            .await
            .expect("re-dream");
        assert_eq!(rerun.subjects_created, 0);
        assert_eq!(rerun.subjects_merged, 2);
        assert_eq!(rerun.links_created, 0);
        assert_eq!(count(&db, "SELECT COUNT(*) FROM subjects").await, 2);
        assert_eq!(
            count(&db, "SELECT COUNT(*) FROM subject_memory_pair_links").await,
            2
        );
    }

    #[tokio::test]
    async fn dream_survives_malformed_model_output() {
        let user = "u1";
        let kg = derive_kg_id(user);
        let (dreamer, db) = dreamer_with(
            vec![
                ("hiking", "I'm sorry, I can't produce JSON today. Tools: {broken"),
                (
                    "sourdough",
                    "```json\n{\"summary\": \"Bread talk.\", \"subjects\": [{\"name\": \"Sourdough baking\", \"description\": \"User bakes bread\", \"type\": \"topic\"}]}\n```",
                ),
            ],
            "{}",
        )
        .await;
        seed_pair(&db, user, &kg, 1, "I love hiking in Colorado", "Noted!").await;
        seed_pair(&db, user, &kg, 2, "Teach me sourdough baking", "Sure.").await;

        let report = dreamer
            .dream(
                user,
                DreamOptions {
                    refine: Some(false),
                    ..DreamOptions::default()
                },
            )
            .await
            .expect("dream tolerates malformed output");
        assert_eq!(report.memories_processed, 2);
        assert_eq!(report.subjects_created, 1);
        assert_eq!(report.links_created, 1);
    }

    #[tokio::test]
    async fn dream_empty_subjects_is_valid() {
        let user = "u1";
        let kg = derive_kg_id(user);
        let (dreamer, db) =
            dreamer_with(vec![], r#"{"summary": "Small talk.", "subjects": []}"#).await;
        seed_pair(&db, user, &kg, 1, "hey", "hello!").await;

        let report = dreamer
            .dream(user, DreamOptions::default())
            .await
            .expect("dream");
        assert_eq!(report.memories_processed, 1);
        assert_eq!(report.subjects_created, 0);
        assert_eq!(report.links_created, 0);
        assert_eq!(count(&db, "SELECT COUNT(*) FROM subjects").await, 0);
    }

    #[tokio::test]
    async fn dream_merges_similar_subjects_and_refines() {
        let user = "u1";
        let kg = derive_kg_id(user);
        // Both subject names start with "Rust" -> identical fake embeddings
        // -> the second merges into the first with " | " accumulation; the
        // refine stage then consolidates name + description.
        let (dreamer, db) = dreamer_with(
            vec![
                // Refine marker first: merged descriptions echo the memory
                // text, so memory-text markers would shadow it otherwise.
                (
                    "accumulated several merged",
                    r#"{"name": "Rust Programming", "summary": "The user is learning Rust, from ownership to async traits."}"#,
                ),
                (
                    "borrow checker",
                    r#"{"summary": "Rust borrow checker.", "subjects": [{"name": "Rust ownership", "description": "User studies the borrow checker", "type": "topic"}]}"#,
                ),
                (
                    "async traits",
                    r#"{"summary": "Rust async.", "subjects": [{"name": "Rust async", "description": "User writes async trait code", "type": "topic"}]}"#,
                ),
            ],
            "{}",
        )
        .await;
        seed_pair(
            &db,
            user,
            &kg,
            1,
            "Explain the borrow checker",
            "Ownership rules...",
        )
        .await;
        seed_pair(
            &db,
            user,
            &kg,
            2,
            "Help with async traits",
            "Use async-trait...",
        )
        .await;

        let report = dreamer
            .dream(user, DreamOptions::default())
            .await
            .expect("dream");
        assert_eq!(report.subjects_created, 1);
        assert_eq!(report.subjects_merged, 1);
        assert_eq!(
            report.links_created, 2,
            "both pairs link to the merged subject"
        );
        assert_eq!(report.subjects_refined, 1);

        assert_eq!(count(&db, "SELECT COUNT(*) FROM subjects").await, 1);
        let mut rows = db
            .connection()
            .query("SELECT subject_text, description_text FROM subjects", ())
            .await
            .expect("select subject");
        let row = rows.next().await.expect("next").expect("row");
        assert_eq!(
            value_text(row.get_value(0).expect("name")),
            "Rust Programming"
        );
        let description = value_text(row.get_value(1).expect("description"));
        assert!(
            !description.contains(MERGE_SEPARATOR),
            "refine must consolidate merged descriptions, got {description:?}"
        );
        assert_eq!(
            count(&db, "SELECT COUNT(*) FROM subject_memory_pair_links").await,
            2
        );
    }

    #[tokio::test]
    async fn dream_requires_user_id() {
        let (dreamer, _db) = dreamer_with(vec![], "{}").await;
        assert!(matches!(
            dreamer.dream("", DreamOptions::default()).await,
            Err(Error::InvalidArgument(_))
        ));
    }

    /// Real-ollama end-to-end dream: requires `ollama serve` with a chat
    /// model (`DITTO_HARNESS_OLLAMA_MODEL`, default `gemma3:4b`) +
    /// `embeddinggemma` pulled. Gated behind `DITTO_HARNESS_OLLAMA=1` so
    /// plain `cargo test` passes offline.
    #[tokio::test]
    async fn dream_ollama_integration() {
        if std::env::var("DITTO_HARNESS_OLLAMA").as_deref() != Ok("1") {
            return;
        }
        use crate::models::{ChatModelConfig, OllamaEmbedder, DEFAULT_OLLAMA_CHAT_MODEL};
        let chat_model = std::env::var("DITTO_HARNESS_OLLAMA_MODEL")
            .unwrap_or_else(|_| DEFAULT_OLLAMA_CHAT_MODEL.to_string());

        let user = "u-ollama";
        let kg = derive_kg_id(user);
        let db = Arc::new(Db::open_memory().await.expect("open db"));
        let embedder: Arc<dyn Embedder> = Arc::new(OllamaEmbedder::default());
        let store = Arc::new(Store::new(StoreOptions {
            db: Arc::clone(&db),
            embedder: Arc::clone(&embedder),
            predictor: None,
        }));
        let model = ChatModelConfig::ollama("", &chat_model)
            .build()
            .expect("build ollama model");
        let dreamer = Dreamer::new(store, model, embedder);

        seed_pair(
            &db,
            user,
            &kg,
            1,
            "My friend Alice and I are building a birdhouse project this summer",
            "That sounds fun! Cedar is a great wood choice for birdhouses.",
        )
        .await;
        seed_pair(
            &db,
            user,
            &kg,
            2,
            "Alice thinks we should paint the birdhouse blue",
            "Blue is a lovely choice; use outdoor-safe paint.",
        )
        .await;

        let report = dreamer
            .dream(user, DreamOptions::default())
            .await
            .expect("ollama dream");
        assert_eq!(report.memories_processed, 2);
        assert!(
            report.subjects_created + report.subjects_merged > 0,
            "expected at least one subject from ollama, report: {report:?}"
        );
        assert!(report.links_created > 0);
        assert!(count(&db, "SELECT COUNT(*) FROM subjects").await > 0);
    }
}
