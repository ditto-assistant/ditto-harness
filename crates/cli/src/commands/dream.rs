// SPDX-License-Identifier: MIT
//! `dream` subcommand: run the subject extraction/consolidation pipeline for
//! the user, print the `DreamReport`, then list the user's subject graph
//! (names, key flags, link counts) so the new subjects are visible.

use std::sync::Arc;

use ditto_harness::db::Db;
use ditto_harness::dream::{DreamOptions, Dreamer};
use ditto_harness::types::Embedder;

use crate::commands::util;
use crate::Common;

pub async fn run(common: &Common) -> anyhow::Result<()> {
    let store = util::open_store(common).await?;
    let model = util::build_chat_model(common)?;
    let embedder: Arc<dyn Embedder> = Arc::new(util::build_embedder(common));
    let dreamer = Dreamer::new(Arc::clone(&store), model, embedder);

    eprintln!("dreaming over memories for user {:?} ...", common.user);
    let report = dreamer
        .dream(&common.user, DreamOptions::default())
        .await
        .map_err(|err| util::ollama_hint(err, "dream"))?;

    println!("subjects created: {}", report.subjects_created);
    println!("links created: {}", report.links_created);
    util::print_costs(&report.usage);

    let subjects = list_subjects(store.db(), &common.user).await?;
    if subjects.is_empty() {
        println!("no subjects stored yet — run `ditto-harness seed` first");
        return Ok(());
    }
    println!("\nsubject graph for user {:?}:", common.user);
    for line in subjects {
        let key = if line.key { " [key]" } else { "" };
        println!(
            "  {}{}  ({} linked {})",
            line.text,
            key,
            line.links,
            if line.links == 1 {
                "memory"
            } else {
                "memories"
            }
        );
        if !line.description.is_empty() {
            println!("    {}", util::snippet(&line.description, 120));
        }
    }
    Ok(())
}

struct SubjectLine {
    text: String,
    description: String,
    key: bool,
    links: i64,
}

/// Lists every subject for the user with its link count (most-linked first).
async fn list_subjects(db: &Db, user_id: &str) -> anyhow::Result<Vec<SubjectLine>> {
    let mut rows = db
        .connection()?
        .query(
            "SELECT s.subject_text, COALESCE(s.description_text, ''), s.is_key_subject,
                    COUNT(l.pair_id) AS links
             FROM subjects s
             LEFT JOIN subject_memory_pair_links l ON l.subject_id = s.id
             WHERE s.user_id = ?
             GROUP BY s.id
             ORDER BY links DESC, s.subject_text ASC",
            (turso::Value::Text(user_id.to_string()),),
        )
        .await?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        out.push(SubjectLine {
            text: text_column(&row, 0)?,
            description: text_column(&row, 1)?,
            key: int_column(&row, 2)? != 0,
            links: int_column(&row, 3)?,
        });
    }
    Ok(out)
}

fn text_column(row: &turso::Row, idx: usize) -> anyhow::Result<String> {
    match row.get_value(idx)? {
        turso::Value::Text(s) => Ok(s),
        turso::Value::Null => Ok(String::new()),
        other => anyhow::bail!("column {idx}: expected text, got {other:?}"),
    }
}

fn int_column(row: &turso::Row, idx: usize) -> anyhow::Result<i64> {
    match row.get_value(idx)? {
        turso::Value::Integer(v) => Ok(v),
        other => anyhow::bail!("column {idx}: expected integer, got {other:?}"),
    }
}
