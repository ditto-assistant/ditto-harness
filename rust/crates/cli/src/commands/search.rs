//! `search` subcommand: composite-search memories and print ranked slim
//! previews with their retrieval scores.

use ditto_harness::memory::{to_slim_memory_preview, CompositeSearchRequest};

use crate::commands::util;
use crate::Common;

pub async fn run(common: &Common, query: &str, top_k: usize) -> anyhow::Result<()> {
    anyhow::ensure!(!query.trim().is_empty(), "--query must not be empty");
    let store = util::open_store(common).await?;
    let (memories, metadata) = store
        .search_composite_memories(CompositeSearchRequest {
            user_id: common.user.clone(),
            query: query.to_string(),
            limit: top_k,
            ..CompositeSearchRequest::default()
        })
        .await
        .map_err(|err| util::ollama_hint(err, "search memories"))?;

    if let Some(meta) = &metadata {
        eprintln!("intent: {}  variant: {}", meta.intent, meta.variant);
    }
    if memories.is_empty() {
        println!("no memories matched {query:?}");
        return Ok(());
    }
    for (rank, mem) in memories.iter().enumerate() {
        let slim = to_slim_memory_preview(mem, 0);
        println!(
            "{:>2}. {}  composite={:.4} cosine={:.4} recency={:.4} frequency={:.4}",
            rank + 1,
            mem.id,
            mem.composite_score,
            mem.similarity,
            mem.recency_score,
            mem.frequency_score,
        );
        if !slim.timestamp.is_empty() || !slim.source.is_empty() {
            println!("    {} {}", slim.timestamp, slim.source);
        }
        if !slim.preview.is_empty() {
            println!("    {}", util::snippet(&slim.preview, 200));
        }
    }
    Ok(())
}
