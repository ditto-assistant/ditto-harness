// SPDX-License-Identifier: MIT
//! `subjects` subcommand: search the subject graph and print subjects with
//! descriptions, key flags, and memory counts.

use ditto_harness::memory::SearchSubjectsRequest;

use crate::commands::util;
use crate::Common;

pub async fn run(common: &Common, query: &str, top_k: usize) -> anyhow::Result<()> {
    anyhow::ensure!(!query.trim().is_empty(), "--query must not be empty");
    let store = util::open_store(common).await?;
    let subjects = store
        .search_subjects(SearchSubjectsRequest {
            user_id: common.user.clone(),
            queries: vec![query.to_string()],
            limit: top_k,
            ..SearchSubjectsRequest::default()
        })
        .await
        .map_err(|err| util::ollama_hint(err, "search subjects"))?;

    if subjects.is_empty() {
        println!("no subjects matched {query:?}");
        return Ok(());
    }
    for (rank, subj) in subjects.iter().enumerate() {
        let key = if subj.key { " [key]" } else { "" };
        println!(
            "{:>2}. {}{}  similarity={:.4}  memories={}",
            rank + 1,
            subj.text,
            key,
            subj.similarity,
            subj.memory_count,
        );
        if !subj.description.is_empty() {
            println!("    {}", util::snippet(&subj.description, 160));
        }
    }
    Ok(())
}
