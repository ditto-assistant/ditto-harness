// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared helpers for the subcommand handlers: opening the store, building
//! chat models and embedders from the global flags, and printing summaries.

use std::sync::Arc;

use anyhow::Context;

use ditto_harness::db::Db;
use ditto_harness::memory::{Store, StoreOptions};
use ditto_harness::models::{
    ChatModelConfig, OllamaEmbedder, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_CHAT_MODEL,
};
use ditto_harness::types::{CostedUsage, Model};

use crate::{Common, Provider};

/// Base URL for the Ollama embedder. `--base-url` only applies when the chat
/// provider is also Ollama; embeddings otherwise use the local default.
pub(crate) fn ollama_base_url(common: &Common) -> String {
    match (common.provider, common.base_url.as_deref()) {
        (Provider::Ollama, Some(url)) if !url.is_empty() => url.to_string(),
        _ => DEFAULT_OLLAMA_BASE_URL.to_string(),
    }
}

/// The Ollama embedder used by every subcommand (embeddinggemma, 768 dims).
pub(crate) fn build_embedder(common: &Common) -> OllamaEmbedder {
    OllamaEmbedder::new(ollama_base_url(common))
}

/// Opens (creating if needed) the database and wraps it in a memory store
/// with the Ollama embedder.
pub(crate) async fn open_store(common: &Common) -> anyhow::Result<Arc<Store>> {
    let db = Db::open(&common.db)
        .await
        .with_context(|| format!("open database {}", common.db))?;
    Ok(Arc::new(Store::new(StoreOptions {
        db: Arc::new(db),
        embedder: Arc::new(build_embedder(common)),
        predictor: None,
        reranker: None,
    })))
}

/// Builds the chat model from `--provider` / `--model` / `--base-url`.
/// OpenRouter reads `OPENROUTER_API_KEY` from the environment.
pub(crate) fn build_chat_model(common: &Common) -> anyhow::Result<Arc<dyn Model>> {
    let config = match common.provider {
        Provider::Ollama => ChatModelConfig::ollama(
            common.base_url.clone().unwrap_or_default(),
            common
                .model
                .clone()
                .unwrap_or_else(|| DEFAULT_OLLAMA_CHAT_MODEL.to_string()),
        ),
        Provider::Openrouter => {
            let api_key = std::env::var("OPENROUTER_API_KEY")
                .context("OPENROUTER_API_KEY is not set; export it to use --provider openrouter")?;
            let model = common.model.clone().context(
                "--model is required with --provider openrouter \
                 (e.g. --model anthropic/claude-3.5-haiku)",
            )?;
            ChatModelConfig::openrouter(api_key, model)
        }
        Provider::Vllm => {
            let base_url = common
                .base_url
                .clone()
                .context("--base-url is required with --provider vllm")?;
            let model = common
                .model
                .clone()
                .context("--model is required with --provider vllm")?;
            ChatModelConfig::vllm(base_url, model)
        }
    };
    config
        .build()
        .map_err(|err| anyhow::anyhow!("build chat model: {err}"))
}

/// Wraps a harness error from an embedding-dependent operation with the
/// local-setup fix (`ollama serve` + `ollama pull embeddinggemma`).
pub(crate) fn ollama_hint(err: ditto_harness::Error, action: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "{action}: {err}\n\
         hint: embeddings require a local Ollama server with the embedding model \
         installed — run `ollama serve` and `ollama pull embeddinggemma`"
    )
}

/// Prints aggregated token usage and (when non-zero) monetary cost.
pub(crate) fn print_costs(costs: &[CostedUsage]) {
    if costs.is_empty() {
        return;
    }
    let mut input = 0i64;
    let mut output = 0i64;
    let mut total = 0i64;
    let mut amount = 0f64;
    let mut currency = String::new();
    for item in costs {
        input += item.usage.input_tokens;
        output += item.usage.output_tokens;
        total += item.usage.total_tokens;
        amount += item.cost.amount;
        if currency.is_empty() && !item.cost.currency.is_empty() {
            currency = item.cost.currency.clone();
        }
    }
    println!(
        "tokens: {input} in / {output} out / {total} total ({} calls)",
        costs.len()
    );
    if amount > 0.0 {
        if currency.is_empty() {
            currency = "USD".to_string();
        }
        println!("cost: {amount:.6} {currency}");
    }
}

/// One-line, whitespace-collapsed snippet capped at `max_chars` characters.
pub(crate) fn snippet(s: &str, max_chars: usize) -> String {
    let collapsed = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        collapsed
    } else {
        let head: String = collapsed.chars().take(max_chars).collect();
        format!("{head}...")
    }
}
