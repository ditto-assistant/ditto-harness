// SPDX-License-Identifier: MIT
//! `chat` subcommand: one chat turn through `chat::Harness` with memory
//! tools enabled, streaming text to stdout and tool events to stderr, then
//! printing cost and which memory tools were called.

use std::sync::{Arc, Mutex};

use ditto_harness::agent::EventHandler;
use ditto_harness::chat::{Harness, Options, PrepareRequest, RunRequest};
use ditto_harness::types::{Error, ToolCallResponse};

use crate::commands::util;
use crate::Common;

/// Prints chat text to stdout and tool/error events to stderr, recording
/// which tools were called.
#[derive(Default)]
struct CliEvents {
    tool_calls: Mutex<Vec<String>>,
}

impl EventHandler for CliEvents {
    fn send_chat_content(&self, text: &str) {
        println!("{text}");
    }

    fn send_tool_call_progress(&self, tool_call_id: &str, data: &serde_json::Value) {
        let name = data
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let args = data.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
        eprintln!(
            "-> tool call {name} [{tool_call_id}] {}",
            util::snippet(args, 160)
        );
        if let Ok(mut calls) = self.tool_calls.lock() {
            calls.push(name.to_string());
        }
    }

    fn send_tool_result(&self, result: &ToolCallResponse) {
        if result.error.is_empty() {
            eprintln!("<- tool result {}: ok", result.name);
        } else {
            eprintln!("<- tool result {}: error: {}", result.name, result.error);
        }
    }

    fn send_error(&self, err: &Error) {
        eprintln!("model error: {err}");
    }
}

pub async fn run(common: &Common, message: &str) -> anyhow::Result<()> {
    anyhow::ensure!(!message.trim().is_empty(), "--message must not be empty");
    let store = util::open_store(common).await?;
    let model = util::build_chat_model(common)?;
    let harness = Harness::new(Options {
        model,
        memory: Some(Arc::clone(&store)),
        tools: Vec::new(),
        include_memory_tools: true,
    });

    let events = CliEvents::default();
    let result = harness
        .run(
            RunRequest {
                prepare: PrepareRequest {
                    user_id: common.user.clone(),
                    user_input: message.to_string(),
                    use_composite: true,
                    ..PrepareRequest::default()
                },
                save_memory: true,
                ..RunRequest::default()
            },
            &events,
        )
        .await
        .map_err(|err| util::ollama_hint(err, "chat"))?;

    if result.result.text.is_empty() {
        eprintln!("(model produced no final text)");
    }
    let calls = events
        .tool_calls
        .lock()
        .map(|c| c.clone())
        .unwrap_or_default();
    if calls.is_empty() {
        eprintln!("memory tools called: none");
    } else {
        eprintln!("memory tools called: {}", calls.join(", "));
    }
    util::print_costs(&result.result.costs);
    if let Some(saved) = &result.saved_memory {
        eprintln!("saved memory pair: {}", saved.id);
    }
    Ok(())
}
