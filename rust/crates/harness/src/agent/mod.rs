//! Importable multi-turn agent loop with injectable model/tools, stream-style
//! event hooks, tool loop detection, and cost collection.
//! Port of Go `pkg/agent`.

pub mod loopdetect;

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::memory::{SaveMemoryRequest, Store};
use crate::types::{
    ChatMessage, Content, ContentType, CostCollector, CostedUsage, Error, Model, Result, Tool,
    ToolCall, ToolCallResponse, ToolDefinition,
};

/// Max model turns per run when unset (Go hardcodes 8).
pub const DEFAULT_MAX_TURNS: usize = 8;

/// Synthesis prompt injected after a tool loop is detected
/// (Go: `LoopBreakSynthesisPrompt`).
pub const LOOP_BREAK_SYNTHESIS_PROMPT: &str = "You repeated the same tool call several times without producing a final answer. Stop calling tools and answer the user's request now using the conversation context and the tool results you already have. If the available results are incomplete, say that briefly and answer with what you have.";

/// Tool output substituted for a loop-detected call (Go:
/// `loopBreakToolResult`), as a JSON value.
pub fn loop_break_tool_result() -> Value {
    serde_json::json!({
        "status": "loop_detected",
        "message": "This repeated tool call was stopped. Do not call tools again. Write the final answer using the tool results already available in this conversation."
    })
}

/// The agent loop (Go: `agent.Loop`).
pub struct Loop {
    model: Arc<dyn Model>,
    memory: Option<Arc<Store>>,
    tools: Vec<Arc<dyn Tool>>,
}

/// Constructor options for [`Loop::new`] (Go: `agent.Options`).
pub struct Options {
    pub model: Arc<dyn Model>,
    /// Required only when `RunRequest::save_memory` is used.
    pub memory: Option<Arc<Store>>,
    pub tools: Vec<Arc<dyn Tool>>,
}

/// Request for [`Loop::run`] / [`Loop::run_streaming`]
/// (Go: `agent.RunRequest`). `max_turns` 0 -> [`DEFAULT_MAX_TURNS`].
#[derive(Debug, Clone, Default)]
pub struct RunRequest {
    pub user_id: String,
    pub kg_id: String,
    pub session_id: String,
    pub messages: Vec<ChatMessage>,
    pub max_turns: usize,
    /// When true (and a memory store is configured) the final text is saved
    /// with source "agent_loop".
    pub save_memory: bool,
}

/// Result of a run (Go: `agent.RunResult`). JSON matches Go tags.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunResult {
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub costs: Vec<CostedUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Map<String, Value>>,
}

/// Stream-style observer for loop events (Go: `agent.EventHandler`).
/// All methods default to no-ops, so hosts implement only what they need.
pub trait EventHandler: Send + Sync {
    fn send_chat_content(&self, _text: &str) {}
    fn send_tool_call_progress(&self, _tool_call_id: &str, _data: &Value) {}
    fn send_tool_call_completed(&self, _tool_call_id: &str, _tool_name: &str) {}
    fn send_tool_result(&self, _result: &ToolCallResponse) {}
    fn send_error(&self, _err: &Error) {}
}

/// Handler that ignores every event (Go: `noopHandler`).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopHandler;

impl EventHandler for NoopHandler {}

impl Loop {
    /// Creates a loop (Go: `agent.NewLoop`).
    pub fn new(opts: Options) -> Loop {
        Loop {
            model: opts.model,
            memory: opts.memory,
            tools: opts.tools,
        }
    }

    /// Runs the loop without event callbacks (Go: `Loop.Run`).
    pub async fn run(&self, req: RunRequest) -> Result<RunResult> {
        self.run_streaming(req, &NoopHandler).await
    }

    /// Runs up to `max_turns` model turns (Go: `Loop.RunStreaming`):
    /// each turn calls `Model::next`; a text chunk finishes the run (emitted
    /// via `send_chat_content` and appended as an assistant message); a tool
    /// call is recorded, checked by [`loopdetect::Detector`] (3 identical
    /// consecutive calls -> substitute [`loop_break_tool_result`], inject
    /// [`LOOP_BREAK_SYNTHESIS_PROMPT`] as a user message, and drop all tool
    /// definitions for subsequent turns), executed, and its
    /// [`ToolCallResponse`] appended as a role:"tool" message whose content
    /// part carries the response (with `output` set to the full serialized
    /// response JSON, as Go's `toolMessage` does). Costs from every chunk are
    /// collected. With `save_memory`, the final text is persisted via the
    /// memory store (prompt = last user text).
    pub async fn run_streaming(
        &self,
        req: RunRequest,
        handler: &dyn EventHandler,
    ) -> Result<RunResult> {
        let max_turns = if req.max_turns == 0 {
            DEFAULT_MAX_TURNS
        } else {
            req.max_turns
        };
        let mut messages = req.messages.clone();
        let mut defs: Vec<ToolDefinition> = Vec::with_capacity(self.tools.len());
        let mut tools_by_name: HashMap<String, Arc<dyn Tool>> =
            HashMap::with_capacity(self.tools.len());
        for tool in &self.tools {
            let def = tool.definition();
            tools_by_name.insert(def.name.clone(), Arc::clone(tool));
            defs.push(def);
        }

        let mut costs = CostCollector::default();
        let mut final_text = String::new();
        let mut detector = loopdetect::Detector::default();
        for turn in 0..max_turns {
            let chunk = match self.model.next(&messages, &defs).await {
                Ok(chunk) => chunk,
                Err(err) => {
                    handler.send_error(&err);
                    return Err(err);
                }
            };
            costs.add(chunk.cost.as_ref());
            let Some(tc) = chunk.tool_call else {
                final_text = chunk.text.clone();
                if !chunk.text.is_empty() {
                    handler.send_chat_content(&chunk.text);
                }
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: vec![Content {
                        content_type: Some(ContentType::Text),
                        content: chunk.text,
                        ..Content::default()
                    }],
                    ..ChatMessage::default()
                });
                break;
            };

            let args_string = tool_call_args_string(&tc);
            handler.send_tool_call_progress(
                &tc.id,
                &serde_json::json!({"name": tc.name, "arguments": args_string}),
            );
            handler.send_tool_call_completed(&tc.id, &tc.name);
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                tool_calls: vec![tc.clone()],
                ..ChatMessage::default()
            });

            let key = loopdetect::ToolCallKey {
                name: tc.name.clone(),
                args: args_string,
            };
            if detector.record_turn(turn, &[key]).is_some() {
                let resp = ToolCallResponse {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    output: loop_break_tool_result(),
                    error: String::new(),
                };
                handler.send_tool_result(&resp);
                messages.push(tool_message(resp));
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: vec![Content {
                        content_type: Some(ContentType::Text),
                        content: LOOP_BREAK_SYNTHESIS_PROMPT.to_string(),
                        ..Content::default()
                    }],
                    ..ChatMessage::default()
                });
                defs = Vec::new();
                tools_by_name = HashMap::new();
                continue;
            }

            let mut results = execute_calls(&tools_by_name, std::slice::from_ref(&tc)).await;
            let resp = if results.is_empty() {
                ToolCallResponse {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    error: "tool produced no result".to_string(),
                    ..ToolCallResponse::default()
                }
            } else {
                results.remove(0)
            };
            handler.send_tool_result(&resp);
            messages.push(tool_message(resp));
        }

        if req.save_memory && !final_text.is_empty() {
            if let Some(store) = &self.memory {
                store
                    .save_memory(SaveMemoryRequest {
                        user_id: req.user_id.clone(),
                        kg_id: req.kg_id.clone(),
                        session_id: req.session_id.clone(),
                        prompt: last_user_text(&req.messages),
                        response: final_text.clone(),
                        output: vec![Content {
                            content_type: Some(ContentType::Text),
                            content: final_text.clone(),
                            ..Content::default()
                        }],
                        source: "agent_loop".to_string(),
                        ..SaveMemoryRequest::default()
                    })
                    .await
                    .map_err(|err| Error::Other(format!("save agent memory: {err}")))?;
            }
        }

        Ok(RunResult {
            messages,
            text: final_text,
            costs: costs.into_items(),
            metadata: None,
        })
    }

    /// Executes tool calls concurrently, preserving order; unknown tools
    /// yield an error response (Go: `Loop.ExecuteToolCalls`). Each tool's
    /// `execute` output is wrapped into a [`ToolCallResponse`] with the
    /// call's id/name; `Err` populates `error`.
    pub async fn execute_tool_calls(&self, calls: &[ToolCall]) -> Vec<ToolCallResponse> {
        let mut tools_by_name: HashMap<String, Arc<dyn Tool>> =
            HashMap::with_capacity(self.tools.len());
        for tool in &self.tools {
            tools_by_name.insert(tool.definition().name, Arc::clone(tool));
        }
        execute_calls(&tools_by_name, calls).await
    }

    /// Definitions of the configured tools (Go: `Loop.Tools`).
    pub fn tools(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|tool| tool.definition()).collect()
    }
}

/// Renders a tool call's raw JSON args as the string Go sees via
/// `string(tc.Args)`; `Null` (absent) becomes the empty string.
fn tool_call_args_string(tc: &ToolCall) -> String {
    if tc.args.is_null() {
        String::new()
    } else {
        tc.args.to_string()
    }
}

/// Executes the given calls concurrently against the tool map, preserving
/// input order (Go: `executeToolCalls`).
async fn execute_calls(
    tools_by_name: &HashMap<String, Arc<dyn Tool>>,
    calls: &[ToolCall],
) -> Vec<ToolCallResponse> {
    if calls.is_empty() {
        return Vec::new();
    }
    let futures = calls.iter().map(|call| async move {
        let Some(tool) = tools_by_name.get(&call.name) else {
            return ToolCallResponse {
                id: call.id.clone(),
                name: call.name.clone(),
                error: format!("unknown tool {:?}", call.name),
                ..ToolCallResponse::default()
            };
        };
        match tool.execute(call.args.clone()).await {
            Ok(output) => ToolCallResponse {
                id: call.id.clone(),
                name: call.name.clone(),
                output,
                error: String::new(),
            },
            Err(err) => ToolCallResponse {
                id: call.id.clone(),
                name: call.name.clone(),
                error: err.to_string(),
                ..ToolCallResponse::default()
            },
        }
    });
    futures::future::join_all(futures).await
}

/// Wraps a tool response into a role:"tool" message whose content part
/// carries the response with `output` replaced by the full serialized
/// response JSON (Go: `toolMessage`).
fn tool_message(resp: ToolCallResponse) -> ChatMessage {
    let raw = serde_json::to_value(&resp).unwrap_or(Value::Null);
    let mut result = resp;
    let tool_call_id = result.id.clone();
    result.output = raw;
    ChatMessage {
        role: "tool".to_string(),
        tool_call_id,
        content: vec![Content {
            content_type: Some(ContentType::ToolResult),
            tool_call_response: Some(result),
            ..Content::default()
        }],
        ..ChatMessage::default()
    }
}

/// Extracts the last user message's concatenated text (Go: `lastUserText`).
pub fn last_user_text(messages: &[ChatMessage]) -> String {
    for msg in messages.iter().rev() {
        if msg.role != "user" {
            continue;
        }
        return msg
            .content
            .iter()
            .map(|part| part.content.as_str())
            .collect();
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;

    use super::*;
    use crate::types::{ChatChunk, Cost, Usage};

    struct ScriptedModel {
        chunks: Mutex<Vec<ChatChunk>>,
    }

    impl ScriptedModel {
        fn new(chunks: Vec<ChatChunk>) -> ScriptedModel {
            ScriptedModel {
                chunks: Mutex::new(chunks),
            }
        }
    }

    #[async_trait]
    impl Model for ScriptedModel {
        async fn next(
            &self,
            _messages: &[ChatMessage],
            _tools: &[ToolDefinition],
        ) -> Result<ChatChunk> {
            let mut chunks = self.chunks.lock().expect("lock chunks");
            Ok(chunks.remove(0))
        }
    }

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".to_string(),
                ..ToolDefinition::default()
            }
        }

        async fn execute(&self, args: Value) -> Result<Value> {
            Ok(args)
        }
    }

    #[derive(Default)]
    struct RecordingHandler {
        chat_content: Mutex<Vec<String>>,
        tool_progress_ids: Mutex<Vec<String>>,
        tool_completed_ids: Mutex<Vec<String>>,
        tool_result_ids: Mutex<Vec<String>>,
        errors: Mutex<Vec<String>>,
    }

    impl EventHandler for RecordingHandler {
        fn send_chat_content(&self, text: &str) {
            self.chat_content
                .lock()
                .expect("lock")
                .push(text.to_string());
        }

        fn send_tool_call_progress(&self, tool_call_id: &str, _data: &Value) {
            self.tool_progress_ids
                .lock()
                .expect("lock")
                .push(tool_call_id.to_string());
        }

        fn send_tool_call_completed(&self, tool_call_id: &str, _tool_name: &str) {
            self.tool_completed_ids
                .lock()
                .expect("lock")
                .push(tool_call_id.to_string());
        }

        fn send_tool_result(&self, result: &ToolCallResponse) {
            self.tool_result_ids
                .lock()
                .expect("lock")
                .push(result.id.clone());
        }

        fn send_error(&self, err: &Error) {
            self.errors.lock().expect("lock").push(err.to_string());
        }
    }

    fn tool_call_chunk(id: &str, args: Value) -> ChatChunk {
        ChatChunk {
            tool_call: Some(ToolCall {
                id: id.to_string(),
                name: "echo".to_string(),
                args,
            }),
            ..ChatChunk::default()
        }
    }

    fn text_chunk(text: &str) -> ChatChunk {
        ChatChunk {
            text: text.to_string(),
            ..ChatChunk::default()
        }
    }

    fn user_message(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: vec![Content {
                content: text.to_string(),
                ..Content::default()
            }],
            ..ChatMessage::default()
        }
    }

    fn echo_loop(chunks: Vec<ChatChunk>) -> Loop {
        Loop::new(Options {
            model: Arc::new(ScriptedModel::new(chunks)),
            memory: None,
            tools: vec![Arc::new(EchoTool)],
        })
    }

    #[tokio::test]
    async fn loop_executes_tool_then_returns_final_text() {
        let agent_loop = echo_loop(vec![
            tool_call_chunk("call_1", serde_json::json!({"ok": true})),
            text_chunk("done"),
        ]);

        let result = agent_loop
            .run(RunRequest {
                messages: vec![user_message("hello")],
                ..RunRequest::default()
            })
            .await
            .expect("run");
        assert_eq!(result.text, "done");
        assert_eq!(result.messages.len(), 4, "messages: {:?}", result.messages);
    }

    #[tokio::test]
    async fn loop_streaming_emits_tool_and_content_events() {
        let handler = RecordingHandler::default();
        let agent_loop = echo_loop(vec![
            tool_call_chunk("call_1", serde_json::json!({"ok": true})),
            text_chunk("done"),
        ]);

        let result = agent_loop
            .run_streaming(
                RunRequest {
                    messages: vec![user_message("hello")],
                    ..RunRequest::default()
                },
                &handler,
            )
            .await
            .expect("run_streaming");
        assert_eq!(result.text, "done");
        assert_eq!(*handler.tool_progress_ids.lock().expect("lock"), ["call_1"]);
        assert_eq!(
            *handler.tool_completed_ids.lock().expect("lock"),
            ["call_1"]
        );
        assert_eq!(*handler.tool_result_ids.lock().expect("lock"), ["call_1"]);
        assert_eq!(*handler.chat_content.lock().expect("lock"), ["done"]);
    }

    #[tokio::test]
    async fn loop_breaks_repeated_tool_call_and_synthesizes_final_answer() {
        let same_args = serde_json::json!({"q": "same"});
        let agent_loop = echo_loop(vec![
            tool_call_chunk("call_1", same_args.clone()),
            tool_call_chunk("call_2", same_args.clone()),
            tool_call_chunk("call_3", same_args),
            text_chunk("final after loop break"),
        ]);

        let result = agent_loop
            .run(RunRequest {
                messages: vec![user_message("hello")],
                max_turns: 5,
                ..RunRequest::default()
            })
            .await
            .expect("run");
        assert_eq!(result.text, "final after loop break");
        let saw_prompt = result.messages.iter().any(|msg| {
            msg.role == "user"
                && msg
                    .content
                    .iter()
                    .any(|content| content.content == LOOP_BREAK_SYNTHESIS_PROMPT)
        });
        assert!(saw_prompt, "loop-break synthesis prompt was not appended");
    }

    #[tokio::test]
    async fn execute_tool_calls_preserves_input_order() {
        let agent_loop = echo_loop(vec![]);
        let results = agent_loop
            .execute_tool_calls(&[
                ToolCall {
                    id: "call_a".to_string(),
                    name: "echo".to_string(),
                    args: serde_json::json!({"a": true}),
                },
                ToolCall {
                    id: "call_b".to_string(),
                    name: "missing".to_string(),
                    args: serde_json::json!({}),
                },
            ])
            .await;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "call_a");
        assert!(results[0].error.is_empty(), "{:?}", results[0]);
        assert_eq!(results[1].id, "call_b");
        assert_eq!(results[1].error, "unknown tool \"missing\"");
    }

    #[tokio::test]
    async fn loop_collects_costs_across_turns() {
        let agent_loop = echo_loop(vec![
            ChatChunk {
                cost: Some(CostedUsage {
                    usage: Usage {
                        model: "m".to_string(),
                        total_tokens: 5,
                        ..Usage::default()
                    },
                    cost: Cost {
                        currency: "USD".to_string(),
                        amount: 0.01,
                    },
                }),
                ..tool_call_chunk("call_1", serde_json::json!({"ok": true}))
            },
            ChatChunk {
                cost: Some(CostedUsage {
                    usage: Usage {
                        model: "m".to_string(),
                        total_tokens: 7,
                        ..Usage::default()
                    },
                    cost: Cost {
                        currency: "USD".to_string(),
                        amount: 0.02,
                    },
                }),
                ..text_chunk("done")
            },
        ]);
        let result = agent_loop
            .run(RunRequest {
                messages: vec![user_message("hello")],
                ..RunRequest::default()
            })
            .await
            .expect("run");
        assert_eq!(result.costs.len(), 2);
        assert_eq!(result.costs[0].usage.total_tokens, 5);
        assert_eq!(result.costs[1].usage.total_tokens, 7);
    }

    #[test]
    fn last_user_text_concatenates_parts_of_last_user_message() {
        let messages = vec![
            user_message("first"),
            ChatMessage {
                role: "assistant".to_string(),
                content: vec![Content::text("reply")],
                ..ChatMessage::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: vec![Content::text("a"), Content::text("b")],
                ..ChatMessage::default()
            },
        ];
        assert_eq!(last_user_text(&messages), "ab");
        assert_eq!(last_user_text(&[]), "");
    }

    #[test]
    fn tool_message_wraps_serialized_response() {
        let msg = tool_message(ToolCallResponse {
            id: "call_1".to_string(),
            name: "echo".to_string(),
            output: serde_json::json!({"ok": true}),
            error: String::new(),
        });
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id, "call_1");
        let part = &msg.content[0];
        assert_eq!(part.content_type, Some(ContentType::ToolResult));
        let resp = part.tool_call_response.as_ref().expect("response");
        // Output carries the full serialized response, like Go's toolMessage.
        assert_eq!(
            resp.output,
            serde_json::json!({"id": "call_1", "name": "echo", "output": {"ok": true}})
        );
    }
}
