// SPDX-License-Identifier: MIT
//! Importable multi-turn agent loop with injectable model/tools, stream-style
//! event hooks, pre-execution repetition detection, and cost collection.

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

/// Max model turns per run when unset.
pub const DEFAULT_MAX_TURNS: usize = 8;

/// Legacy loop-break prompt, retained for callers that display old transcripts.
pub const LOOP_BREAK_SYNTHESIS_PROMPT: &str = "You repeated the same tool call several times without producing a final answer. Stop calling tools and answer the user's request now using the conversation context and the tool results you already have. If the available results are incomplete, say that briefly and answer with what you have.";

/// Legacy loop-break result, retained for callers that display old transcripts.
pub fn loop_break_tool_result() -> Value {
    serde_json::json!({
        "status": "loop_detected",
        "message": "This repeated tool call was stopped. Do not call tools again. Write the final answer using the tool results already available in this conversation."
    })
}

/// The agent loop.
pub struct Loop {
    model: Arc<dyn Model>,
    memory: Option<Arc<Store>>,
    tools: Vec<Arc<dyn Tool>>,
}

/// Constructor options for [`Loop::new`].
pub struct Options {
    pub model: Arc<dyn Model>,
    /// Required only when `RunRequest::save_memory` is used.
    pub memory: Option<Arc<Store>>,
    pub tools: Vec<Arc<dyn Tool>>,
}

/// Request for [`Loop::run`] / [`Loop::run_streaming`].
/// `max_turns` 0 -> [`DEFAULT_MAX_TURNS`].
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

/// Result of a run. JSON field names match the original wire format.
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

/// Stream-style observer for loop events.
/// All methods default to no-ops, so hosts implement only what they need.
pub trait EventHandler: Send + Sync {
    /// Fires when a turn yields non-empty final chat text, ending the run.
    fn send_chat_content(&self, _text: &str) {}
    /// Fires when the model emits a tool call, carrying its name/arguments; before execution.
    fn send_tool_call_progress(&self, _tool_call_id: &str, _data: &Value) {}
    /// Fires when the model finished streaming the call block — emitted BEFORE the tool executes.
    fn send_tool_call_completed(&self, _tool_call_id: &str, _tool_name: &str) {}
    /// Decides a second or later identical call before execution. The previous
    /// response is supplied so the host can distinguish a successful effect
    /// from a pre-delivery failure. The default preserves model-selected calls;
    /// hosts with side-effect contracts can block an unapproved repeat.
    fn decide_repeated_tool_call(
        &self,
        _detection: &loopdetect::Detection,
        _previous_response: Option<&ToolCallResponse>,
    ) -> RepeatedCallDecision {
        RepeatedCallDecision::Execute
    }
    /// Fires after the tool executed.
    fn send_tool_result(&self, _result: &ToolCallResponse) {}
    /// Fires when a model call fails, just before the run returns that error.
    fn send_error(&self, _err: &Error) {}
}

/// Handler that ignores every event.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopHandler;

impl EventHandler for NoopHandler {}

/// A host's decision for a detected repeated tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatedCallDecision {
    Execute,
    Block,
}

impl Loop {
    /// Creates a loop.
    pub fn new(opts: Options) -> Loop {
        Loop {
            model: opts.model,
            memory: opts.memory,
            tools: opts.tools,
        }
    }

    /// Runs the loop without event callbacks.
    pub async fn run(&self, req: RunRequest) -> Result<RunResult> {
        self.run_streaming(req, &NoopHandler).await
    }

    /// Runs up to `max_turns` model turns: a text chunk finishes the run; a
    /// tool call is executed and appended as a role:"tool" message, with a
    /// bounded number of model turns. With `save_memory`, the final text is
    /// persisted via the memory store. Identical tool calls are still executed:
    /// only the caller can know whether a repetition was separately requested.
    ///
    /// "Streaming" here means per-event [`EventHandler`] hooks (chat content,
    /// tool progress/results, errors), not token streaming: each turn is one
    /// whole-chunk `Model::next` call.
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
        let mut defs: Vec<ToolDefinition> = Vec::with_capacity(self.tools.len());
        let mut tools_by_name: HashMap<String, Arc<dyn Tool>> =
            HashMap::with_capacity(self.tools.len());
        for tool in &self.tools {
            let def = tool.definition();
            tools_by_name.insert(def.name.clone(), Arc::clone(tool));
            defs.push(def);
        }
        let mut state = TurnState {
            messages: req.messages.clone(),
            detector: loopdetect::Detector::default(),
            last_result: None,
            defs,
            tools_by_name,
        };

        let mut costs = CostCollector::default();
        let mut final_text = String::new();
        let mut finished = false;
        for turn in 0..max_turns {
            let chunk = match self.model.next(&state.messages, &state.defs).await {
                Ok(chunk) => chunk,
                Err(err) => {
                    handler.send_error(&err);
                    return Err(err);
                }
            };
            costs.add(chunk.cost.as_ref());
            let Some(tc) = chunk.tool_call else {
                finished = true;
                final_text = chunk.text.clone();
                if !chunk.text.is_empty() {
                    handler.send_chat_content(&chunk.text);
                }
                state.messages.push(ChatMessage {
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

            handle_tool_call(handler, &mut state, turn, tc).await;
        }

        if req.save_memory && !final_text.is_empty() {
            self.save_final_memory(&req, &final_text).await?;
        }

        // Exhausting max_turns mid-tool-loop is distinguishable from an
        // empty final answer via metadata (additive; absent otherwise).
        let metadata = if finished {
            None
        } else {
            let mut map = serde_json::Map::new();
            map.insert(
                "stop_reason".to_string(),
                Value::String("max_turns".to_string()),
            );
            Some(map)
        };

        Ok(RunResult {
            messages: state.messages,
            text: final_text,
            costs: costs.into_items(),
            metadata,
        })
    }

    /// Persists the run's final text via the memory store (source
    /// "agent_loop", prompt = last user text of the request messages).
    async fn save_final_memory(&self, req: &RunRequest, final_text: &str) -> Result<()> {
        let Some(store) = &self.memory else {
            return Ok(());
        };
        store
            .save_memory(SaveMemoryRequest {
                user_id: req.user_id.clone(),
                kg_id: req.kg_id.clone(),
                session_id: req.session_id.clone(),
                prompt: last_user_text(&req.messages),
                response: final_text.to_string(),
                output: vec![Content {
                    content_type: Some(ContentType::Text),
                    content: final_text.to_string(),
                    ..Content::default()
                }],
                source: "agent_loop".to_string(),
                ..SaveMemoryRequest::default()
            })
            .await
            .map_err(|err| Error::Other(format!("save agent memory: {err}")))?;
        Ok(())
    }

    /// Executes tool calls concurrently, preserving order; unknown tools
    /// yield an error response. Each tool's
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

    /// Definitions of the configured tools.
    pub fn tools(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|tool| tool.definition()).collect()
    }
}

/// Mutable per-run state owned by [`Loop::run_streaming`] and threaded
/// through each tool-call turn.
struct TurnState {
    messages: Vec<ChatMessage>,
    detector: loopdetect::Detector,
    last_result: Option<(loopdetect::ToolCallKey, ToolCallResponse)>,
    defs: Vec<ToolDefinition>,
    tools_by_name: HashMap<String, Arc<dyn Tool>>,
}

/// Handles one tool-call turn: emits the tool events, appends the assistant
/// tool-call message, then executes the call and appends its role:"tool" response.
async fn handle_tool_call(
    handler: &dyn EventHandler,
    state: &mut TurnState,
    turn: usize,
    tc: ToolCall,
) {
    let args_string = tool_call_args_string(&tc);
    handler.send_tool_call_progress(
        &tc.id,
        &serde_json::json!({"name": tc.name, "arguments": args_string}),
    );
    handler.send_tool_call_completed(&tc.id, &tc.name);
    state.messages.push(ChatMessage {
        role: "assistant".to_string(),
        tool_calls: vec![tc.clone()],
        ..ChatMessage::default()
    });

    let key = loopdetect::ToolCallKey {
        name: tc.name.clone(),
        args: args_string,
    };
    if let Some(detection) = state.detector.record_turn(turn, std::slice::from_ref(&key)) {
        let previous_response = state
            .last_result
            .as_ref()
            .and_then(|(previous_key, response)| (previous_key == &key).then_some(response));
        if handler.decide_repeated_tool_call(&detection, previous_response)
            == RepeatedCallDecision::Block
        {
            let resp = ToolCallResponse {
                id: tc.id.clone(),
                name: tc.name.clone(),
                output: serde_json::json!({"status": "blocked_before_execution"}),
                error: "repeated tool call blocked before execution".to_string(),
            };
            handler.send_tool_result(&resp);
            state.messages.push(tool_message(resp));
            return;
        }
    }

    let mut results = execute_calls(&state.tools_by_name, std::slice::from_ref(&tc)).await;
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
    state.last_result = Some((key, resp.clone()));
    state.messages.push(tool_message(resp));
}

/// Renders a tool call's raw JSON args as a compact JSON string;
/// `Null` (absent) becomes the empty string.
fn tool_call_args_string(tc: &ToolCall) -> String {
    if tc.args.is_null() {
        String::new()
    } else {
        tc.args.to_string()
    }
}

/// Executes the given calls concurrently against the tool map, preserving
/// input order.
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
/// response JSON.
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

/// Extracts the last user message's concatenated text.
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
        repeated_calls: Mutex<Vec<loopdetect::Detection>>,
        execution_events: Mutex<Vec<String>>,
        block_successful_repeats: bool,
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

        fn decide_repeated_tool_call(
            &self,
            detection: &loopdetect::Detection,
            previous_response: Option<&ToolCallResponse>,
        ) -> RepeatedCallDecision {
            self.repeated_calls
                .lock()
                .expect("lock")
                .push(detection.clone());
            self.execution_events
                .lock()
                .expect("lock")
                .push("repeat".to_string());
            if self.block_successful_repeats
                && previous_response.is_some_and(|response| response.error.is_empty())
            {
                RepeatedCallDecision::Block
            } else {
                RepeatedCallDecision::Execute
            }
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
            self.execution_events
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
    async fn loop_preserves_requested_repetitions_and_later_tools() {
        let same_args = serde_json::json!({"q": "same"});
        let handler = RecordingHandler::default();
        let agent_loop = echo_loop(vec![
            tool_call_chunk("call_1", same_args.clone()),
            tool_call_chunk("call_2", same_args.clone()),
            tool_call_chunk("call_3", same_args),
            tool_call_chunk("call_4", serde_json::json!({"q": "different"})),
            text_chunk("final after requested calls"),
        ]);

        let result = agent_loop
            .run_streaming(
                RunRequest {
                    messages: vec![user_message("send it three times, then do another task")],
                    max_turns: 5,
                    ..RunRequest::default()
                },
                &handler,
            )
            .await
            .expect("run");
        assert_eq!(result.text, "final after requested calls");
        let repeated = handler.repeated_calls.lock().expect("lock");
        assert_eq!(repeated.len(), 2);
        assert_eq!(repeated[0].turn, 1);
        assert_eq!(repeated[0].consecutive_calls, 2);
        assert_eq!(repeated[1].turn, 2);
        assert_eq!(repeated[1].consecutive_calls, 3);
        assert_eq!(
            *handler.execution_events.lock().expect("lock"),
            ["call_1", "repeat", "call_2", "repeat", "call_3", "call_4"]
                .map(str::to_string)
                .to_vec()
        );
        assert_eq!(
            result
                .messages
                .iter()
                .filter(|msg| msg.role == "tool")
                .count(),
            4,
            "every model-emitted call must have a tool response"
        );
        let tool_outputs: Vec<&Value> = result
            .messages
            .iter()
            .filter(|msg| msg.role == "tool")
            .filter_map(|msg| msg.content.first())
            .filter_map(|part| part.tool_call_response.as_ref())
            .map(|response| &response.output)
            .collect();
        assert_eq!(tool_outputs[2]["output"], serde_json::json!({"q": "same"}));
        assert_eq!(
            tool_outputs[3]["output"],
            serde_json::json!({"q": "different"})
        );
        assert!(!result.messages.iter().any(|msg| {
            msg.role == "user"
                && msg
                    .content
                    .iter()
                    .any(|content| content.content == LOOP_BREAK_SYNTHESIS_PROMPT)
        }));
    }

    #[tokio::test]
    async fn host_blocks_successful_duplicate_before_execution_without_revoking_tools() {
        let handler = RecordingHandler {
            block_successful_repeats: true,
            ..RecordingHandler::default()
        };
        let agent_loop = echo_loop(vec![
            tool_call_chunk("call_1", serde_json::json!({"q": "same"})),
            tool_call_chunk("call_2", serde_json::json!({"q": "same"})),
            tool_call_chunk("call_3", serde_json::json!({"q": "different"})),
            text_chunk("done"),
        ]);
        let result = agent_loop
            .run_streaming(
                RunRequest {
                    messages: vec![user_message("call once, then do another task")],
                    max_turns: 4,
                    ..RunRequest::default()
                },
                &handler,
            )
            .await
            .expect("run");
        let outputs: Vec<&Value> = result
            .messages
            .iter()
            .filter(|msg| msg.role == "tool")
            .filter_map(|msg| msg.content.first())
            .filter_map(|part| part.tool_call_response.as_ref())
            .map(|response| &response.output)
            .collect();
        assert_eq!(outputs[0]["output"], serde_json::json!({"q": "same"}));
        assert_eq!(outputs[1]["output"]["status"], "blocked_before_execution");
        assert_eq!(outputs[2]["output"], serde_json::json!({"q": "different"}));
        assert_eq!(
            *handler.execution_events.lock().expect("lock"),
            ["call_1", "repeat", "call_2", "call_3"]
                .map(str::to_string)
                .to_vec()
        );
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
        // Output carries the full serialized response.
        assert_eq!(
            resp.output,
            serde_json::json!({"id": "call_1", "name": "echo", "output": {"ok": true}})
        );
    }
}
