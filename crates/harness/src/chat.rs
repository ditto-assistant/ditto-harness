// SPDX-License-Identifier: MIT
//! Chat harness facade: prepares memory context, combines injected tools with
//! memory tools, runs the agent loop, and saves the resulting memory pair.

use std::sync::Arc;

use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{self, EventHandler};
use crate::memory::{
    memory_tools, prompt_memory_title, PromptMemoryRequest, PromptMemoryResponse,
    SaveMemoryRequest, Store, SubjectInput,
};
use crate::retrieval::Variant;
use crate::types::{
    kg_id, ChatMessage, Content, Error, Memory, Model, Result, Tool, ToolDefinition,
    MAIN_SESSION_ID,
};

/// Chat harness.
pub struct Harness {
    model: Arc<dyn Model>,
    memory: Option<Arc<Store>>,
    tools: Vec<Arc<dyn Tool>>,
    include_memory_tools: bool,
}

/// Constructor options.
pub struct Options {
    pub model: Arc<dyn Model>,
    pub memory: Option<Arc<Store>>,
    /// Host application tools, combined with memory tools per run.
    pub tools: Vec<Arc<dyn Tool>>,
    /// When true (and memory is set), the five standard memory tools are
    /// appended for the request's user/kg ids.
    pub include_memory_tools: bool,
}

/// Request for [`Harness::prepare`].
/// Defaults: empty `kg_id` -> derived, empty `session_id` -> "main".
#[derive(Debug, Clone, Default)]
pub struct PrepareRequest {
    pub user_id: String,
    pub kg_id: String,
    pub session_id: String,
    /// Drives memory retrieval; empty skips it.
    pub user_input: String,
    /// Prepended as a system message when non-blank.
    pub system_prompt: String,
    /// Existing conversation; when it contains no non-system messages,
    /// `user_input` becomes the first user message.
    pub messages: Vec<ChatMessage>,
    pub long_term_limit: usize,
    pub short_term_limit: usize,
    pub candidate_pool_size: usize,
    pub exclude_pair_ids: Vec<String>,
    pub variant: Variant,
    pub request_path: String,
    pub log_retrieval: bool,
    /// Selects the long-term retrieval path. **Defaults to false**: plain
    /// vector search, bypassing composite scoring, the [`WeightPredictor`]
    /// (learned weights), and the [`Reranker`] entirely — the subject graph
    /// contributes nothing to ranking. Set true to route through
    /// `Store::search_composite_memories` (the full production-parity stack;
    /// the DittoBench reference baseline runs with true).
    ///
    /// [`WeightPredictor`]: crate::retrieval::WeightPredictor
    /// [`Reranker`]: crate::retrieval::Reranker
    pub use_composite: bool,
}

/// Result of preparation. JSON field names match the original wire format.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrepareResult {
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub memories: PromptMemoryResponse,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub already_found_pair_ids: Vec<String>,
}

/// Request for [`Harness::run`]: a [`PrepareRequest`] plus run options.
#[derive(Debug, Clone, Default)]
pub struct RunRequest {
    pub prepare: PrepareRequest,
    pub max_turns: usize,
    pub save_memory: bool,
    /// Saved memory source; empty -> "chatv2".
    pub source: String,
    pub source_context: String,
    /// `None` -> now (UTC).
    pub timestamp: Option<DateTime<Utc>>,
    pub timezone_offset: i32,
    pub subjects: Vec<SubjectInput>,
}

/// Result of a full run: an [`agent::RunResult`] plus preparation and
/// saved-memory details.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunResult {
    /// Flattened in JSON.
    #[serde(flatten)]
    pub result: agent::RunResult,
    #[serde(default, skip_serializing_if = "is_default_preparation")]
    pub preparation: PromptMemoryResponse,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved_memory: Option<Memory>,
}

fn is_default_preparation(p: &PromptMemoryResponse) -> bool {
    p.long_term.is_empty() && p.short_term.is_empty() && p.ids.is_empty()
}

impl Harness {
    /// Creates a harness.
    pub fn new(opts: Options) -> Harness {
        Harness {
            model: opts.model,
            memory: opts.memory,
            tools: opts.tools,
            include_memory_tools: opts.include_memory_tools,
        }
    }

    /// Normalizes messages (system prompt first; `user_input` becomes the
    /// first user message when the history is empty), retrieves prompt
    /// memories when memory + user input are present, inserts the memory
    /// context message after the leading system messages, and returns
    /// messages + tool definitions + memories.
    /// Errors when `user_id` is empty.
    pub async fn prepare(&self, req: PrepareRequest) -> Result<PrepareResult> {
        if req.user_id.is_empty() {
            return Err(Error::InvalidArgument(
                "chatv2: user id is required".to_string(),
            ));
        }
        let mut req = req;
        let (resolved_kg_id, resolved_session_id) = resolve_ids(&req);
        req.kg_id = resolved_kg_id;
        req.session_id = resolved_session_id;
        let mut messages = normalize_messages(&req.messages, &req.user_input, &req.system_prompt);

        let mut memories = PromptMemoryResponse::default();
        if let Some(store) = &self.memory {
            if !req.user_input.is_empty() {
                memories = store
                    .get_prompt_memories(PromptMemoryRequest {
                        user_id: req.user_id.clone(),
                        kg_id: req.kg_id.clone(),
                        session_id: req.session_id.clone(),
                        query: req.user_input.clone(),
                        long_term_limit: req.long_term_limit,
                        short_term_limit: req.short_term_limit,
                        candidate_pool_size: req.candidate_pool_size,
                        exclude_pair_ids: req.exclude_pair_ids.clone(),
                        variant: req.variant,
                        request_path: req.request_path.clone(),
                        log_retrieval: req.log_retrieval,
                        use_composite: req.use_composite,
                    })
                    .await?;
                if let Some(msg) = memory_context_message(&memories) {
                    messages = insert_after_system(messages, msg);
                }
            }
        }

        let tools = self.tools_for(&req.user_id, &req.kg_id);
        let defs = tools.iter().map(|tool| tool.definition()).collect();
        Ok(PrepareResult {
            messages,
            tools: defs,
            already_found_pair_ids: memories.ids.clone(),
            memories,
        })
    }

    /// Prepares, runs the agent loop with combined tools, and (with
    /// `save_memory`) persists the final exchange with seed/retrieval
    /// metadata from preparation. The saved memory's
    /// input is the last user message content (falling back to
    /// `user_input`), output is the final text, source defaults to "chatv2".
    pub async fn run(&self, req: RunRequest, handler: &dyn EventHandler) -> Result<RunResult> {
        let prepared = self.prepare(req.prepare.clone()).await?;
        let (resolved_kg_id, resolved_session_id) = resolve_ids(&req.prepare);
        let agent_loop = self.agent_loop(&req.prepare);
        let result = agent_loop
            .run_streaming(
                agent::RunRequest {
                    user_id: req.prepare.user_id.clone(),
                    kg_id: resolved_kg_id.clone(),
                    session_id: resolved_session_id.clone(),
                    messages: prepared.messages.clone(),
                    max_turns: req.max_turns,
                    save_memory: false,
                },
                handler,
            )
            .await?;

        let mut saved = None;
        if req.save_memory && !result.text.is_empty() {
            if let Some(store) = &self.memory {
                let mem = store
                    .save_memory(SaveMemoryRequest {
                        user_id: req.prepare.user_id.clone(),
                        kg_id: resolved_kg_id,
                        session_id: resolved_session_id,
                        prompt: req.prepare.user_input.clone(),
                        response: result.text.clone(),
                        input: last_user_input(&prepared.messages, &req.prepare.user_input),
                        output: vec![Content::text(result.text.clone())],
                        source: first_non_empty(&req.source, "chatv2"),
                        source_context: req.source_context.clone(),
                        timestamp: req.timestamp,
                        timezone_offset: req.timezone_offset,
                        seed_memories: prepared.memories.seed_memory_nodes.clone(),
                        retrieval_metadata: prepared.memories.retrieval_metadata.clone(),
                        subjects: req.subjects.clone(),
                        ..SaveMemoryRequest::default()
                    })
                    .await?;
                saved = Some(mem);
            }
        }
        Ok(RunResult {
            result,
            preparation: prepared.memories,
            saved_memory: saved,
        })
    }

    /// An agent loop wired with this harness's model and the combined tools
    /// for the request's user/kg ids.
    pub fn agent_loop(&self, req: &PrepareRequest) -> agent::Loop {
        agent::Loop::new(agent::Options {
            model: Arc::clone(&self.model),
            memory: None,
            tools: self.tools_for(&req.user_id, &req.kg_id),
        })
    }

    /// Injected host tools plus (when enabled) the standard memory tools for
    /// the given user/kg ids.
    fn tools_for(&self, user_id: &str, kg_id: &str) -> Vec<Arc<dyn Tool>> {
        let mut tools: Vec<Arc<dyn Tool>> = self.tools.to_vec();
        if self.include_memory_tools {
            if let Some(store) = &self.memory {
                tools.extend(
                    memory_tools(Arc::clone(store), user_id, kg_id)
                        .into_iter()
                        .map(|boxed| Arc::from(boxed) as Arc<dyn Tool>),
                );
            }
        }
        tools
    }
}

/// Builds the system message carrying memory context: role "system", text
/// `"Relevant memory context for this turn:\n" + JSON` where the JSON object
/// holds `longTerm` (the raw `long_term_json` payload) and `shortTerm`
/// (compact entries: pairID, timestamp, title, optional summary). Returns
/// `None` when there are no memories.
pub fn memory_context_message(memories: &PromptMemoryResponse) -> Option<ChatMessage> {
    if memories.long_term.is_empty() && memories.short_term.is_empty() {
        return None;
    }
    let mut payload = serde_json::Map::new();
    if !memories.long_term.is_empty() {
        // Embedded as raw JSON. A malformed long-term payload must not drop
        // the short-term context too, so log and fall through.
        match serde_json::from_str::<Value>(&memories.long_term_json) {
            Ok(raw) => {
                payload.insert("longTerm".to_string(), raw);
            }
            Err(err) => {
                tracing::warn!(error = %err, "memory context: long_term_json failed to parse; emitting short-term context only");
            }
        }
    }
    if !memories.short_term.is_empty() {
        payload.insert(
            "shortTerm".to_string(),
            Value::Array(compact_memories(&memories.short_term)),
        );
    }
    if payload.is_empty() {
        return None;
    }
    let raw = serde_json::to_string(&Value::Object(payload)).ok()?;
    Some(ChatMessage {
        role: "system".to_string(),
        content: vec![Content::text(format!(
            "Relevant memory context for this turn:\n{raw}"
        ))],
        ..ChatMessage::default()
    })
}

/// Prepends the system prompt (when non-blank) and seeds the conversation
/// with `user_input` when there are no non-system messages, so the model
/// always sees the user's turn even when a system prompt is set.
fn normalize_messages(
    messages: &[ChatMessage],
    user_input: &str,
    system_prompt: &str,
) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    if !system_prompt.trim().is_empty() {
        out.push(ChatMessage {
            role: "system".to_string(),
            content: vec![Content::text(system_prompt)],
            ..ChatMessage::default()
        });
    }
    out.extend_from_slice(messages);
    let has_non_system = out.iter().any(|msg| msg.role != "system");
    if !has_non_system && !user_input.trim().is_empty() {
        out.push(ChatMessage {
            role: "user".to_string(),
            content: vec![Content::text(user_input)],
            ..ChatMessage::default()
        });
    }
    out
}

/// Inserts `msg` before the first non-system message at index > 0, else
/// appends it. The `i > 0` guard means a history with no system prompt (a
/// non-system message at index 0) gets the memory context appended after it
/// rather than prepended — upstream-parity behavior kept deliberately.
fn insert_after_system(messages: Vec<ChatMessage>, msg: ChatMessage) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    let mut inserted = false;
    for (i, existing) in messages.into_iter().enumerate() {
        if !inserted && i > 0 && existing.role != "system" {
            out.push(msg.clone());
            inserted = true;
        }
        out.push(existing);
    }
    if !inserted {
        out.push(msg);
    }
    out
}

/// Compact short-term entries for the memory context payload: pairID,
/// timestamp (RFC3339), title, optional summary.
fn compact_memories(memories: &[Memory]) -> Vec<Value> {
    memories
        .iter()
        .map(|mem| {
            let mut item = serde_json::Map::new();
            item.insert("pairID".to_string(), Value::String(mem.id.clone()));
            if !mem.summary.is_empty() {
                item.insert("summary".to_string(), Value::String(mem.summary.clone()));
            }
            item.insert(
                "timestamp".to_string(),
                Value::String(mem.timestamp.to_rfc3339_opts(SecondsFormat::Secs, true)),
            );
            item.insert("title".to_string(), Value::String(prompt_memory_title(mem)));
            Value::Object(item)
        })
        .collect()
}

/// Content of the last user message, falling back to `fallback` as a single
/// text part.
fn last_user_input(messages: &[ChatMessage], fallback: &str) -> Vec<Content> {
    for msg in messages.iter().rev() {
        if msg.role == "user" && !msg.content.is_empty() {
            return msg.content.clone();
        }
    }
    if fallback.is_empty() {
        return Vec::new();
    }
    vec![Content::text(fallback)]
}

/// Resolved `(kg_id, session_id)` for a request: an empty `kg_id` derives
/// from `user_id`, an empty `session_id` falls back to "main". Used by both
/// `prepare` (which normalizes its own clone) and `run` (which saves against
/// the caller's request).
fn resolve_ids(req: &PrepareRequest) -> (String, String) {
    (
        first_non_empty(&req.kg_id, &kg_id(&req.user_id)),
        first_non_empty(&req.session_id, MAIN_SESSION_ID),
    )
}

/// First non-empty of the two values.
fn first_non_empty(value: &str, fallback: &str) -> String {
    if !value.is_empty() {
        value.to_string()
    } else {
        fallback.to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::sync::Mutex;

    use async_trait::async_trait;
    use chrono::TimeZone;

    use super::*;
    use crate::agent::NoopHandler;
    use crate::db::Db;
    use crate::memory::StoreOptions;
    use crate::types::{
        ChatChunk, Cost, CostedUsage, EmbedRequest, EmbedResponse, Embedder, Usage,
    };

    /// Scripted model recording the messages/tools of its latest call.
    #[derive(Default)]
    struct ScriptedModel {
        chunks: Mutex<Vec<ChatChunk>>,
        seen_tools: Mutex<Vec<String>>,
        seen_messages: Mutex<Vec<ChatMessage>>,
    }

    impl ScriptedModel {
        fn new(chunks: Vec<ChatChunk>) -> ScriptedModel {
            ScriptedModel {
                chunks: Mutex::new(chunks),
                ..ScriptedModel::default()
            }
        }
    }

    #[async_trait]
    impl Model for ScriptedModel {
        async fn next(
            &self,
            messages: &[ChatMessage],
            tools: &[ToolDefinition],
        ) -> Result<ChatChunk> {
            *self.seen_messages.lock().expect("lock") = messages.to_vec();
            *self.seen_tools.lock().expect("lock") =
                tools.iter().map(|tool| tool.name.clone()).collect();
            Ok(self.chunks.lock().expect("lock").remove(0))
        }
    }

    /// Host tool stand-in.
    struct MarkerTool;

    #[async_trait]
    impl Tool for MarkerTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "marker".to_string(),
                ..ToolDefinition::default()
            }
        }

        async fn execute(&self, args: Value) -> Result<Value> {
            Ok(args)
        }
    }

    /// Deterministic token-bucket embedder.
    struct HashEmbedder;

    #[async_trait]
    impl Embedder for HashEmbedder {
        async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse> {
            let embeddings = req.texts.iter().map(|text| hash_embedding(text)).collect();
            Ok(EmbedResponse {
                embeddings,
                ..EmbedResponse::default()
            })
        }
    }

    fn hash_embedding(text: &str) -> Vec<f32> {
        let mut vec = vec![0f32; 768];
        for token in text.to_lowercase().split_whitespace() {
            let mut hasher = DefaultHasher::new();
            token.hash(&mut hasher);
            let idx = (hasher.finish() % 768) as usize;
            vec[idx] += 1.0;
        }
        let norm: f64 = vec.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        if norm == 0.0 {
            vec[0] = 1.0;
            return vec;
        }
        let scale = (1.0 / norm.sqrt()) as f32;
        for v in &mut vec {
            *v *= scale;
        }
        vec
    }

    fn final_chunk() -> ChatChunk {
        ChatChunk {
            text: "final answer".to_string(),
            cost: Some(CostedUsage {
                usage: Usage {
                    model: "test-model".to_string(),
                    total_tokens: 12,
                    ..Usage::default()
                },
                cost: Cost {
                    currency: "USD".to_string(),
                    amount: 0.01,
                },
            }),
            ..ChatChunk::default()
        }
    }

    fn contains_memory_context(messages: &[ChatMessage]) -> bool {
        messages.iter().any(|msg| {
            msg.role == "system"
                && msg.content.iter().any(|part| {
                    part.content.contains("Relevant memory context")
                        && part.content.contains("seed-memory")
                })
        })
    }

    #[tokio::test]
    async fn prepare_normalizes_system_and_user_messages() {
        let model = Arc::new(ScriptedModel::new(vec![]));
        let h = Harness::new(Options {
            model,
            memory: None,
            tools: vec![Arc::new(MarkerTool)],
            include_memory_tools: false,
        });

        // Empty history + user input (no system prompt) seeds a user message.
        let prepared = h
            .prepare(PrepareRequest {
                user_id: "user_123".to_string(),
                user_input: "How should backend import ditto-harness?".to_string(),
                ..PrepareRequest::default()
            })
            .await
            .expect("prepare");
        assert_eq!(prepared.messages.len(), 1);
        assert_eq!(prepared.messages[0].role, "user");
        assert_eq!(
            prepared.messages[0].content[0].content,
            "How should backend import ditto-harness?"
        );
        let names: Vec<&str> = prepared.tools.iter().map(|def| def.name.as_str()).collect();
        assert_eq!(names, ["marker"]);

        // A system prompt with an otherwise-empty history still seeds
        // user_input as the first user message — the model must see the
        // user's turn, not just the system message.
        let prepared = h
            .prepare(PrepareRequest {
                user_id: "user_123".to_string(),
                user_input: "How should backend import ditto-harness?".to_string(),
                system_prompt: "You are a concise assistant.".to_string(),
                ..PrepareRequest::default()
            })
            .await
            .expect("prepare");
        assert_eq!(
            prepared
                .messages
                .iter()
                .map(|m| m.role.as_str())
                .collect::<Vec<_>>(),
            ["system", "user"]
        );
        assert_eq!(
            prepared.messages[1].content[0].content,
            "How should backend import ditto-harness?"
        );

        // System prompt + existing history -> system prepended.
        let prepared = h
            .prepare(PrepareRequest {
                user_id: "user_123".to_string(),
                system_prompt: "You are a concise assistant.".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: vec![Content::text("hello")],
                    ..ChatMessage::default()
                }],
                ..PrepareRequest::default()
            })
            .await
            .expect("prepare");
        assert_eq!(
            prepared
                .messages
                .iter()
                .map(|m| m.role.as_str())
                .collect::<Vec<_>>(),
            ["system", "user"]
        );
    }

    #[tokio::test]
    async fn prepare_requires_user_id() {
        let h = Harness::new(Options {
            model: Arc::new(ScriptedModel::new(vec![])),
            memory: None,
            tools: vec![],
            include_memory_tools: false,
        });
        let err = h
            .prepare(PrepareRequest::default())
            .await
            .expect_err("prepare should fail");
        assert!(err.to_string().contains("user id is required"), "{err}");
    }

    /// A memory-less run returns the final text and collected costs.
    #[tokio::test]
    async fn run_without_memory_returns_text_and_costs() {
        let model = Arc::new(ScriptedModel::new(vec![final_chunk()]));
        let h = Harness::new(Options {
            model: model.clone(),
            memory: None,
            tools: vec![Arc::new(MarkerTool)],
            include_memory_tools: false,
        });
        let result = h
            .run(
                RunRequest {
                    prepare: PrepareRequest {
                        user_id: "user_123".to_string(),
                        user_input: "How should backend import ditto-harness?".to_string(),
                        system_prompt: "You are a concise assistant.".to_string(),
                        ..PrepareRequest::default()
                    },
                    ..RunRequest::default()
                },
                &NoopHandler,
            )
            .await
            .expect("run");
        assert_eq!(result.result.text, "final answer");
        assert_eq!(result.result.costs.len(), 1);
        assert_eq!(result.result.costs[0].usage.total_tokens, 12);
        assert!(result.saved_memory.is_none());
        let seen = model.seen_tools.lock().expect("lock");
        assert_eq!(*seen, ["marker"]);
    }

    /// End-to-end prepare, run, and save with memory context and tools.
    #[tokio::test]
    async fn harness_prepare_run_and_save() {
        let db = Db::open_memory().await.expect("open db");
        db.migrate().await.expect("migrate");
        let store = Arc::new(Store::new(StoreOptions {
            db: Arc::new(db),
            embedder: Arc::new(HashEmbedder),
            predictor: None,
            reranker: None,
        }));

        store
            .save_memory(SaveMemoryRequest {
                id: "seed-memory".to_string(),
                user_id: "chat-user".to_string(),
                session_id: "thread-a".to_string(),
                prompt: "Remember that chatv2 should import the harness.".to_string(),
                response: "The harness prepares memory context before running the agent loop."
                    .to_string(),
                summary: "chatv2 imports the harness for memory context.".to_string(),
                timestamp: Some(
                    Utc.with_ymd_and_hms(2026, 1, 1, 12, 0, 0)
                        .single()
                        .expect("timestamp"),
                ),
                ..SaveMemoryRequest::default()
            })
            .await
            .expect("save seed memory");

        let model = Arc::new(ScriptedModel::new(vec![final_chunk()]));
        let h = Harness::new(Options {
            model: model.clone(),
            memory: Some(Arc::clone(&store)),
            tools: vec![Arc::new(MarkerTool)],
            include_memory_tools: true,
        });

        let result = h
            .run(
                RunRequest {
                    prepare: PrepareRequest {
                        user_id: "chat-user".to_string(),
                        session_id: "thread-a".to_string(),
                        user_input: "How should chatv2 use memory context?".to_string(),
                        system_prompt: "You are a harness test.".to_string(),
                        long_term_limit: 1,
                        short_term_limit: 0,
                        ..PrepareRequest::default()
                    },
                    save_memory: true,
                    source: "chatv2_test".to_string(),
                    ..RunRequest::default()
                },
                &NoopHandler,
            )
            .await
            .expect("run");

        assert_eq!(result.result.text, "final answer");
        assert_eq!(result.result.costs.len(), 1);
        let saved = result.saved_memory.as_ref().expect("saved memory");
        assert_eq!(saved.source, "chatv2_test");
        assert_eq!(saved.seed_memories.len(), 1);
        assert_eq!(saved.seed_memories[0].pair_id, "seed-memory");
        let seen_tools = model.seen_tools.lock().expect("lock").clone();
        assert!(
            seen_tools.iter().any(|name| name == "marker")
                && seen_tools.iter().any(|name| name == "save_memory"),
            "model tools = {seen_tools:?}, want injected and memory tools"
        );
        let seen_messages = model.seen_messages.lock().expect("lock").clone();
        assert!(
            contains_memory_context(&seen_messages),
            "model did not receive memory context: {seen_messages:?}"
        );
    }

    #[test]
    fn memory_context_message_embeds_long_term_json_and_short_term() {
        // Long-term only: shortTerm key must be absent, longTerm embedded raw.
        let memories = PromptMemoryResponse {
            long_term: vec![Memory {
                id: "pair-1".to_string(),
                ..Memory::default()
            }],
            long_term_json: r#"{"memories":[{"pairID":"pair-1"}]}"#.to_string(),
            ..PromptMemoryResponse::default()
        };
        let msg = memory_context_message(&memories).expect("message");
        assert_eq!(msg.role, "system");
        let text = &msg.content[0].content;
        assert!(
            text.starts_with("Relevant memory context for this turn:\n"),
            "{text}"
        );
        let payload: Value = serde_json::from_str(
            text.strip_prefix("Relevant memory context for this turn:\n")
                .expect("prefix"),
        )
        .expect("payload json");
        assert_eq!(
            payload["longTerm"],
            serde_json::json!({"memories": [{"pairID": "pair-1"}]})
        );
        assert!(payload.get("shortTerm").is_none());
    }

    #[test]
    fn memory_context_message_empty_returns_none() {
        assert!(memory_context_message(&PromptMemoryResponse::default()).is_none());
    }

    #[test]
    fn insert_after_system_places_message_after_leading_system_block() {
        let system = ChatMessage {
            role: "system".to_string(),
            ..ChatMessage::default()
        };
        let user = ChatMessage {
            role: "user".to_string(),
            ..ChatMessage::default()
        };
        let ctx = ChatMessage {
            role: "system".to_string(),
            content: vec![Content::text("ctx")],
            ..ChatMessage::default()
        };

        // [system, user] -> [system, ctx, user]
        let out = insert_after_system(vec![system.clone(), user.clone()], ctx.clone());
        assert_eq!(
            out.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            ["system", "system", "user"]
        );
        assert_eq!(out[1].content, ctx.content);

        // [user] -> [user, ctx]: see the function rustdoc — a non-system
        // message at index 0 never gets the context prepended.
        let out = insert_after_system(vec![user.clone()], ctx.clone());
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].content, ctx.content);

        // [] -> [ctx]
        let out = insert_after_system(vec![], ctx.clone());
        assert_eq!(out.len(), 1);

        // [system] -> [system, ctx]
        let out = insert_after_system(vec![system], ctx.clone());
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].content, ctx.content);
    }

    #[test]
    fn normalize_messages_handles_blank_inputs() {
        // Blank system prompt is skipped; empty history + blank input -> empty.
        assert!(normalize_messages(&[], "   ", "  ").is_empty());
        // Existing history suppresses the user_input seeding.
        let history = vec![ChatMessage {
            role: "user".to_string(),
            content: vec![Content::text("hi")],
            ..ChatMessage::default()
        }];
        let out = normalize_messages(&history, "ignored", "");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content[0].content, "hi");
    }

    #[test]
    fn last_user_input_falls_back_to_user_input() {
        let history = vec![ChatMessage {
            role: "assistant".to_string(),
            content: vec![Content::text("reply")],
            ..ChatMessage::default()
        }];
        let out = last_user_input(&history, "fallback");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content, "fallback");
        assert!(last_user_input(&history, "").is_empty());
    }
}
