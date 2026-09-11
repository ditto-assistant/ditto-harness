// SPDX-License-Identifier: MIT
//! Model and embedder providers: Ollama plus OpenAI-compatible endpoints
//! (vLLM, OpenRouter, Chutes), bridged to the [`Model`]/[`Embedder`] traits via
//! rig-core.
//!
//! The bridge sits on rig's low-level `CompletionModel` trait (not its typed
//! agents) so the harness can pass dynamic tool definitions through
//! unchanged. Streaming note: `next_streaming` uses the trait's default
//! implementation (one aggregated chunk emitted once); rig's per-provider
//! streaming API could replace it later without changing the trait.

use std::sync::Arc;

use async_trait::async_trait;
use rig_core::client::{BearerAuth, CompletionClient};
use rig_core::completion::{
    AssistantContent, CompletionModel as RigCompletionModel, CompletionRequest,
    ToolDefinition as RigToolDefinition,
};
use rig_core::message::{
    Message as RigMessage, Text as RigText, ToolCall as RigToolCall,
    ToolFunction as RigToolFunction, ToolResult as RigToolResult,
    ToolResultContent as RigToolResultContent, UserContent as RigUserContent,
};
use rig_core::providers::{
    ollama as rig_ollama, openai as rig_openai, openrouter as rig_openrouter,
};
use rig_core::OneOrMany;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::types::{
    ChatChunk, ChatMessage, ContentType, Cost, CostedUsage, EmbedRequest, EmbedResponse, Embedder,
    Error, Model, Result, ToolCall, ToolDefinition, Usage,
};

/// Default Ollama endpoint.
pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";
/// OpenRouter OpenAI-compatible endpoint.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// Ditto identity required by OpenRouter's app-attribution contract.
pub const OPENROUTER_APP_REFERER: &str = "https://heyditto.ai";
/// Ditto display name required by OpenRouter's app-attribution contract.
pub const OPENROUTER_APP_TITLE: &str = "Ditto";
/// Default local chat model for the dream pipeline.
pub const DEFAULT_OLLAMA_CHAT_MODEL: &str = "gemma3:4b";
/// Default embedding model (768 dims).
pub const DEFAULT_EMBED_MODEL: &str = "embeddinggemma";
/// Embedding dimensions produced by [`DEFAULT_EMBED_MODEL`].
pub const DEFAULT_EMBED_DIMS: usize = 768;

/// Chat model configuration covering Ollama and OpenAI-compatible servers
/// (vLLM, OpenRouter, Chutes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatModelConfig {
    Ollama {
        base_url: String,
        model: String,
    },
    OpenAiCompat {
        base_url: String,
        /// Empty for servers that don't require auth (e.g. local vLLM).
        api_key: String,
        model: String,
    },
}

/// Optional sampling parameters applied to every request issued by a model
/// built via [`ChatModelConfig::build_with_params`]. `None` fields defer to
/// the provider's defaults.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ModelParams {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    /// Forwarded to the provider as the request `seed` (top-level, via rig's
    /// flattened additional_params). It reaches the provider, but on batched
    /// MoE backends (measured on OpenRouter and Chutes qwen3-32b) it does NOT
    /// yield run-to-run reproducibility even at temperature 0, so it is a
    /// best-effort hint, not a determinism guarantee. `None` omits it.
    pub seed: Option<u64>,
}

impl ChatModelConfig {
    /// Ollama config; empty `base_url` -> [`DEFAULT_OLLAMA_BASE_URL`].
    pub fn ollama(base_url: impl Into<String>, model: impl Into<String>) -> ChatModelConfig {
        let base_url = base_url.into();
        ChatModelConfig::Ollama {
            base_url: if base_url.is_empty() {
                DEFAULT_OLLAMA_BASE_URL.to_string()
            } else {
                base_url
            },
            model: model.into(),
        }
    }

    /// OpenRouter config ([`OPENROUTER_BASE_URL`]).
    pub fn openrouter(api_key: impl Into<String>, model: impl Into<String>) -> ChatModelConfig {
        ChatModelConfig::OpenAiCompat {
            base_url: OPENROUTER_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    /// vLLM (OpenAI-compatible, no API key) config.
    pub fn vllm(base_url: impl Into<String>, model: impl Into<String>) -> ChatModelConfig {
        ChatModelConfig::OpenAiCompat {
            base_url: base_url.into(),
            api_key: String::new(),
            model: model.into(),
        }
    }

    /// Builds a [`Model`] backed by rig-core for this configuration with
    /// default sampling parameters. See [`ChatModelConfig::build_with_params`].
    pub fn build(&self) -> Result<Arc<dyn Model>> {
        self.build_with_params(ModelParams::default())
    }

    /// Builds like [`ChatModelConfig::build_with_params`], additionally
    /// attaching `extra_headers` to every request the returned model issues.
    ///
    /// These are the model's DEFAULT headers, fixed when it is built. They are
    /// not chosen per [`Model::next`] call, so one model carries one set of
    /// header values for its whole life. Build a separate model per unit of
    /// work whose headers differ:
    ///
    /// ```no_run
    /// # use ditto_harness::models::{ChatModelConfig, ModelParams};
    /// # fn example(config: &ChatModelConfig, case_id: &str) -> ditto_harness::Result<()> {
    /// let case_model = config.build_with_params_and_headers(
    ///     ModelParams::default(),
    ///     &[("X-Ditto-Case-Id", case_id)],
    /// )?;
    /// # let _ = case_model;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Sharing ONE headered model across overlapping units is a correctness
    /// bug, not merely imprecise: a model built for case A stamps A's header on
    /// case B's calls too, and while A is still in flight a receiver that
    /// verifies the claim against the work actually in flight will accept it,
    /// filing B's calls under A. Models are independent — building one per case
    /// needs no lock around inference, and concurrent calls on distinct models
    /// do not serialize.
    ///
    /// Plain OpenAI-compatible endpoints only; the use case is attribution a
    /// receiver records verbatim (e.g. SN118's `X-Ditto-Case-Id`). OpenRouter
    /// and Ollama builds reject non-empty headers rather than silently
    /// dropping them, so a caller cannot believe attribution is flowing when it
    /// is not. An empty slice is exactly
    /// [`ChatModelConfig::build_with_params`] for every provider.
    pub fn build_with_params_and_headers(
        &self,
        params: ModelParams,
        extra_headers: &[(&str, &str)],
    ) -> Result<Arc<dyn Model>> {
        if extra_headers.is_empty() {
            return self.build_with_params(params);
        }
        let ChatModelConfig::OpenAiCompat {
            base_url,
            api_key,
            model,
        } = self
        else {
            return Err(Error::InvalidArgument(
                "extra headers are only supported for openai-compatible endpoints".into(),
            ));
        };
        if is_openrouter_base_url(base_url) {
            return Err(Error::InvalidArgument(
                "extra headers are only supported for openai-compatible endpoints".into(),
            ));
        }
        if base_url.is_empty() {
            return Err(Error::InvalidArgument(
                "openai-compatible base_url required".into(),
            ));
        }
        if model.is_empty() {
            return Err(Error::InvalidArgument(
                "openai-compatible model name required".into(),
            ));
        }
        // rig bundles its own reqwest (0.13); the workspace one (0.12) is a
        // different crate and does not satisfy rig's `HttpClientExt`. Build
        // the headered client through rig's re-export so the versions agree.
        let mut headers = rig_core::http_client::HeaderMap::new();
        for (name, value) in extra_headers {
            let header_name = http::HeaderName::from_bytes(name.as_bytes())
                .map_err(|err| Error::InvalidArgument(format!("header name {name:?}: {err}")))?;
            let header_value =
                rig_core::http_client::HeaderValue::from_str(value).map_err(|err| {
                    Error::InvalidArgument(format!("value for header {name:?}: {err}"))
                })?;
            headers.insert(header_name, header_value);
        }
        let http = rig_core::http_client::ReqwestClient::builder()
            .default_headers(headers)
            .build()
            .map_err(|err| Error::Model(format!("http client: {err}")))?;
        let client = rig_openai::CompletionsClient::builder()
            .api_key::<BearerAuth>(api_key.clone())
            .base_url(base_url)
            .http_client(http)
            .build()
            .map_err(|err| Error::Model(format!("openai-compatible client: {err}")))?;
        Ok(Arc::new(RigModel {
            inner: client.completion_model(model.clone()),
            provider: "openai-compat".to_string(),
            model: model.clone(),
            params,
        }))
    }

    /// Builds a [`Model`] backed by rig-core for this configuration. The
    /// returned model maps `ChatMessage`/`ToolDefinition` to the provider
    /// request, and provider tool calls / text / usage back into a
    /// [`ChatChunk`] with `cost`. `params` is applied to every request.
    pub fn build_with_params(&self, params: ModelParams) -> Result<Arc<dyn Model>> {
        match self {
            ChatModelConfig::Ollama { base_url, model } => {
                if model.is_empty() {
                    return Err(Error::InvalidArgument("ollama model name required".into()));
                }
                let client = rig_ollama::Client::builder()
                    .api_key::<rig_ollama::OllamaApiKey>("")
                    .base_url(base_url)
                    .build()
                    .map_err(|err| Error::Model(format!("ollama client: {err}")))?;
                Ok(Arc::new(RigModel {
                    inner: client.completion_model(model.clone()),
                    provider: "ollama".to_string(),
                    model: model.clone(),
                    params,
                }))
            }
            ChatModelConfig::OpenAiCompat {
                base_url,
                api_key,
                model,
            } => {
                if base_url.is_empty() {
                    return Err(Error::InvalidArgument(
                        "openai-compatible base_url required".into(),
                    ));
                }
                if model.is_empty() {
                    return Err(Error::InvalidArgument(
                        "openai-compatible model name required".into(),
                    ));
                }
                if is_openrouter_base_url(base_url) {
                    let client = build_openrouter_client(api_key, base_url)?;
                    return Ok(Arc::new(RigModel {
                        inner: client.completion_model(model.clone()),
                        provider: "openrouter".to_string(),
                        model: model.clone(),
                        params,
                    }));
                }
                let client = rig_openai::CompletionsClient::builder()
                    .api_key::<BearerAuth>(api_key.clone())
                    .base_url(base_url)
                    .build()
                    .map_err(|err| Error::Model(format!("openai-compatible client: {err}")))?;
                Ok(Arc::new(RigModel {
                    inner: client.completion_model(model.clone()),
                    provider: "openai-compat".to_string(),
                    model: model.clone(),
                    params,
                }))
            }
        }
    }
}

fn is_openrouter_base_url(base_url: &str) -> bool {
    let authority = base_url
        .split_once("://")
        .map_or(base_url, |(_, rest)| rest)
        .split('/')
        .next()
        .unwrap_or_default();
    let host = authority
        .rsplit('@')
        .next()
        .unwrap_or_default()
        .split(':')
        .next()
        .unwrap_or_default();
    host == "openrouter.ai" || host.ends_with(".openrouter.ai")
}

// Use rig's OpenRouter client so both attribution headers are attached to all
// completion requests, including provider retries and streamed requests.
fn build_openrouter_client(api_key: &str, base_url: &str) -> Result<rig_openrouter::Client> {
    rig_openrouter::Client::builder()
        .with_app_identity(OPENROUTER_APP_TITLE, OPENROUTER_APP_REFERER)
        .api_key::<BearerAuth>(api_key.to_string())
        .base_url(base_url)
        .build()
        .map_err(|err| Error::Model(format!("openrouter client: {err}")))
}

/// Bridge from rig's low-level `CompletionModel` to the harness [`Model`]
/// trait.
struct RigModel<M> {
    inner: M,
    provider: String,
    model: String,
    params: ModelParams,
}

#[async_trait]
impl<M> Model for RigModel<M>
where
    M: RigCompletionModel + Send + Sync + 'static,
{
    async fn next(&self, messages: &[ChatMessage], tools: &[ToolDefinition]) -> Result<ChatChunk> {
        let chat_history = to_rig_messages(messages)?;
        let chat_history = OneOrMany::many(chat_history)
            .map_err(|_| Error::InvalidArgument("at least one chat message is required".into()))?;
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history,
            documents: Vec::new(),
            tools: to_rig_tools(tools),
            temperature: self.params.temperature,
            max_tokens: self.params.max_tokens,
            tool_choice: None,
            additional_params: self.params.seed.map(|s| json!({ "seed": s })),
            output_schema: None,
        };
        let response = self
            .inner
            .completion(request)
            .await
            .map_err(|err| Error::Model(format!("{} completion: {err}", self.provider)))?;
        Ok(chunk_from_choice(
            &self.provider,
            &self.model,
            &response.choice,
            response.usage,
        ))
    }
}

/// Converts harness chat history into rig messages. Empty/system-only edge
/// cases are the caller's concern; messages that carry no usable content are
/// dropped (an all-dropped history yields an empty vec).
fn to_rig_messages(messages: &[ChatMessage]) -> Result<Vec<RigMessage>> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        match msg.role.as_str() {
            "system" => out.push(RigMessage::System {
                content: message_text(msg),
            }),
            "user" => {
                let mut parts: Vec<RigUserContent> = msg
                    .content
                    .iter()
                    .filter(|part| is_textual(part))
                    .map(|part| RigUserContent::Text(RigText::new(part.content.clone())))
                    .collect();
                if parts.is_empty() {
                    parts.push(RigUserContent::Text(RigText::new(String::new())));
                }
                // `parts` is never empty (a blank text part is pushed above).
                let content = OneOrMany::many(parts).expect("parts is non-empty");
                out.push(RigMessage::User { content });
            }
            "assistant" => {
                let mut parts: Vec<AssistantContent> = msg
                    .content
                    .iter()
                    .filter(|part| is_textual(part) && !part.content.is_empty())
                    .map(|part| AssistantContent::Text(RigText::new(part.content.clone())))
                    .collect();
                for tc in &msg.tool_calls {
                    parts.push(AssistantContent::ToolCall(RigToolCall::new(
                        tc.id.clone(),
                        RigToolFunction::new(tc.name.clone(), tc.args.clone()),
                    )));
                }
                if let Ok(content) = OneOrMany::many(parts) {
                    out.push(RigMessage::Assistant { id: None, content });
                }
            }
            "tool" => {
                let text = tool_result_text(msg);
                out.push(RigMessage::User {
                    content: OneOrMany::one(RigUserContent::ToolResult(RigToolResult {
                        id: msg.tool_call_id.clone(),
                        call_id: None,
                        content: OneOrMany::one(RigToolResultContent::Text(RigText::new(text))),
                    })),
                });
            }
            other => {
                return Err(Error::InvalidArgument(format!(
                    "unsupported chat message role {other:?}"
                )));
            }
        }
    }
    Ok(out)
}

/// True for parts that hold plain/markdown text (or untyped text content).
fn is_textual(part: &crate::types::Content) -> bool {
    matches!(
        part.content_type,
        None | Some(ContentType::Text) | Some(ContentType::Markdown)
    )
}

/// Joins the textual parts of a message with newlines.
fn message_text(msg: &ChatMessage) -> String {
    msg.content
        .iter()
        .filter(|part| is_textual(part) && !part.content.is_empty())
        .map(|part| part.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extracts the tool-result payload text from a `role == "tool"` message.
fn tool_result_text(msg: &ChatMessage) -> String {
    for part in &msg.content {
        if let Some(resp) = &part.tool_call_response {
            if let Value::String(s) = &resp.output {
                return s.clone();
            }
            if !resp.output.is_null() {
                return resp.output.to_string();
            }
            if !resp.error.is_empty() {
                return json!({ "error": resp.error }).to_string();
            }
        }
        if !part.content.is_empty() {
            return part.content.clone();
        }
    }
    String::new()
}

/// Maps harness tool definitions to rig's; a `Null` input schema becomes an
/// empty object schema (providers reject `null` parameter schemas).
fn to_rig_tools(tools: &[ToolDefinition]) -> Vec<RigToolDefinition> {
    tools
        .iter()
        .map(|tool| RigToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: if tool.input_schema.is_null() {
                json!({ "type": "object", "properties": {} })
            } else {
                tool.input_schema.clone()
            },
        })
        .collect()
}

/// Aggregates a rig completion choice + usage into a harness [`ChatChunk`].
/// Text parts concatenate; the first tool call wins (the harness loop executes
/// one tool per turn). Token-only usage is reported with an empty `Cost`.
fn chunk_from_choice(
    provider: &str,
    model: &str,
    choice: &OneOrMany<AssistantContent>,
    usage: rig_core::completion::Usage,
) -> ChatChunk {
    let mut text = String::new();
    let mut tool_call: Option<ToolCall> = None;
    for item in choice.iter() {
        match item {
            AssistantContent::Text(t) => text.push_str(&t.text),
            AssistantContent::ToolCall(tc) if tool_call.is_none() => {
                let id = if tc.id.is_empty() {
                    format!("call_{}", uuid::Uuid::new_v4())
                } else {
                    tc.id.clone()
                };
                tool_call = Some(ToolCall {
                    id,
                    name: tc.function.name.clone(),
                    args: tc.function.arguments.clone(),
                });
            }
            _ => {}
        }
    }
    let input_tokens = usage.input_tokens as i64;
    let output_tokens = usage.output_tokens as i64;
    let total_tokens = if usage.total_tokens > 0 {
        usage.total_tokens as i64
    } else {
        input_tokens + output_tokens
    };
    ChatChunk {
        text,
        tool_call,
        cost: Some(CostedUsage {
            usage: Usage {
                provider: provider.to_string(),
                model: model.to_string(),
                input_tokens,
                output_tokens,
                total_tokens,
            },
            cost: Cost::default(),
        }),
        metadata: None,
    }
}

/// Ollama embeddings client (`/api/embed`) producing
/// [`DEFAULT_EMBED_DIMS`]-dim vectors with [`DEFAULT_EMBED_MODEL`] by
/// default.
#[derive(Debug, Clone)]
pub struct OllamaEmbedder {
    pub base_url: String,
    pub model: String,
    pub dims: usize,
    /// Shared HTTP client (connection pool + request timeout), built once in
    /// [`OllamaEmbedder::new`] and reused across `embed` calls.
    client: reqwest::Client,
}

/// Request timeout for embedding calls; large batches on a cold model can be
/// slow, but a hung server must not stall the caller forever.
const EMBED_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

impl Default for OllamaEmbedder {
    fn default() -> OllamaEmbedder {
        OllamaEmbedder::new(DEFAULT_OLLAMA_BASE_URL)
    }
}

impl OllamaEmbedder {
    /// Embedder with the default model ("embeddinggemma", 768 dims).
    pub fn new(base_url: impl Into<String>) -> OllamaEmbedder {
        OllamaEmbedder {
            base_url: base_url.into(),
            model: DEFAULT_EMBED_MODEL.to_string(),
            dims: DEFAULT_EMBED_DIMS,
            client: reqwest::Client::builder()
                .timeout(EMBED_REQUEST_TIMEOUT)
                .build()
                // Builder failure is a startup programming/TLS error; falling
                // back to a default client would silently drop the timeout.
                .expect("build embedder http client"),
        }
    }
}

#[derive(Serialize)]
struct OllamaEmbedRequestBody<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct OllamaEmbedResponseBody {
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    prompt_eval_count: Option<i64>,
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    /// Embeds each text via Ollama (batched in one `/api/embed` call),
    /// validating that returned vectors have `dims` dimensions (error
    /// otherwise). Fills `EmbedResponse::cost` usage tokens when the API
    /// reports them.
    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse> {
        if req.texts.is_empty() {
            return Ok(EmbedResponse::default());
        }
        let base = self.base_url.trim_end_matches('/');
        let url = format!("{base}/api/embed");
        let body = OllamaEmbedRequestBody {
            model: &self.model,
            input: &req.texts,
        };
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|err| {
                Error::Embedding(format!(
                    "ollama embed request to {url} failed (is ollama running?): {err}"
                ))
            })?;
        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            return Err(Error::Embedding(format!(
                "ollama embed request to {url} returned {status}: {detail}"
            )));
        }
        let parsed: OllamaEmbedResponseBody = response.json().await.map_err(|err| {
            Error::Embedding(format!("ollama embed response decode failed: {err}"))
        })?;
        if parsed.embeddings.len() != req.texts.len() {
            return Err(Error::Embedding(format!(
                "ollama embed returned {} embeddings for {} texts",
                parsed.embeddings.len(),
                req.texts.len()
            )));
        }
        for emb in &parsed.embeddings {
            if emb.len() != self.dims {
                return Err(Error::Embedding(format!(
                    "ollama embed returned {}-dim vector, expected {} (model {})",
                    emb.len(),
                    self.dims,
                    self.model
                )));
            }
        }
        let cost = parsed.prompt_eval_count.map(|tokens| CostedUsage {
            usage: Usage {
                provider: "ollama".to_string(),
                model: self.model.clone(),
                input_tokens: tokens,
                output_tokens: 0,
                total_tokens: tokens,
            },
            cost: Cost::default(),
        });
        Ok(EmbedResponse {
            embeddings: parsed.embeddings,
            cost,
            metadata: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Content, ToolCallResponse};

    fn user(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: vec![Content::text(text)],
            ..ChatMessage::default()
        }
    }

    /// One-shot OpenAI-compatible stub: accepts a single connection, replies
    /// with a canned `chat.completion`, and reports the value of
    /// `x-ditto-case-id` it saw (None when the header never arrived).
    async fn one_shot_openai_stub() -> (String, tokio::sync::oneshot::Receiver<Option<String>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let header_end = loop {
                let n = sock.read(&mut chunk).await.unwrap();
                if n == 0 {
                    break buf.len();
                }
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
            let content_length = head
                .lines()
                .find_map(|l| {
                    let (name, value) = l.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0);
            while buf.len() < header_end + content_length {
                let n = sock.read(&mut chunk).await.unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            let case = head.lines().find_map(|l| {
                let (name, value) = l.split_once(':')?;
                name.eq_ignore_ascii_case("x-ditto-case-id")
                    .then(|| value.trim().to_string())
            });
            let body = serde_json::json!({
                "id": "cmpl-stub", "object": "chat.completion", "created": 0,
                "model": "stub-model",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })
            .to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).await.unwrap();
            let _ = sock.shutdown().await;
            let _ = tx.send(case);
        });
        (base_url, rx)
    }

    async fn barrier_openai_stub(
        parties: usize,
    ) -> (
        String,
        std::sync::Arc<tokio::sync::Mutex<Vec<Option<String>>>>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let seen = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(parties));
        let collected = seen.clone();
        tokio::spawn(async move {
            for _ in 0..parties {
                let (mut sock, _) = listener.accept().await.unwrap();
                let seen = collected.clone();
                let barrier = barrier.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let header_end = loop {
                        let n = sock.read(&mut chunk).await.unwrap();
                        if n == 0 {
                            break buf.len();
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break pos + 4;
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                    let content_length = head
                        .lines()
                        .find_map(|l| {
                            let (name, value) = l.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())?
                        })
                        .unwrap_or(0);
                    while buf.len() < header_end + content_length {
                        let n = sock.read(&mut chunk).await.unwrap();
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let case = head.lines().find_map(|l| {
                        let (name, value) = l.split_once(':')?;
                        name.eq_ignore_ascii_case("x-ditto-case-id")
                            .then(|| value.trim().to_string())
                    });
                    seen.lock().await.push(case);
                    barrier.wait().await;
                    let body = serde_json::json!({
                        "id": "cmpl-stub", "object": "chat.completion", "created": 0,
                        "model": "stub-model",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    })
                    .to_string();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    sock.write_all(resp.as_bytes()).await.unwrap();
                    let _ = sock.shutdown().await;
                });
            }
        });
        (base_url, seen)
    }

    #[tokio::test]
    async fn per_case_models_attribute_overlapping_calls() {
        let (base_url, seen) = barrier_openai_stub(2).await;
        let config = ChatModelConfig::vllm(base_url, "stub-model");
        let model_a = config
            .build_with_params_and_headers(ModelParams::default(), &[("X-Ditto-Case-Id", "case-A")])
            .expect("build case-A model");
        let model_b = config
            .build_with_params_and_headers(ModelParams::default(), &[("X-Ditto-Case-Id", "case-B")])
            .expect("build case-B model");

        let call_a = tokio::spawn(async move { model_a.next(&[user("a")], &[]).await });
        let call_b = tokio::spawn(async move { model_b.next(&[user("b")], &[]).await });
        let (ra, rb) = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            futures::future::join(call_a, call_b),
        )
        .await
        .expect("overlapping calls did not both reach the server");

        assert_eq!(ra.unwrap().expect("case-A call").text, "ok");
        assert_eq!(rb.unwrap().expect("case-B call").text, "ok");
        let mut got: Vec<String> = seen
            .lock()
            .await
            .iter()
            .map(|c| c.clone().unwrap_or_default())
            .collect();
        got.sort();
        assert_eq!(got, vec!["case-A".to_string(), "case-B".to_string()]);
    }

    #[tokio::test]
    async fn one_shared_model_stamps_one_case_on_every_call() {
        let (base_url, seen) = barrier_openai_stub(2).await;
        let shared = ChatModelConfig::vllm(base_url, "stub-model")
            .build_with_params_and_headers(ModelParams::default(), &[("X-Ditto-Case-Id", "case-A")])
            .expect("build shared model");

        let serving_b = shared.clone();
        let call_a = tokio::spawn(async move { shared.next(&[user("a")], &[]).await });
        let call_b = tokio::spawn(async move { serving_b.next(&[user("b")], &[]).await });
        let (ra, rb) = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            futures::future::join(call_a, call_b),
        )
        .await
        .expect("overlapping calls did not both reach the server");
        assert_eq!(ra.unwrap().expect("case-A call").text, "ok");
        assert_eq!(rb.unwrap().expect("shared-model call").text, "ok");

        let got: Vec<String> = seen
            .lock()
            .await
            .iter()
            .map(|c| c.clone().unwrap_or_default())
            .collect();
        assert_eq!(got, vec!["case-A".to_string(), "case-A".to_string()]);
    }

    #[tokio::test]
    async fn extra_headers_reach_the_wire() {
        let (base_url, saw) = one_shot_openai_stub().await;
        let model = ChatModelConfig::vllm(base_url, "stub-model")
            .build_with_params_and_headers(
                ModelParams::default(),
                &[("X-Ditto-Case-Id", "case-123")],
            )
            .expect("build headered model");
        let chunk = model.next(&[user("hi")], &[]).await.expect("chat call");
        assert_eq!(chunk.text, "ok");
        assert_eq!(saw.await.unwrap().as_deref(), Some("case-123"));
    }

    #[tokio::test]
    async fn plain_build_sends_no_attribution_header() {
        let (base_url, saw) = one_shot_openai_stub().await;
        let model = ChatModelConfig::vllm(base_url, "stub-model")
            .build_with_params(ModelParams::default())
            .expect("build plain model");
        let chunk = model.next(&[user("hi")], &[]).await.expect("chat call");
        assert_eq!(chunk.text, "ok");
        assert_eq!(saw.await.unwrap(), None);
    }

    #[test]
    fn extra_headers_rejected_off_openai_compat() {
        let headers = [("X-Ditto-Case-Id", "case-123")];
        let err = ChatModelConfig::ollama("", "gemma3:4b")
            .build_with_params_and_headers(ModelParams::default(), &headers)
            .err()
            .expect("ollama must reject extra headers");
        assert!(err.to_string().contains("openai-compatible"));
        let err = ChatModelConfig::openrouter("k", "meta-llama/llama-3-8b")
            .build_with_params_and_headers(ModelParams::default(), &headers)
            .err()
            .expect("openrouter must reject extra headers");
        assert!(err.to_string().contains("openai-compatible"));
        let err = ChatModelConfig::vllm("http://127.0.0.1:1", "m")
            .build_with_params_and_headers(ModelParams::default(), &[("bad name", "v")])
            .err()
            .expect("invalid header name must be rejected");
        assert!(err.to_string().contains("header name"));
    }

    /// Chat model for the ollama-gated integration tests. Defaults to
    /// [`DEFAULT_OLLAMA_CHAT_MODEL`]; override with `DITTO_HARNESS_OLLAMA_MODEL`
    /// when a different model is pulled locally (e.g. `gemma4:e4b`).
    pub(crate) fn ollama_test_chat_model() -> String {
        std::env::var("DITTO_HARNESS_OLLAMA_MODEL")
            .unwrap_or_else(|_| DEFAULT_OLLAMA_CHAT_MODEL.to_string())
    }

    #[test]
    fn config_constructors_apply_defaults() {
        assert_eq!(
            ChatModelConfig::ollama("", "gemma3:4b"),
            ChatModelConfig::Ollama {
                base_url: DEFAULT_OLLAMA_BASE_URL.into(),
                model: "gemma3:4b".into()
            }
        );
        assert_eq!(
            ChatModelConfig::openrouter("k", "meta-llama/llama-3-8b"),
            ChatModelConfig::OpenAiCompat {
                base_url: OPENROUTER_BASE_URL.into(),
                api_key: "k".into(),
                model: "meta-llama/llama-3-8b".into()
            }
        );
        assert_eq!(
            ChatModelConfig::vllm("http://localhost:8000/v1", "qwen"),
            ChatModelConfig::OpenAiCompat {
                base_url: "http://localhost:8000/v1".into(),
                api_key: String::new(),
                model: "qwen".into()
            }
        );
    }

    #[test]
    fn build_constructs_models_offline() {
        ChatModelConfig::ollama("", DEFAULT_OLLAMA_CHAT_MODEL)
            .build()
            .expect("ollama model builds without network");
        ChatModelConfig::openrouter("test-key", "openai/gpt-4o-mini")
            .build()
            .expect("openrouter model builds without network");
        ChatModelConfig::vllm("http://localhost:8000/v1", "qwen")
            .build()
            .expect("vllm model builds without network");
        assert!(matches!(
            ChatModelConfig::ollama("", "").build(),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            ChatModelConfig::vllm("", "qwen").build(),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn openrouter_client_has_app_attribution_headers() {
        let client = build_openrouter_client("test-key", OPENROUTER_BASE_URL)
            .expect("openrouter client builds");
        assert_eq!(
            client.headers().get("http-referer").unwrap(),
            OPENROUTER_APP_REFERER
        );
        assert_eq!(
            client.headers().get("x-openrouter-title").unwrap(),
            OPENROUTER_APP_TITLE
        );
    }

    #[test]
    fn recognizes_default_and_regional_openrouter_hosts() {
        assert!(is_openrouter_base_url(OPENROUTER_BASE_URL));
        assert!(is_openrouter_base_url("https://eu.openrouter.ai/api/v1/"));
        assert!(!is_openrouter_base_url(
            "https://openrouter.ai.example/api/v1"
        ));
        assert!(!is_openrouter_base_url("http://localhost:8000/v1"));
    }

    #[test]
    fn message_conversion_maps_roles() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: vec![Content::text("be helpful")],
                ..ChatMessage::default()
            },
            user("hello"),
            ChatMessage {
                role: "assistant".into(),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "search_memories".into(),
                    args: json!({"queries": ["x"]}),
                }],
                ..ChatMessage::default()
            },
            ChatMessage {
                role: "tool".into(),
                tool_call_id: "call_1".into(),
                content: vec![Content {
                    content_type: Some(ContentType::ToolResult),
                    tool_call_response: Some(ToolCallResponse {
                        id: "call_1".into(),
                        name: "search_memories".into(),
                        output: json!({"memories": []}),
                        error: String::new(),
                    }),
                    ..Content::default()
                }],
                ..ChatMessage::default()
            },
        ];
        let rig_messages = to_rig_messages(&messages).expect("convert");
        assert_eq!(rig_messages.len(), 4);
        assert!(
            matches!(&rig_messages[0], RigMessage::System { content } if content == "be helpful")
        );
        assert!(matches!(&rig_messages[1], RigMessage::User { .. }));
        match &rig_messages[2] {
            RigMessage::Assistant { content, .. } => match content.first() {
                AssistantContent::ToolCall(tc) => {
                    assert_eq!(tc.id, "call_1");
                    assert_eq!(tc.function.name, "search_memories");
                }
                other => panic!("expected tool call, got {other:?}"),
            },
            other => panic!("expected assistant message, got {other:?}"),
        }
        match &rig_messages[3] {
            RigMessage::User { content } => match content.first() {
                RigUserContent::ToolResult(tr) => {
                    assert_eq!(tr.id, "call_1");
                    match tr.content.first() {
                        RigToolResultContent::Text(t) => {
                            assert_eq!(t.text, json!({"memories": []}).to_string());
                        }
                        other => panic!("expected text tool result, got {other:?}"),
                    }
                }
                other => panic!("expected tool result, got {other:?}"),
            },
            other => panic!("expected tool message as user, got {other:?}"),
        }
        assert!(to_rig_messages(&[ChatMessage {
            role: "robot".into(),
            ..ChatMessage::default()
        }])
        .is_err());
    }

    #[test]
    fn tool_schema_null_becomes_empty_object() {
        let tools = to_rig_tools(&[ToolDefinition {
            name: "t".into(),
            description: "d".into(),
            input_schema: Value::Null,
        }]);
        assert_eq!(
            tools[0].parameters,
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn chunk_aggregation_maps_text_tool_call_and_usage() {
        let choice = OneOrMany::many(vec![
            AssistantContent::Text(RigText::new("hel")),
            AssistantContent::Text(RigText::new("lo")),
            AssistantContent::ToolCall(RigToolCall::new(
                String::new(),
                RigToolFunction::new("fetch_memories".into(), json!({"pairIds": ["p1"]})),
            )),
        ])
        .expect("choice");
        let mut usage = rig_core::completion::Usage::new();
        usage.input_tokens = 10;
        usage.output_tokens = 5;
        let chunk = chunk_from_choice("ollama", "gemma3:4b", &choice, usage);
        assert_eq!(chunk.text, "hello");
        let tc = chunk.tool_call.expect("tool call");
        assert_eq!(tc.name, "fetch_memories");
        assert!(!tc.id.is_empty(), "empty provider id must be synthesized");
        let cost = chunk.cost.expect("cost");
        assert_eq!(cost.usage.input_tokens, 10);
        assert_eq!(cost.usage.output_tokens, 5);
        assert_eq!(cost.usage.total_tokens, 15);
        assert_eq!(cost.usage.provider, "ollama");
    }

    /// Real-ollama integration test: requires `ollama serve` with a chat
    /// model (`DITTO_HARNESS_OLLAMA_MODEL`, default `gemma3:4b`) +
    /// `embeddinggemma` pulled. Gated behind `DITTO_HARNESS_OLLAMA=1` so
    /// plain `cargo test` passes offline.
    #[tokio::test]
    async fn ollama_chat_and_embed_integration() {
        if std::env::var("DITTO_HARNESS_OLLAMA").as_deref() != Ok("1") {
            return;
        }
        let model = ChatModelConfig::ollama("", ollama_test_chat_model())
            .build()
            .expect("build model");
        let chunk = model
            .next(&[user("Reply with the single word: pong")], &[])
            .await
            .expect("ollama chat");
        assert!(!chunk.text.is_empty());

        let embedder = OllamaEmbedder::default();
        let resp = embedder
            .embed(EmbedRequest {
                texts: vec!["hello world".into(), "goodbye".into()],
            })
            .await
            .expect("ollama embed");
        assert_eq!(resp.embeddings.len(), 2);
        assert_eq!(resp.embeddings[0].len(), DEFAULT_EMBED_DIMS);
    }
}
