//! Memory tools exposed to the agent loop (and MCP hosts).
//! Port of Go `pkg/memory/tools.go`.
//!
//! Tool surface (names, descriptions, and JSON arg shapes match Go exactly):
//! - `save_memory`: args `{prompt, response, summary, sessionId,
//!   subjects: [{text, description, key}]}`; saves with source
//!   "agent_tool"; returns `{"memory": <Memory JSON>}`.
//! - `search_memories`: args `{queries: [string] (required), topK,
//!   sessionId}`; returns `{"memories": [SlimMemory previews]}`.
//! - `search_subjects`: args `{queries (required), topK}`; returns
//!   `{"subjects": [Subject JSON]}`.
//! - `search_memories_in_subjects`: args `{subject_id, queries (required),
//!   topK}`; returns `{"memories": [SlimMemory previews]}`.
//! - `fetch_memories`: args `{pairIds: [string] (required)}`; returns
//!   `{"memories": [SlimMemory truncated]}`.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{
    slim_previews, slim_truncated, FetchMemoriesRequest, SaveMemoryRequest,
    SearchMemoriesInSubjectsRequest, SearchMemoriesRequest, SearchSubjectsRequest, Store,
    SubjectInput, SubjectMemoryQuery,
};
use crate::types::{Result, Tool, ToolDefinition};

/// Options for building memory tools (Go: `memory.ToolOptions`).
#[derive(Clone)]
pub struct ToolOptions {
    pub store: Arc<Store>,
    pub user_id: String,
    pub kg_id: String,
    /// `0` -> [`super::DEFAULT_PREVIEW_LEN`].
    pub preview_len: usize,
    /// `0` -> [`super::DEFAULT_FETCH_MAX_BYTES`].
    pub fetch_max_bytes: usize,
}

impl ToolOptions {
    /// Options with default preview/fetch budgets.
    pub fn new(
        store: Arc<Store>,
        user_id: impl Into<String>,
        kg_id: impl Into<String>,
    ) -> ToolOptions {
        ToolOptions {
            store,
            user_id: user_id.into(),
            kg_id: kg_id.into(),
            preview_len: 0,
            fetch_max_bytes: 0,
        }
    }
}

/// All five memory tools with default budgets (Go: `memory.Tools`).
pub fn memory_tools(store: Arc<Store>, user_id: &str, kg_id: &str) -> Vec<Box<dyn Tool>> {
    memory_tools_with(ToolOptions::new(store, user_id, kg_id))
}

/// All five memory tools with explicit options.
pub fn memory_tools_with(opts: ToolOptions) -> Vec<Box<dyn Tool>> {
    vec![
        save_memory_tool(&opts),
        search_memories_tool(&opts),
        search_subjects_tool(&opts),
        search_memories_in_subjects_tool(&opts),
        fetch_memories_tool(&opts),
    ]
}

/// `save_memory` (Go: `SaveMemoryTool`).
pub fn save_memory_tool(opts: &ToolOptions) -> Box<dyn Tool> {
    Box::new(SaveMemoryTool { opts: opts.clone() })
}

/// `search_memories` (Go: `SearchMemoriesTool`).
pub fn search_memories_tool(opts: &ToolOptions) -> Box<dyn Tool> {
    Box::new(SearchMemoriesTool { opts: opts.clone() })
}

/// `search_subjects` (Go: `SearchSubjectsTool`).
pub fn search_subjects_tool(opts: &ToolOptions) -> Box<dyn Tool> {
    Box::new(SearchSubjectsTool { opts: opts.clone() })
}

/// `search_memories_in_subjects` (Go: `SearchMemoriesInSubjectsTool`).
pub fn search_memories_in_subjects_tool(opts: &ToolOptions) -> Box<dyn Tool> {
    Box::new(SearchMemoriesInSubjectsTool { opts: opts.clone() })
}

/// `fetch_memories` (Go: `FetchMemoriesTool`).
pub fn fetch_memories_tool(opts: &ToolOptions) -> Box<dyn Tool> {
    Box::new(FetchMemoriesTool { opts: opts.clone() })
}

impl ToolOptions {
    fn preview_len(&self) -> usize {
        if self.preview_len > 0 {
            return self.preview_len;
        }
        super::DEFAULT_PREVIEW_LEN
    }

    fn fetch_max_bytes(&self) -> usize {
        if self.fetch_max_bytes > 0 {
            return self.fetch_max_bytes;
        }
        super::DEFAULT_FETCH_MAX_BYTES
    }
}

/// Non-negative `topK` (Go ints <= 0 fall back to the store default of 8).
fn limit_from_top_k(top_k: i64) -> usize {
    top_k.max(0) as usize
}

struct SaveMemoryTool {
    opts: ToolOptions,
}

#[async_trait]
impl Tool for SaveMemoryTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "save_memory".to_string(),
            description: "Save a durable memory for the current user.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "response": {"type": "string"},
                    "summary": {"type": "string"},
                    "sessionId": {"type": "string"},
                    "subjects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "description": {"type": "string"},
                                "key": {"type": "boolean"}
                            }
                        }
                    }
                }
            }),
        }
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        #[derive(Debug, Default, Deserialize)]
        struct Args {
            #[serde(default)]
            prompt: String,
            #[serde(default)]
            response: String,
            #[serde(default)]
            summary: String,
            #[serde(default, rename = "sessionId")]
            session_id: String,
            #[serde(default)]
            subjects: Vec<SubjectInput>,
        }
        let args: Args = serde_json::from_value(args)?;
        let mem = self
            .opts
            .store
            .save_memory(SaveMemoryRequest {
                user_id: self.opts.user_id.clone(),
                kg_id: self.opts.kg_id.clone(),
                session_id: args.session_id,
                prompt: args.prompt,
                response: args.response,
                summary: args.summary,
                subjects: args.subjects,
                source: "agent_tool".to_string(),
                ..SaveMemoryRequest::default()
            })
            .await?;
        Ok(json!({ "memory": mem }))
    }
}

struct SearchMemoriesTool {
    opts: ToolOptions,
}

#[async_trait]
impl Tool for SearchMemoriesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_memories".to_string(),
            description: "Search past memories and return compact memory objects. Use \
                          fetch_memories for selected IDs that need full text."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["queries"],
                "properties": {
                    "queries": {"type": "array", "items": {"type": "string"}},
                    "topK": {"type": "integer"},
                    "sessionId": {"type": "string"}
                }
            }),
        }
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        #[derive(Debug, Default, Deserialize)]
        struct Args {
            #[serde(default)]
            queries: Vec<String>,
            #[serde(default, rename = "topK")]
            top_k: i64,
            #[serde(default, rename = "sessionId")]
            session_id: String,
        }
        let args: Args = serde_json::from_value(args)?;
        let memories = self
            .opts
            .store
            .search_memories(SearchMemoriesRequest {
                user_id: self.opts.user_id.clone(),
                kg_id: self.opts.kg_id.clone(),
                session_id: args.session_id,
                queries: args.queries,
                limit: limit_from_top_k(args.top_k),
                ..SearchMemoriesRequest::default()
            })
            .await?;
        Ok(json!({ "memories": slim_previews(&memories, self.opts.preview_len()) }))
    }
}

struct SearchSubjectsTool {
    opts: ToolOptions,
}

#[async_trait]
impl Tool for SearchSubjectsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_subjects".to_string(),
            description: "Search the user's subject graph and return subject objects with ids \
                          for subject-scoped memory search."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["queries"],
                "properties": {
                    "queries": {"type": "array", "items": {"type": "string"}},
                    "topK": {"type": "integer"}
                }
            }),
        }
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        #[derive(Debug, Default, Deserialize)]
        struct Args {
            #[serde(default)]
            queries: Vec<String>,
            #[serde(default, rename = "topK")]
            top_k: i64,
        }
        let args: Args = serde_json::from_value(args)?;
        let subjects = self
            .opts
            .store
            .search_subjects(SearchSubjectsRequest {
                user_id: self.opts.user_id.clone(),
                kg_id: self.opts.kg_id.clone(),
                queries: args.queries,
                limit: limit_from_top_k(args.top_k),
                ..SearchSubjectsRequest::default()
            })
            .await?;
        Ok(json!({ "subjects": subjects }))
    }
}

struct SearchMemoriesInSubjectsTool {
    opts: ToolOptions,
}

#[async_trait]
impl Tool for SearchMemoriesInSubjectsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_memories_in_subjects".to_string(),
            description: "Search memories inside one or more subject IDs.".to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["queries"],
                "properties": {
                    "subject_id": {"type": "string"},
                    "queries": {"type": "array", "items": {"type": "string"}},
                    "topK": {"type": "integer"}
                }
            }),
        }
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        #[derive(Debug, Default, Deserialize)]
        struct Args {
            #[serde(default)]
            subject_id: String,
            #[serde(default)]
            queries: Vec<String>,
            #[serde(default, rename = "topK")]
            top_k: i64,
        }
        let args: Args = serde_json::from_value(args)?;
        let queries = args
            .queries
            .into_iter()
            .map(|query| SubjectMemoryQuery {
                subject_id: args.subject_id.clone(),
                query,
            })
            .collect();
        let memories = self
            .opts
            .store
            .search_memories_in_subjects(SearchMemoriesInSubjectsRequest {
                user_id: self.opts.user_id.clone(),
                queries,
                limit: limit_from_top_k(args.top_k),
                ..SearchMemoriesInSubjectsRequest::default()
            })
            .await?;
        Ok(json!({ "memories": slim_previews(&memories, self.opts.preview_len()) }))
    }
}

struct FetchMemoriesTool {
    opts: ToolOptions,
}

#[async_trait]
impl Tool for FetchMemoriesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fetch_memories".to_string(),
            description: "Fetch full memory content for selected memory pair IDs.".to_string(),
            input_schema: json!({
                "type": "object",
                "required": ["pairIds"],
                "properties": {
                    "pairIds": {"type": "array", "items": {"type": "string"}}
                }
            }),
        }
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        #[derive(Debug, Default, Deserialize)]
        struct Args {
            #[serde(default, rename = "pairIds")]
            pair_ids: Vec<String>,
        }
        let args: Args = serde_json::from_value(args)?;
        let memories = self
            .opts
            .store
            .fetch_memories(FetchMemoriesRequest {
                user_id: self.opts.user_id.clone(),
                pair_ids: args.pair_ids,
            })
            .await?;
        Ok(json!({ "memories": slim_truncated(&memories, self.opts.fetch_max_bytes()) }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::super::test_support::new_test_store;
    use super::*;

    /// Port of Go `TestMemoryToolsExposeExpectedDefinitions`.
    #[tokio::test]
    async fn memory_tools_expose_expected_definitions() {
        let store = Arc::new(new_test_store().await);
        let tools = memory_tools(store, "u", "kg");
        let names: Vec<String> = tools
            .iter()
            .map(|tool| {
                let def = tool.definition();
                assert!(
                    def.input_schema.is_object(),
                    "{} has invalid schema {:?}",
                    def.name,
                    def.input_schema
                );
                def.name
            })
            .collect();
        assert_eq!(
            names,
            [
                "save_memory",
                "search_memories",
                "search_subjects",
                "search_memories_in_subjects",
                "fetch_memories",
            ]
        );
    }

    /// Port of Go `TestMemoryToolsCallThroughStore`.
    #[tokio::test]
    async fn memory_tools_call_through_store() {
        let store = Arc::new(new_test_store().await);
        let tools = memory_tools(store, "tool-user", "");

        let save = &tools[0];
        let saved = save
            .execute(json!({
                "prompt": "Remember the importable tool adapter",
                "response": "It should call through the store",
                "summary": "Tool adapter works",
                "subjects": [{"text": "Harness tools"}]
            }))
            .await
            .expect("save tool");
        assert!(
            saved.get("memory").is_some(),
            "save tool returned empty output: {saved}"
        );

        let search = &tools[1];
        let found = search
            .execute(json!({"queries": ["importable tool adapter"], "topK": 5}))
            .await
            .expect("search tool");
        let memories = found["memories"].as_array().expect("memories array");
        assert_eq!(memories.len(), 1, "search output: {found}");
        assert!(
            memories[0].get("preview").is_some()
                && memories[0].get("user").is_none()
                && memories[0].get("ditto").is_none(),
            "search should return slim previews only: {:?}",
            memories[0]
        );
        let pair_id = memories[0]["id"].as_str().expect("memory id");

        let fetch = &tools[4];
        let fetched = fetch
            .execute(json!({ "pairIds": [pair_id] }))
            .await
            .expect("fetch tool");
        let memories = fetched["memories"].as_array().expect("memories array");
        assert_eq!(memories.len(), 1, "fetch output: {fetched}");
        assert!(
            memories[0].get("user").is_some()
                && memories[0].get("ditto").is_some()
                && memories[0].get("preview").is_none(),
            "fetch should return truncated full slim memory: {:?}",
            memories[0]
        );

        let subjects_tool = &tools[2];
        let subjects = subjects_tool
            .execute(json!({"queries": ["harness tools"], "topK": 5}))
            .await
            .expect("subjects tool");
        let subject_list = subjects["subjects"].as_array().expect("subjects array");
        assert_eq!(subject_list.len(), 1, "subjects output: {subjects}");
        let subject_id = subject_list[0]["id"].as_str().expect("subject id");

        let in_subjects = &tools[3];
        let scoped = in_subjects
            .execute(json!({
                "subject_id": subject_id,
                "queries": ["importable tool adapter"],
                "topK": 5
            }))
            .await
            .expect("search in subjects tool");
        let memories = scoped["memories"].as_array().expect("memories array");
        assert_eq!(memories.len(), 1, "scoped output: {scoped}");
        assert_eq!(memories[0]["id"], pair_id);
    }
}
