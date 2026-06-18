// SPDX-License-Identifier: AGPL-3.0-or-later
//! ditto-harness: open-source agent memory harness (Rust port of the Go
//! original). Stores and retrieves agent memories with vector search, exposes
//! memory tools, an importable agent loop, and a chat facade.

pub mod agent;
pub mod chat;
pub mod db;
pub mod dream;
pub mod memory;
pub mod models;
pub mod retrieval;
pub mod types;

pub use types::*;
