// SPDX-License-Identifier: MIT
//! ditto-harness: the open-source memory and agent harness extracted from the
//! Ditto backend. Stores and retrieves agent memories with vector search,
//! exposes memory tools, an importable agent loop, and a chat facade.

pub mod agent;
pub mod chat;
pub mod db;
pub mod dream;
pub mod memory;
pub mod models;
pub mod retrieval;
pub mod types;

pub use types::*;
