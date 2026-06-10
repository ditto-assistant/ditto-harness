//! ditto-harness CLI: seed/dream/search/subjects/chat against a local Turso
//! database with Ollama / OpenRouter / vLLM model providers.
//!
//! The clap surface lives here (architect-owned); each subcommand dispatches
//! to its own handler file under `commands/` (CLI port agent-owned).

mod commands;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// Model provider selector for `--provider`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Provider {
    /// Local Ollama server (default base URL http://localhost:11434).
    Ollama,
    /// OpenRouter (requires OPENROUTER_API_KEY).
    Openrouter,
    /// OpenAI-compatible vLLM server (requires --base-url).
    Vllm,
}

/// Flags shared by every subcommand.
#[derive(Debug, Clone, Args)]
pub struct Common {
    /// Path to the Turso/SQLite database file (created if missing).
    #[arg(long, global = true, default_value = "ditto-harness.db")]
    pub db: String,

    /// User id that owns the memories.
    #[arg(long, global = true, default_value = "local")]
    pub user: String,

    /// Chat model provider.
    #[arg(long, global = true, value_enum, default_value_t = Provider::Ollama)]
    pub provider: Provider,

    /// Chat model name (provider default when omitted, e.g. gemma3:4b on
    /// Ollama).
    #[arg(long, global = true)]
    pub model: Option<String>,

    /// Base URL override for the provider endpoint.
    #[arg(long, global = true)]
    pub base_url: Option<String>,
}

#[derive(Debug, Parser)]
#[command(
    name = "ditto-harness",
    version,
    about = "Ditto agent memory harness: store, search, and chat over memories"
)]
struct Cli {
    #[command(flatten)]
    common: Common,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Seed memories into the database from a JSON file (or built-in sample
    /// data when omitted).
    Seed {
        /// JSON file of memories to ingest.
        #[arg(long)]
        file: Option<PathBuf>,
    },
    /// Run the dream pipeline (subject extraction/consolidation) for the
    /// user.
    Dream,
    /// Vector-search memories.
    Search {
        /// Search query text.
        #[arg(long)]
        query: String,
        /// Max results.
        #[arg(long, default_value_t = 8)]
        top_k: usize,
    },
    /// Search the user's subject graph.
    Subjects {
        /// Search query text.
        #[arg(long)]
        query: String,
        /// Max results.
        #[arg(long, default_value_t = 8)]
        top_k: usize,
    },
    /// Run a single chat turn through the agent loop with memory tools.
    Chat {
        /// User message.
        #[arg(long)]
        message: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Seed { file } => commands::seed::run(&cli.common, file.as_deref()).await,
        Command::Dream => commands::dream::run(&cli.common).await,
        Command::Search { query, top_k } => commands::search::run(&cli.common, &query, top_k).await,
        Command::Subjects { query, top_k } => {
            commands::subjects::run(&cli.common, &query, top_k).await
        }
        Command::Chat { message } => commands::chat::run(&cli.common, &message).await,
    }
}
