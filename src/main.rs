mod config;
mod embeddings;
mod llm;
mod rag;
mod storage;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rust-rag")]
#[command(about = "A local RAG system using normalized JSON files", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Index JSON documents into the RAG system
    Index {
        /// Path to the directory containing JSON files
        #[arg(short, long)]
        path: PathBuf,
    },
    /// Query the RAG system with a question
    Query {
        /// The question to ask
        question: String,
        
        /// Number of relevant documents to retrieve
        #[arg(short, long, default_value = "3")]
        top_k: usize,
    },
    /// Start an interactive chat session
    Chat,
    /// Debug commands to inspect indexed data
    Debug {
        #[command(subcommand)]
        action: DebugAction,
    },
}

#[derive(Subcommand)]
enum DebugAction {
    /// List all indexed documents
    List {
        /// Show full content of each document
        #[arg(short, long)]
        full: bool,
    },
    /// Search for documents without LLM generation
    Search {
        /// Search query
        query: String,
        
        /// Number of results to show
        #[arg(short, long, default_value = "5")]
        top_k: usize,
        
        /// Show full document content
        #[arg(short, long)]
        full: bool,
    },
    /// Show a specific document by ID
    Show {
        /// Document ID
        id: String,
    },
    /// Show statistics about the indexed data
    Stats,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { path } => {
            println!("Indexing documents from: {:?}", path);
            let mut rag = rag::RagSystem::new()?;
            rag.index_directory(&path).await?;
            println!("Indexing complete!");
        }
        Commands::Query { question, top_k } => {
            println!("Query: {}", question);
            let rag = rag::RagSystem::new()?;
            let answer = rag.query(&question, top_k).await?;
            println!("\nAnswer:\n{}", answer);
        }
        Commands::Chat => {
            println!("Starting interactive chat session...");
            let rag = rag::RagSystem::new()?;
            rag.interactive_chat().await?;
        }
        Commands::Debug { action } => {
            let rag = rag::RagSystem::new()?;
            match action {
                DebugAction::List { full } => {
                    rag.debug_list(full)?;
                }
                DebugAction::Search { query, top_k, full } => {
                    rag.debug_search(&query, top_k, full).await?;
                }
                DebugAction::Show { id } => {
                    rag.debug_show(&id)?;
                }
                DebugAction::Stats => {
                    rag.debug_stats()?;
                }
            }
        }
    }

    Ok(())
}
