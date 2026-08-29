# Rust RAG System

A local **Retrieval-Augmented Generation (RAG)** system built in Rust that uses normalized JSON files to build up an AI knowledge base for answering questions with a local LLM.

## Features

✨ **Local-first**: Runs entirely on your machine with local LLM (via Ollama)

📄 **JSON-based Knowledge Base**: Index normalized JSON files from any directory structure

🔍 **Semantic Search**: Uses embeddings to find relevant documents based on meaning, not just keywords

🤖 **LLM Integration**: Connects to local Ollama instance for embeddings and text generation

💬 **Interactive Chat**: Chat interface for asking questions about your indexed documents

🎯 **Retrieval-Augmented Answers**: Combines semantic search with LLM generation for accurate, context-aware responses

## Architecture

```
┌─────────────┐
│ JSON Files  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Index Process  │
│  - Load JSON    │
│  - Generate     │
│    Embeddings   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────┐
│ Document Store  │◄──────┤ Query Engine │
│ (documents.json)│       │  - Semantic  │
│                 │       │    Search    │
│ Embeddings ─────┼──────►│  - Ranking   │
└─────────────────┘       └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │  LLM Client  │
                          │   (Ollama)   │
                          │  - Context   │
                          │  - Generate  │
                          └──────────────┘
```

## Prerequisites

1. **Rust** (1.70+)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Ollama** - Local LLM runtime
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Then pull a model
   ollama pull llama3.2
   ```

## Installation

```bash
git clone https://github.com/coilyqb/rust-rag-system.git
cd rust-rag-system
cargo build --release
```

## Usage

### 1. Prepare Your JSON Data

Create a directory with normalized JSON files. The system supports:
- Single JSON objects
- Arrays of JSON objects
- Nested directory structures

Example JSON structure:

```json
[
  {
    "id": "doc1",
    "title": "Getting Started with Rust",
    "content": "Rust is a systems programming language...",
    "category": "programming"
  },
  {
    "id": "doc2",
    "title": "Advanced Rust Patterns",
    "content": "Ownership and borrowing are key concepts...",
    "category": "programming"
  }
]
```

### 2. Index Your Documents

```bash
cargo run --release -- index --path /path/to/your/json/files
```

This will:
- Recursively scan the directory for `.json` files
- Generate embeddings for each document
- Store everything in `./data/documents.json`

### 3. Query the System

**Single query:**
```bash
cargo run --release -- query "How does ownership work in Rust?"
```

**With custom number of context documents:**
```bash
cargo run --release -- query "What are Rust patterns?" --top-k 5
```

**Interactive chat mode:**
```bash
cargo run --release -- chat
```

## Configuration

The system uses sensible defaults, but you can customize by modifying `src/config.rs`:

```rust
pub struct Config {
    pub storage_path: PathBuf,          // Default: ./data/documents.json
    pub embeddings_path: PathBuf,       // Default: ./data/embeddings.json
    pub ollama_url: String,             // Default: http://localhost:11434
    pub ollama_model: String,           // Default: llama3.2
}
```

## How It Works

### Indexing Phase

1. **Document Loading**: Recursively scans directories for JSON files
2. **Normalization**: Parses JSON into structured documents
3. **Embedding Generation**: Uses Ollama embeddings API to create vector representations
4. **Storage**: Saves documents + embeddings in a normalized JSON store

### Query Phase

1. **Query Embedding**: Converts user question into a vector
2. **Semantic Search**: Computes cosine similarity between query and all documents
3. **Ranking**: Retrieves top-k most relevant documents
4. **Context Building**: Creates a prompt with relevant documents as context
5. **LLM Generation**: Sends prompt to Ollama for final answer generation

## Example Session

```bash
$ cargo run --release -- chat

RAG Chat Session Started!
Type 'exit' or 'quit' to end the session.

You: What is Rust's ownership system?
Generating query embedding...
Finding relevant documents...

Top 3 relevant documents:
  1. Similarity: 0.8923
  2. Similarity: 0.7654
  3. Similarity: 0.7201

Generating answer...

Assistant: Rust's ownership system is a set of rules that the compiler 
checks at compile time. It ensures memory safety without garbage collection. 
The key principles are:

1. Each value has a single owner
2. When the owner goes out of scope, the value is dropped
3. Values can be borrowed immutably or mutably, but not both simultaneously

This prevents common bugs like use-after-free and data races.

You: exit
Goodbye!
```

## Project Structure

```
rust-rag-system/
├── Cargo.toml           # Dependencies and project metadata
├── src/
│   ├── main.rs         # CLI entry point
│   ├── config.rs       # Configuration management
│   ├── storage.rs      # Document storage and persistence
│   ├── embeddings.rs   # Embedding generation and similarity
│   ├── llm.rs          # LLM client (Ollama integration)
│   └── rag.rs          # RAG orchestration logic
├── data/               # Auto-created storage directory
│   └── documents.json  # Indexed documents with embeddings
└── README.md
```

## Performance Tips

- **Batch Processing**: The indexer processes files sequentially; for large datasets, consider implementing batch embedding calls
- **Model Selection**: Smaller models (e.g., `llama3.2`) are faster; larger models provide better quality
- **Document Chunking**: For very large JSON documents, consider splitting them into smaller chunks for better retrieval

## Troubleshooting

**"Ollama API error"**:
- Ensure Ollama is running: `ollama serve`
- Check the model is available: `ollama list`

**"No relevant documents found"**:
- Verify documents were indexed: check `./data/documents.json`
- Try increasing `--top-k` parameter

**Slow embedding generation**:
- Use a smaller/faster embedding model
- Ensure Ollama is using GPU acceleration if available

## Future Enhancements

- [ ] Support for other LLM backends (OpenAI, Anthropic, local transformers)
- [ ] Vector database integration (Qdrant, Milvus)
- [ ] Document chunking strategies
- [ ] Hybrid search (semantic + keyword)
- [ ] Web UI interface
- [ ] Multi-modal support (images, PDFs)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Ollama](https://ollama.com/) for local LLM runtime
- [rust-bert](https://github.com/guillaume-be/rust-bert) for transformer models
- The Rust community for excellent tooling and libraries
