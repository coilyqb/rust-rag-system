use crate::config::Config;
use crate::embeddings::{cosine_similarity, EmbeddingGenerator};
use crate::llm::LlmClient;
use crate::storage::{Document, DocumentStore};
use anyhow::Result;
use std::io::{self, Write};
use std::path::Path;
use walkdir::WalkDir;

pub struct RagSystem {
    config: Config,
    store: DocumentStore,
    embedder: EmbeddingGenerator,
    llm: LlmClient,
}

impl RagSystem {
    pub fn new() -> Result<Self> {
        let config = Config::default();
        config.ensure_directories()?;

        let store = DocumentStore::load(&config.storage_path)?;
        let embedder = EmbeddingGenerator::new(config.use_simple_embeddings);
        let llm = LlmClient::new(config.chat_url.clone());

        Ok(Self {
            config,
            store,
            embedder,
            llm,
        })
    }

    pub async fn index_directory(&mut self, path: &Path) -> Result<()> {
        println!("Scanning for JSON files...");
        let mut documents = Vec::new();
        let mut file_count = 0;

        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                file_count += 1;
                println!("\n[File #{}] Processing: {:?}", file_count, path);
                match self.load_json_file(&path).await {
                    Ok(mut docs) => {
                        println!("  ✓ Extracted {} document(s)", docs.len());
                        for (i, doc) in docs.iter().enumerate() {
                            println!("    - Document #{}: {} chars, embedding dim: {}", 
                                i + 1, 
                                doc.content.len(),
                                doc.embedding.as_ref().map(|e| e.len()).unwrap_or(0)
                            );
                        }
                        documents.append(&mut docs);
                    }
                    Err(e) => eprintln!("  ✗ Error processing {:?}: {}", path, e),
                }
            }
        }

        println!("\n{}", "=".repeat(60));
        println!("✓ Indexing complete!");
        println!("  Files processed: {}", file_count);
        println!("  Total documents: {}", documents.len());
        println!("{}", "=".repeat(60));
        
        self.store.add_documents(documents);
        self.store.save(&self.config.storage_path)?;
        println!("\n✓ Saved to: {:?}", self.config.storage_path);

        Ok(())
    }

    async fn load_json_file(&self, path: &Path) -> Result<Vec<Document>> {
        let content = std::fs::read_to_string(path)?;
        let json_value: serde_json::Value = serde_json::from_str(&content)?;

        let mut documents = Vec::new();

        match json_value {
            serde_json::Value::Array(items) => {
                for (idx, item) in items.iter().enumerate() {
                    let doc = self.create_document_from_json(
                        &format!("{}#{}", path.display(), idx),
                        item,
                    ).await?;
                    documents.push(doc);
                }
            }
            serde_json::Value::Object(_) => {
                let doc = self.create_document_from_json(
                    &path.display().to_string(),
                    &json_value,
                ).await?;
                documents.push(doc);
            }
            _ => anyhow::bail!("Unsupported JSON format"),
        }

        Ok(documents)
    }

    async fn create_document_from_json(
        &self,
        id: &str,
        json: &serde_json::Value,
    ) -> Result<Document> {
        let content = serde_json::to_string_pretty(json)?;
        let embedding = self.embedder.generate_embedding(&content).await?;

        Ok(Document {
            id: id.to_string(),
            content,
            metadata: json.clone(),
            embedding: Some(embedding),
        })
    }

    pub async fn query(&self, question: &str, top_k: usize) -> Result<String> {
        println!("Generating query embedding...");
        let query_embedding = self.embedder.generate_embedding(question).await?;

        println!("Finding relevant documents...");
        let mut scored_docs: Vec<(&Document, f32)> = self
            .store
            .get_all_documents()
            .iter()
            .filter_map(|doc| {
                doc.embedding.as_ref().map(|emb| {
                    let score = cosine_similarity(&query_embedding, emb);
                    (doc, score)
                })
            })
            .collect();

        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_docs: Vec<(String, f32)> = scored_docs
            .iter()
            .take(top_k)
            .map(|(doc, score)| (doc.content.clone(), *score))
            .collect();

        println!("\nTop {} relevant documents:", top_k);
        for (i, (_, score)) in top_docs.iter().enumerate() {
            println!("  {}. Similarity: {:.4}", i + 1, score);
        }

        println!("\nGenerating answer...");
        let prompt = self.llm.build_rag_prompt(question, &top_docs);
        let answer = self.llm.generate(&prompt).await?;

        Ok(answer)
    }

    pub async fn interactive_chat(&self) -> Result<()> {
        println!("RAG Chat Session Started!");
        println!("Type 'exit' or 'quit' to end the session.\n");

        loop {
            print!("You: ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let question = input.trim();

            if question.eq_ignore_ascii_case("exit") || question.eq_ignore_ascii_case("quit") {
                println!("Goodbye!");
                break;
            }

            if question.is_empty() {
                continue;
            }

            match self.query(question, 3).await {
                Ok(answer) => println!("\nAssistant: {}\n", answer),
                Err(e) => eprintln!("Error: {}\n", e),
            }
        }

        Ok(())
    }

    // Debug commands
    pub fn debug_list(&self, full: bool) -> Result<()> {
        let docs = self.store.get_all_documents();
        
        if docs.is_empty() {
            println!("No documents indexed yet.");
            println!("Run: cargo run --release -- index --path <path_to_json_files>");
            return Ok(());
        }

        println!("\n{}", "=".repeat(80));
        println!("INDEXED DOCUMENTS ({} total)", docs.len());
        println!("{}", "=".repeat(80));

        for (i, doc) in docs.iter().enumerate() {
            println!("\n[{}] ID: {}", i + 1, doc.id);
            println!("    Content length: {} chars", doc.content.len());
            println!("    Embedding dimension: {}", 
                doc.embedding.as_ref().map(|e| e.len()).unwrap_or(0));
            
            if full {
                println!("\n    --- FULL CONTENT ---");
                println!("{}", doc.content);
                println!("    --- END CONTENT ---");
            } else {
                let preview = if doc.content.len() > 200 {
                    format!("{}...", &doc.content[..200])
                } else {
                    doc.content.clone()
                };
                println!("\n    Content preview:");
                for line in preview.lines().take(5) {
                    println!("    {}", line);
                }
            }
            
            if !full && i < docs.len() - 1 {
                println!("    {}", "-".repeat(76));
            }
        }

        println!("\n{}", "=".repeat(80));
        println!("Tip: Use --full flag to see complete document contents");
        println!("{}\n", "=".repeat(80));

        Ok(())
    }

    pub async fn debug_search(&self, query: &str, top_k: usize, full: bool) -> Result<()> {
        let docs = self.store.get_all_documents();
        
        if docs.is_empty() {
            println!("No documents indexed yet.");
            return Ok(());
        }

        println!("\n{}", "=".repeat(80));
        println!("SEARCH DEBUG: '{}'", query);
        println!("{}", "=".repeat(80));

        println!("\n[1/2] Generating query embedding...");
        let query_embedding = self.embedder.generate_embedding(query).await?;
        println!("      ✓ Embedding dimension: {}", query_embedding.len());

        println!("\n[2/2] Computing similarities...");
        let mut scored_docs: Vec<(&Document, f32)> = docs
            .iter()
            .filter_map(|doc| {
                doc.embedding.as_ref().map(|emb| {
                    let score = cosine_similarity(&query_embedding, emb);
                    (doc, score)
                })
            })
            .collect();

        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("\n{}", "=".repeat(80));
        println!("TOP {} MATCHES", top_k.min(scored_docs.len()));
        println!("{}", "=".repeat(80));

        for (i, (doc, score)) in scored_docs.iter().take(top_k).enumerate() {
            println!("\n[Match #{}] Similarity: {:.4}", i + 1, score);
            println!("           Document ID: {}", doc.id);
            println!("           Content length: {} chars", doc.content.len());
            
            if full {
                println!("\n    --- FULL CONTENT ---");
                println!("{}", doc.content);
                println!("    --- END CONTENT ---");
            } else {
                let preview = if doc.content.len() > 300 {
                    format!("{}...", &doc.content[..300])
                } else {
                    doc.content.clone()
                };
                println!("\n    Preview:");
                for line in preview.lines().take(8) {
                    println!("    {}", line);
                }
            }
            
            if i < top_k.min(scored_docs.len()) - 1 {
                println!("\n    {}", "-".repeat(76));
            }
        }

        println!("\n{}", "=".repeat(80));
        println!("Note: This shows raw search results without LLM generation");
        println!("{}\n", "=".repeat(80));

        Ok(())
    }

    pub fn debug_show(&self, id: &str) -> Result<()> {
        let docs = self.store.get_all_documents();
        
        match docs.iter().find(|d| d.id == id) {
            Some(doc) => {
                println!("\n{}", "=".repeat(80));
                println!("DOCUMENT DETAILS");
                println!("{}", "=".repeat(80));
                println!("\nID: {}", doc.id);
                println!("Content length: {} chars", doc.content.len());
                println!("Embedding dimension: {}", 
                    doc.embedding.as_ref().map(|e| e.len()).unwrap_or(0));
                
                if let Some(emb) = &doc.embedding {
                    let sample: Vec<String> = emb.iter().take(10).map(|v| format!("{:.4}", v)).collect();
                    println!("Embedding sample (first 10): [{}...]", sample.join(", "));
                }

                println!("\n--- METADATA ---");
                println!("{}", serde_json::to_string_pretty(&doc.metadata)?);

                println!("\n--- FULL CONTENT ---");
                println!("{}", doc.content);
                println!("--- END ---");
                println!("\n{}\n", "=".repeat(80));
            }
            None => {
                println!("\n✗ Document not found: {}", id);
                println!("\nAvailable document IDs:");
                for doc in docs {
                    println!("  - {}", doc.id);
                }
                println!();
            }
        }

        Ok(())
    }

    pub fn debug_stats(&self) -> Result<()> {
        let docs = self.store.get_all_documents();
        
        if docs.is_empty() {
            println!("\nNo documents indexed yet.\n");
            return Ok(());
        }

        let total_docs = docs.len();
        let total_chars: usize = docs.iter().map(|d| d.content.len()).sum();
        let avg_chars = total_chars / total_docs.max(1);
        let with_embeddings = docs.iter().filter(|d| d.embedding.is_some()).count();
        let embedding_dim = docs.first()
            .and_then(|d| d.embedding.as_ref())
            .map(|e| e.len())
            .unwrap_or(0);

        println!("\n{}", "=".repeat(80));
        println!("RAG SYSTEM STATISTICS");
        println!("{}", "=".repeat(80));
        println!("\nStorage:");
        println!("  Documents path: {:?}", self.config.storage_path);
        println!("  Embeddings path: {:?}", self.config.embeddings_path);
        
        println!("\nDocuments:");
        println!("  Total documents: {}", total_docs);
        println!("  With embeddings: {}", with_embeddings);
        println!("  Total characters: {}", total_chars);
        println!("  Average chars/doc: {}", avg_chars);
        
        println!("\nEmbeddings:");
        println!("  Embedding dimension: {}", embedding_dim);
        println!("  Using simple embeddings: {}", self.config.use_simple_embeddings);
        
        println!("\nLLM Configuration:");
        println!("  Chat URL: {}", self.config.chat_url);
        println!("  Model: {}", self.config.chat_model);
        
        println!("\nDocument IDs:");
        for (i, doc) in docs.iter().enumerate() {
            println!("  [{}] {}", i + 1, doc.id);
        }

        println!("\n{}\n", "=".repeat(80));

        Ok(())
    }
}
