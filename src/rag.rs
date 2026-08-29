use crate::config::Config;
use crate::embeddings::{cosine_similarity, EmbeddingGenerator};
use crate::llm::LlmClient;
use crate::storage::{Document, DocumentStore};
use anyhow::Result;
use std::io::{self, Write};
use std::path::Path;
use walker::Walker;

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
        let embedder = EmbeddingGenerator::new(
            config.ollama_url.clone(),
            config.ollama_model.clone(),
        );
        let llm = LlmClient::new(config.ollama_url.clone(), config.ollama_model.clone());

        Ok(Self {
            config,
            store,
            embedder,
            llm,
        })
    }

    pub async fn index_directory(&mut self, path: &Path) -> Result<()> {
        println!("Scanning for JSON files...");
        let walker = Walker::new(path);
        let mut documents = Vec::new();

        for entry in walker.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                println!("Processing: {:?}", path);
                match self.load_json_file(&path).await {
                    Ok(mut docs) => documents.append(&mut docs),
                    Err(e) => eprintln!("Error processing {:?}: {}", path, e),
                }
            }
        }

        println!("\nIndexed {} documents", documents.len());
        self.store.add_documents(documents);
        self.store.save(&self.config.storage_path)?;

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
}
