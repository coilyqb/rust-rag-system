use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub storage_path: PathBuf,
    pub embeddings_path: PathBuf,
    pub model_type: ModelType,
    pub ollama_url: String,
    pub ollama_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Ollama,
    LocalBert,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("./data/documents.json"),
            embeddings_path: PathBuf::from("./data/embeddings.json"),
            model_type: ModelType::Ollama,
            ollama_url: "http://localhost:11434".to_string(),
            ollama_model: "llama3.2".to_string(),
        }
    }
}

impl Config {
    pub fn ensure_directories(&self) -> anyhow::Result<()> {
        if let Some(parent) = self.storage_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if let Some(parent) = self.embeddings_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(())
    }
}
