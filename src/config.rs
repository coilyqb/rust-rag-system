use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub storage_path: PathBuf,
    pub embeddings_path: PathBuf,
    pub model_type: ModelType,
    pub chat_url: String,
    pub chat_model: String,
    pub use_simple_embeddings: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    CustomChat,
    Ollama,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            storage_path: PathBuf::from("./data/documents.json"),
            embeddings_path: PathBuf::from("./data/embeddings.json"),
            model_type: ModelType::CustomChat,
            chat_url: "http://localhost:3000/chat".to_string(),
            chat_model: "gemma".to_string(),
            use_simple_embeddings: true,
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
