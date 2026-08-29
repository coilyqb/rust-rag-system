use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

pub struct EmbeddingGenerator {
    client: Client,
    use_simple: bool,
}

impl EmbeddingGenerator {
    pub fn new(use_simple: bool) -> Self {
        Self {
            client: Client::new(),
            use_simple,
        }
    }

    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        if self.use_simple {
            Ok(self.simple_embedding(text))
        } else {
            anyhow::bail!("Ollama embeddings not configured. Using simple embeddings.")
        }
    }

    pub async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        for text in texts {
            let embedding = self.generate_embedding(text).await?;
            embeddings.push(embedding);
        }
        Ok(embeddings)
    }

    // Simple TF-IDF-like embedding for local use without external models
    fn simple_embedding(&self, text: &str) -> Vec<f32> {
        const EMBEDDING_DIM: usize = 384;
        let mut embedding = vec![0.0; EMBEDDING_DIM];
        
        // Normalize text
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        
        if words.is_empty() {
            return embedding;
        }
        
        // Create a simple bag-of-words embedding
        for word in &words {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();
            
            // Map hash to multiple dimensions for better distribution
            for i in 0..3 {
                let idx = ((hash.wrapping_add(i as u64)) as usize) % EMBEDDING_DIM;
                embedding[idx] += 1.0 / words.len() as f32;
            }
        }
        
        // Add character n-grams for better matching
        for window in text_lower.chars().collect::<Vec<_>>().windows(3) {
            let ngram: String = window.iter().collect();
            let mut hasher = DefaultHasher::new();
            ngram.hash(&mut hasher);
            let hash = hasher.finish();
            let idx = (hash as usize) % EMBEDDING_DIM;
            embedding[idx] += 0.5 / (text_lower.len() as f32);
        }
        
        // Normalize the embedding vector
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut embedding {
                *val /= magnitude;
            }
        }
        
        embedding
    }
}

// Cosine similarity for finding relevant documents
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}
