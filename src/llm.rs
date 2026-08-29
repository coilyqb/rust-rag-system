use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
}

pub struct LlmClient {
    client: Client,
    ollama_url: String,
    model: String,
}

impl LlmClient {
    pub fn new(ollama_url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            ollama_url,
            model,
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.ollama_url);
        let request = OllamaRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("Ollama API error: {}", response.status());
        }

        let llm_response: OllamaResponse = response.json().await?;
        Ok(llm_response.response)
    }

    pub fn build_rag_prompt(
        &self,
        question: &str,
        context_documents: &[(String, f32)],
    ) -> String {
        let mut prompt = String::from(
            "You are a helpful AI assistant. Answer the question based on the provided context.\n\n",
        );

        prompt.push_str("Context:\n");
        for (i, (doc, score)) in context_documents.iter().enumerate() {
            prompt.push_str(&format!(
                "Document {} (relevance: {:.2}):\n{}\n\n",
                i + 1,
                score,
                doc
            ));
        }

        prompt.push_str(&format!(
            "Question: {}\n\nAnswer based on the context above:",
            question
        ));

        prompt
    }
}
