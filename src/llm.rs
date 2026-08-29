use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct ChatRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    response: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    content: String,
}

pub struct LlmClient {
    client: Client,
    chat_url: String,
}

impl LlmClient {
    pub fn new(chat_url: String) -> Self {
        Self {
            client: Client::new(),
            chat_url,
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String> {
        let request = ChatRequest {
            prompt: prompt.to_string(),
            temperature: Some(0.7),
        };

        let response = self
            .client
            .post(&self.chat_url)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Chat API error {}: {}", status, error_text);
        }

        let chat_response: ChatResponse = response.json().await?;
        
        // Try different possible response fields
        let text = if !chat_response.response.is_empty() {
            chat_response.response
        } else if !chat_response.text.is_empty() {
            chat_response.text
        } else if !chat_response.content.is_empty() {
            chat_response.content
        } else {
            anyhow::bail!("Empty response from chat API")
        };
        
        Ok(text)
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
