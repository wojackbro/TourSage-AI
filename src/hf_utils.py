import os
import requests
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class HuggingFaceClient:
    def __init__(self, api_key: str):
        """Initialize Hugging Face client.
        
        Args:
            api_key: Hugging Face API key
        """
        self.api_key = api_key
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Model IDs for different tasks
        self.models = {
            'summarization': 'facebook/bart-large-cnn',
            'text_generation': 'cutycat2000x/MeowGPT-3.5',
            'classification': 'facebook/bart-large-mnli'
        }

    def generate_text(self, prompt: str, max_length: int = 500) -> str:
        """Generate text using Together AI's meta-llama/Llama-3.3-70B-Instruct-Turbo-Free model."""
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            return "Together AI API key not set. Please set TOGETHER_API_KEY in your environment."
        url = "https://api.together.xyz/v1/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "prompt": prompt,
            "max_tokens": max_length,
            "temperature": 0.7,
            "top_p": 0.9
        }
        for attempt in range(2):  # Try twice to handle cold start
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                print("[DEBUG] TogetherAI Text Generation raw response:", response.text)
                if response.status_code in (503, 504):
                    if attempt == 0:
                        print("[INFO] TogetherAI model cold start or timeout, retrying...")
                        continue  # Try again
                    else:
                        return ("TogetherAI text generation timed out or model is cold. Try again later, or consider using a paid TogetherAI endpoint for faster and more reliable results.")
                if not response.ok:
                    return (f"TogetherAI text generation failed: {response.status_code} {response.reason}. "
                            "This may be due to model unavailability or API limits.")
                try:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0 and "text" in result["choices"][0]:
                        return result["choices"][0]["text"].strip()
                    else:
                        return f"Unexpected TogetherAI text generation output format: {result}"
                except Exception as e:
                    return f"Failed to parse TogetherAI text generation result: {str(e)} | Raw output: {response.text}"
            except requests.Timeout:
                if attempt == 0:
                    print("[INFO] TogetherAI text generation request timed out, retrying...")
                    continue
                return ("TogetherAI text generation request timed out. Try again later, or use a paid endpoint for better reliability.")
            except Exception as e:
                return (f"TogetherAI text generation failed: {str(e)}. This may be due to API issues or model unavailability.")

    def summarize_text(self, text: str, max_length: int = 130) -> str:
        """Summarize text using the Hugging Face facebook/bart-large-cnn model."""
        # Truncate input to 700 words to avoid model token limits
        if len(text.split()) > 700:
            text = " ".join(text.split()[:700])
        api_url = f"https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}", "Content-Type": "application/json"}
        payload = {
            "inputs": text,
            "parameters": {"max_length": max_length, "min_length": 30}
        }
        for attempt in range(2):  # Try twice to handle cold start
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                print("[DEBUG] Summarization raw response:", response.text)
                if response.status_code in (503, 504):
                    if attempt == 0:
                        print("[INFO] Model cold start or timeout, retrying...")
                        continue  # Try again
                    else:
                        return ("Summarization timed out or model is cold. This often happens with free-tier models on the Hugging Face API. "
                                "Try again later, or consider using a paid Hugging Face Inference Endpoint for faster and more reliable results.")
                if not response.ok:
                    return (f"Summarization failed: {response.status_code} {response.reason}. "
                            "This may be due to model unavailability or API limits. Consider using a paid endpoint or running a local model for better results.")
                try:
                    result = response.json()
                    if isinstance(result, list) and 'summary_text' in result[0]:
                        return result[0]['summary_text']
                    else:
                        return f"Unexpected summarization output format: {result}"
                except Exception as e:
                    return f"Failed to parse summarization result: {str(e)} | Raw output: {response.text}"
            except requests.Timeout:
                if attempt == 0:
                    print("[INFO] Summarization request timed out, retrying...")
                    continue
                return ("Summarization request timed out. This is common on the free API tier. Try again later, or use a paid endpoint for better reliability.")
            except Exception as e:
                return (f"Summarization failed: {str(e)}. This may be due to API issues or model unavailability. Consider using a paid endpoint or running a local model.")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        return self.embedding_model.encode(texts).tolist()

    def classify_text(self, text: str, labels: List[str]) -> Dict[str, float]:
        """Classify text into given labels using Hugging Face HTTP API."""
        api_url = f"https://api-inference.huggingface.co/models/{self.models['classification']}"
        headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": labels}
        }
        response = requests.post(api_url, headers=headers, json=payload)
        print("[DEBUG] Classification raw response:", response.text)
        result = response.json()
        
        # Ensure scores are properly normalized between 0 and 1
        scores = result["scores"]
        total = sum(scores)
        if total > 0:
            normalized_scores = [score/total for score in scores]
        else:
            normalized_scores = scores
            
        return dict(zip(result["labels"], normalized_scores)) 