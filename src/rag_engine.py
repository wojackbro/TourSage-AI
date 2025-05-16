from typing import List
from langchain.schema import Document
from hf_utils import HuggingFaceClient
import os
import re
import logging

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine."""
        self.hf_client = HuggingFaceClient(api_key=os.getenv("HF_API_KEY"))

    def answer_question(self, question: str, context_docs: List[Document]) -> str:
        """Answer a question based on the provided context documents."""
        try:
            # Combine context documents
            context = "\n".join(doc.page_content for doc in context_docs)
            
            # Create a more focused prompt
            prompt = f"""Answer ONLY the following question based on the provided context. 
Be concise, direct, and do NOT repeat information. If the answer is a list, provide it only once. Do not include any additional questions or answers. If the answer is not found, say so clearly.

Question: {question}

Context: {context}

Answer:"""
            
            # Generate answer
            answer = self.hf_client.generate_text(prompt)
            
            # Clean up the answer
            answer = answer.strip()
            
            # Remove any additional questions or answers that might have been generated
            answer = re.sub(r'\nQuestion:.*', '', answer, flags=re.DOTALL)
            answer = re.sub(r'\nAnswer:.*', '', answer, flags=re.DOTALL)
            answer = re.sub(r'\d+\.\s*Question:.*', '', answer, flags=re.DOTALL)
            answer = re.sub(r'^(?:Based on|According to|From) the (?:provided|given) (?:context|information),?\s*', '', answer, flags=re.IGNORECASE)
            
            # Remove repeated sentences
            def remove_repeated_sentences(text):
                sentences = re.split(r'(?<=[.!?])\s+', text)
                seen = set()
                result = []
                for s in sentences:
                    s_clean = s.strip().lower()
                    if s_clean and s_clean not in seen:
                        seen.add(s_clean)
                        result.append(s.strip())
                return ' '.join(result)
            answer = remove_repeated_sentences(answer)
            
            # Ensure the answer is not empty
            if not answer.strip():
                return "I couldn't find a specific answer to your question in the provided context."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Error generating answer: {str(e)}" 