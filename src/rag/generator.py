"""LLM generation pipeline for RAG system."""

import json
import os
from typing import List, Optional

import requests

try:
    from sentence_transformers import SentenceTransformer
    has_local_transformer = True
except ImportError:
    has_local_transformer = False

from src.config import EMBEDDING_CONFIG, OLLAMA_URL, LLM_MODEL
from src.db.pgvector import PGVectorDB, SearchResult


def resolve_host_via_doh(host: str) -> Optional[str]:
    """Resolve host using Google's DNS-over-HTTPS API over IP to bypass Vercel DNS bugs."""
    try:
        url = f"https://8.8.8.8/resolve?name={host}&type=A"
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, verify=False, timeout=4)
        response.raise_for_status()
        data = response.json()
        if "Answer" in data:
            for answer in data["Answer"]:
                if answer.get("type") == 1:  # A record (IPv4)
                    return answer.get("data")
    except Exception as e:
        print(f"DoH resolution failed for {host}: {e}")
    return None


class RAGPipeline:
    """End-to-end RAG pipeline combining retrieval and generation."""

    def __init__(
        self,
        db: Optional[PGVectorDB] = None,
        embed_model = None,
        llm_url: str = OLLAMA_URL,
        llm_model: str = LLM_MODEL,
    ):
        self.db = db or PGVectorDB()
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.embed_model = embed_model

        if embed_model is None:
            if os.getenv("VERCEL") == "1" or os.getenv("HF_TOKEN") or not has_local_transformer:
                print("Using HuggingFace Inference API for embeddings (no local SentenceTransformer).")
                self.embed_model = None
            else:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                    print(f"Loading local embedding model ({EMBEDDING_CONFIG.model_name}) on {device}...")
                    self.embed_model = SentenceTransformer(EMBEDDING_CONFIG.model_name, device=device)
                except Exception as e:
                    print(f"Failed to load local model: {e}. Falling back to API mode.")
                    self.embed_model = None
        else:
            self.embed_model = embed_model

    def get_query_embedding(self, query: str) -> List[float]:
        """Embed query using HuggingFace Inference API or local model."""
        if self.embed_model is None:
            hf_token = os.getenv("HF_TOKEN")
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            
            # Dynamic DNS resolution via Google DNS-over-HTTPS to bypass Vercel DNS bug
            host = "router.huggingface.co"
            ip = resolve_host_via_doh(host)
            
            if ip:
                print(f"DNS Over HTTPS: Resolved {host} to {ip}")
                url = f"https://{ip}/hf-inference/models/{EMBEDDING_CONFIG.model_name}"
                headers["Host"] = host
            else:
                print("DNS Over HTTPS failed. Falling back to direct DNS subdomain lookup.")
                url = f"https://{host}/hf-inference/models/{EMBEDDING_CONFIG.model_name}"
            
            last_err = None
            import time
            for attempt in range(3):
                try:
                    # If we query via raw IP, disable SSL host validation (since cert is for *.huggingface.co)
                    verify_ssl = False if ip else True
                    if not verify_ssl:
                        import urllib3
                        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

                    response = requests.post(
                        url, 
                        headers=headers, 
                        json={"inputs": [query], "options": {"wait_for_model": True}},
                        timeout=8,
                        verify=verify_ssl
                    )
                    response.raise_for_status()
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            return result[0]
                        return result
                    raise ValueError(f"Unexpected response format from HF API: {result}")
                except Exception as e:
                    last_err = e
                    print(f"HuggingFace query attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
            
            raise RuntimeError(f"Failed to generate embedding via HuggingFace Inference API after 3 attempts: {last_err}")
        else:
            return self.embed_model.encode(query, convert_to_numpy=True).tolist()

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        use_hybrid: bool = True,
        filters: Optional[dict] = None,
    ) -> List[SearchResult]:
        """Retrieve relevant documents for query.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            use_hybrid: Use hybrid (vector + keyword) search if True.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult objects.
        """
        query_embedding = self.get_query_embedding(query)

        if use_hybrid:
            return self.db.hybrid_search(query, query_embedding, limit)
        else:
            return self.db.search(query_embedding, limit, filters)

    def generate(
        self,
        query: str,
        retrieved_docs: List[SearchResult],
        stream: bool = True,
    ) -> str:
        """Generate response using LLM with retrieved context.

        Args:
            query: User question.
            retrieved_docs: Retrieved documents as context.
            stream: Stream response token-by-token if True.

        Returns:
            Generated response string.
        """
        context_text = "\n\n".join([
            f"Document {doc.id}: {doc.content}"
            for doc in retrieved_docs
        ])

        prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context documents.
If the answer is not contained in the context, say "I don't have enough information to answer that."

Context:
{context_text}

Question: {query}
Answer:"""

        if stream:
            return self._stream_generate(prompt)
        else:
            return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        """Generate response without streaming, falling back to hosted APIs if configured."""
        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if openai_key:
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Error querying OpenAI API: {e}"

        elif groq_key:
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            model = os.getenv("LLM_MODEL", "llama3-8b-8192")
            if model in ["tinyllama", "llama3.2"]:
                model = "llama3-8b-8192"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            try:
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Error querying Groq API: {e}"

        # Default fallback to local Ollama
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"Ollama connection refused. Verify Ollama is running locally, or configure OPENAI_API_KEY / GROQ_API_KEY for cloud hosted APIs. Details: {e}"

    def _stream_generate(self, prompt: str) -> str:
        """Generate response with streaming output (local Ollama only)."""
        print(f"\nQuerying {self.llm_model}...")
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": True,
                },
                stream=True,
            )
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    print(token, end="", flush=True)
                    full_response += token
            print("\n")
            return full_response
        except Exception as e:
            return f"Failed to connect to local Ollama: {e}"

    def query(
        self,
        query: str,
        limit: int = 5,
        use_hybrid: bool = True,
        filters: Optional[dict] = None,
        stream: bool = True,
    ) -> str:
        """End-to-end RAG query.

        Args:
            query: User question.
            limit: Number of documents to retrieve.
            use_hybrid: Use hybrid search if True.
            filters: Optional metadata filters.
            stream: Stream LLM response if True.

        Returns:
            Generated response.
        """
        print("1. Embedding query...")
        docs = self.retrieve(query, limit, use_hybrid, filters)

        if not docs:
            return "No documents found."

        print(f"2. Retrieved {len(docs)} documents...")
        return self.generate(query, docs, stream)


def main():
    """Test the RAG pipeline."""
    pipeline = RAGPipeline()

    test_query = "What is the main topic of the papers regarding deep learning?"
    print(f"\nQuery: {test_query}\n")
    print("-" * 50)

    pipeline.query(test_query, limit=3)


if __name__ == "__main__":
    main()
