"""LLM generation pipeline for RAG system."""

import json
from typing import List, Optional

import requests
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_CONFIG, OLLAMA_URL, LLM_MODEL
from src.db.pgvector import PGVectorDB, SearchResult


class RAGPipeline:
    """End-to-end RAG pipeline combining retrieval and generation."""

    def __init__(
        self,
        db: Optional[PGVectorDB] = None,
        embed_model: Optional[SentenceTransformer] = None,
        llm_url: str = OLLAMA_URL,
        llm_model: str = LLM_MODEL,
    ):
        self.db = db or PGVectorDB()
        self.llm_url = llm_url
        self.llm_model = llm_model

        if embed_model is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"Loading embedding model ({EMBEDDING_CONFIG.model_name}) on {device}...")
            self.embed_model = SentenceTransformer(EMBEDDING_CONFIG.model_name, device=device)
        else:
            self.embed_model = embed_model

    def get_query_embedding(self, query: str) -> List[float]:
        """Embed query using the configured model."""
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
        """Generate response without streaming."""
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

    def _stream_generate(self, prompt: str) -> str:
        """Generate response with streaming output."""
        print(f"\nQuerying {self.llm_model}...")
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
