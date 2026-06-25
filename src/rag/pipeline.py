"""
RAG Pipeline
============
Full retrieval-augmented generation pipeline.
Retrieves context from the vector store and generates answers using the configured LLM provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import get_config, PipelineConfig
from src.embed.embedder import Embedder
from src.llm_provider import generate, LLMResponse


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    query: str
    answer: str
    retrieval_context: list[str]
    metadata: list[dict[str, Any]]
    scores: list[float]             # Similarity scores from vector search

    @property
    def context_str(self) -> str:
        """Formatted context string."""
        return "\n\n---\n\n".join(self.retrieval_context)


SYSTEM_PROMPT = """You are a code-knowledgeable AI assistant. You answer questions about source code 
based ONLY on the provided context. If the context does not contain enough information to answer, 
say so clearly. Do not make up information.

When referencing code, mention the file path and relevant function/class names."""

QUERY_PROMPT_TEMPLATE = """Context (retrieved code chunks):
{context}

---

User Question: {query}

Provide a clear, accurate answer based on the context above."""


class RAGPipeline:
    """
    Orchestrates retrieval and generation.

    Flow:
    1. Query embedding + vector search (retrieval)
    2. Optional reranking
    3. LLM generation with retrieved context
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or get_config()
        self._embedder = Embedder(config=self.config)

    def query(self, user_query: str) -> RAGResponse:
        """
        Execute the full RAG pipeline for a user query.

        Args:
            user_query: Natural language question about the codebase.

        Returns:
            RAGResponse with answer and retrieved context.
        """
        # Step 1: Retrieve
        retrieval_results = self._retrieve(user_query)

        documents = retrieval_results["documents"][0] if retrieval_results["documents"] else []
        metadatas = retrieval_results["metadatas"][0] if retrieval_results["metadatas"] else []
        distances = retrieval_results["distances"][0] if retrieval_results["distances"] else []

        # Step 2: Filter by similarity threshold
        filtered_docs = []
        filtered_meta = []
        filtered_scores = []

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: lower = more similar. Convert to similarity.
            similarity = 1 - dist
            if similarity >= self.config.rag.similarity_threshold:
                filtered_docs.append(doc)
                filtered_meta.append(meta)
                filtered_scores.append(similarity)

        # Step 3: Rerank (if enabled and we have results)
        if self.config.rag.rerank and filtered_docs:
            filtered_docs, filtered_meta, filtered_scores = self._rerank(
                user_query, filtered_docs, filtered_meta, filtered_scores
            )

        # Step 4: Generate answer
        answer = self._generate(user_query, filtered_docs)

        return RAGResponse(
            query=user_query,
            answer=answer,
            retrieval_context=filtered_docs,
            metadata=filtered_meta,
            scores=filtered_scores,
        )

    def retrieve_only(self, query: str) -> dict[str, Any]:
        """Retrieve without generation - useful for evaluation."""
        return self._retrieve(query)

    def _retrieve(self, query: str) -> dict[str, Any]:
        """Perform vector search."""
        return self._embedder.query(query_text=query, top_k=self.config.rag.top_k)

    def _rerank(
        self,
        query: str,
        docs: list[str],
        metas: list[dict],
        scores: list[float],
    ) -> tuple[list[str], list[dict], list[float]]:
        """
        Rerank retrieved documents using LLM-based scoring.
        Returns reordered docs, metas, and scores.
        """
        rerank_prompt = (
            f'Given the query: "{query}"\n\n'
            f"Rate each of the following code chunks on relevance (0-10):\n\n"
            + "\n".join(f"[{i}] {doc[:200]}..." for i, doc in enumerate(docs))
            + "\n\nReturn a JSON list of scores like: [8, 3, 9, ...]"
        )

        try:
            response = generate(
                prompt=rerank_prompt,
                config=self.config,
                model_override=self.config.rag.rerank_model,  # None = use default model
            )
            import json
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                rerank_scores = json.loads(content[start:end])
            else:
                return docs, metas, scores

            # Sort by rerank score (descending)
            combined = list(zip(rerank_scores, docs, metas, scores))
            combined.sort(key=lambda x: x[0], reverse=True)

            return (
                [c[1] for c in combined],
                [c[2] for c in combined],
                [c[0] / 10.0 for c in combined],  # Normalize to 0-1
            )
        except Exception:
            # If reranking fails, return original order
            return docs, metas, scores

    def _generate(self, query: str, context_docs: list[str]) -> str:
        """Generate an answer using the LLM with retrieved context."""
        if not context_docs:
            return "I couldn't find relevant information in the codebase to answer your question."

        context = "\n\n---\n\n".join(context_docs)
        user_prompt = QUERY_PROMPT_TEMPLATE.format(context=context, query=query)

        response = generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            config=self.config,
        )

        return response.content
