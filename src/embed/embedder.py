"""
Embedder
========
Takes code chunks, generates embeddings via the configured provider, and stores them in ChromaDB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_config, PipelineConfig
from src.llm_provider import embed_texts
from src.transform.chunker import CodeChunk


class Embedder:
    """
    Generates embeddings for code chunks and upserts them into ChromaDB.

    Usage:
        embedder = Embedder()
        embedder.embed_and_store(chunks)
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or get_config()
        self._chroma_client = self._init_chromadb()
        self._collection = self._get_or_create_collection()

    def _init_chromadb(self) -> chromadb.ClientAPI:
        """Initialize ChromaDB persistent client."""
        persist_dir = Path(self.config.vector_db.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(persist_dir))

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get or create the vector collection."""
        return self._chroma_client.get_or_create_collection(
            name=self.config.vector_db.collection_name,
            metadata={"hnsw:space": self.config.vector_db.distance_metric},
        )

    def embed_and_store(self, chunks: list[CodeChunk]) -> int:
        """
        Generate embeddings for chunks and store in vector DB.

        Returns the number of chunks successfully stored.
        """
        batch_size = self.config.embedding.batch_size
        total_stored = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            response = embed_texts([c.content for c in batch], config=self.config)
            embeddings = response.embeddings

            # Prepare for ChromaDB upsert
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [c.metadata for c in batch]

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total_stored += len(batch)

        return total_stored

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        where_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Query the vector DB.

        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances'
        """
        k = top_k or self.config.rag.top_k

        # Generate query embedding
        response = embed_texts([query_text], config=self.config)
        query_embedding = response.embeddings[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        return results

    def get_collection_stats(self) -> dict[str, Any]:
        """Get stats about the current collection."""
        return {
            "collection_name": self.config.vector_db.collection_name,
            "total_documents": self._collection.count(),
        }

    def reset_collection(self) -> None:
        """Delete and recreate the collection. Use with caution."""
        self._chroma_client.delete_collection(self.config.vector_db.collection_name)
        self._collection = self._get_or_create_collection()
