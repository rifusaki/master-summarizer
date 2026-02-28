"""
ChromaDB vector store for chunk embeddings.

Provides storage, retrieval, and similarity search for document
chunks using ChromaDB with its default embedding model.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import CHROMA_DIR

logger = logging.getLogger(__name__)

# Collection names
CHUNKS_COLLECTION = "document_chunks"


class VectorStore:
    """
    ChromaDB-backed vector store for chunk embeddings.

    Uses ChromaDB's default embedding function (all-MiniLM-L6-v2)
    for local, cost-free embedding generation.
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        self._persist_dir = persist_dir or str(CHROMA_DIR)
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    def initialize(self) -> None:
        """Initialize the ChromaDB client and collection."""
        logger.info("Initializing ChromaDB at %s", self._persist_dir)
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=CHUNKS_COLLECTION,
            metadata={"description": "Document chunks for summarization pipeline"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d documents)",
            CHUNKS_COLLECTION,
            self._collection.count(),
        )

    @property
    def collection(self) -> chromadb.Collection:
        assert self._collection is not None, "VectorStore not initialized"
        return self._collection

    # ------------------------------------------------------------------
    # Add chunks
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a single chunk to the vector store."""
        self.collection.upsert(
            ids=[chunk_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def add_chunks_batch(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple chunks in a single batch."""
        if not chunk_ids:
            return
        self.collection.upsert(
            ids=chunk_ids,
            documents=texts,
            metadatas=metadatas or [{}] * len(chunk_ids),
        )
        logger.info("Added %d chunks to vector store", len(chunk_ids))

    # ------------------------------------------------------------------
    # Query / retrieval
    # ------------------------------------------------------------------

    def query_similar(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find chunks most similar to the query text.

        Returns list of dicts with 'id', 'document', 'metadata', 'distance'.
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, self.collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        items = []
        for i in range(len(results["ids"][0])):
            items.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i]
                    if results["documents"]
                    else "",
                    "metadata": results["metadatas"][0][i]
                    if results["metadatas"]
                    else {},
                    "distance": results["distances"][0][i]
                    if results["distances"]
                    else 0,
                }
            )
        return items

    def query_by_metadata(
        self,
        where: dict[str, Any],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks by metadata filter."""
        results = self.collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        items = []
        for i in range(len(results["ids"])):
            items.append(
                {
                    "id": results["ids"][i],
                    "document": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                }
            )
        return items

    def get_all_for_document(self, document_id: str) -> list[dict[str, Any]]:
        """Get all chunks belonging to a specific document."""
        return self.query_by_metadata(
            where={"document_id": document_id},
            limit=10000,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of chunks in the store."""
        return self.collection.count()

    def clear(self) -> None:
        """Remove all chunks from the collection."""
        if self._client and self._collection:
            self._client.delete_collection(CHUNKS_COLLECTION)
            self._collection = self._client.get_or_create_collection(
                name=CHUNKS_COLLECTION,
            )
            logger.info("Vector store cleared")
