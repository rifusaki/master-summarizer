"""
Chunking agent (deterministic, no LLM).

Splits normalized document artifacts into semantic chunks that
respect heading boundaries and token budgets, then stores them
in ChromaDB for embedding-based retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

from src.config import pipeline_config
from src.models import (
    ArtifactType,
    Chunk,
    DocumentParseResult,
    NormalizedArtifact,
    SourceLocation,
)
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Use cl100k_base (GPT-4/Claude tokenizer) for token counting
_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding."""
    return len(_get_encoding().encode(text))


class Chunker:
    """
    Deterministic semantic chunker.

    Splits document artifacts into chunks that:
    - Respect heading/section boundaries (semantic splits)
    - Stay within a configurable token budget
    - Include overlap for context continuity
    - Preserve provenance metadata
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        token_budget: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.token_budget = token_budget or pipeline_config.chunk_token_budget
        self.overlap_tokens = overlap_tokens or pipeline_config.chunk_overlap_tokens

    def chunk_document(self, parse_result: DocumentParseResult) -> list[Chunk]:
        """
        Chunk a parsed document into semantic chunks.

        Strategy:
        1. Group artifacts by top-level section (heading level 1-2)
        2. Within each section, accumulate content until token budget
        3. Split at paragraph boundaries when budget exceeded
        4. Add overlap from previous chunk for context
        """
        logger.info(
            "Chunking document: %s (%d artifacts)",
            parse_result.source_file,
            len(parse_result.artifacts),
        )

        # Group artifacts by section
        sections = self._group_by_section(parse_result.artifacts)
        chunks: list[Chunk] = []
        sequence_idx = 0

        for section_title, artifacts in sections:
            section_chunks = self._chunk_section(
                artifacts=artifacts,
                section_title=section_title,
                document_id=parse_result.document_id,
                source_file=parse_result.source_file,
                start_sequence=sequence_idx,
            )
            chunks.extend(section_chunks)
            sequence_idx += len(section_chunks)

        # Store in vector DB if available
        if self.vector_store and chunks:
            self._store_embeddings(chunks)

        logger.info(
            "Created %d chunks from %s (avg %d tokens/chunk)",
            len(chunks),
            parse_result.source_file,
            sum(c.token_count for c in chunks) // max(len(chunks), 1),
        )

        return chunks

    def _group_by_section(
        self, artifacts: list[NormalizedArtifact]
    ) -> list[tuple[str, list[NormalizedArtifact]]]:
        """Group artifacts by top-level section headings."""
        sections: list[tuple[str, list[NormalizedArtifact]]] = []
        current_section = "Introduction"
        current_artifacts: list[NormalizedArtifact] = []

        for artifact in artifacts:
            # Check if this is a high-level heading (level 1 or 2)
            is_heading = artifact.metadata.get("is_heading", False)
            heading_level = artifact.metadata.get("heading_level", 0)

            if is_heading and heading_level <= 2 and current_artifacts:
                # Start new section
                sections.append((current_section, current_artifacts))
                current_section = artifact.content
                current_artifacts = [artifact]
            else:
                current_artifacts.append(artifact)

        # Don't forget the last section
        if current_artifacts:
            sections.append((current_section, current_artifacts))

        return sections

    def _chunk_section(
        self,
        artifacts: list[NormalizedArtifact],
        section_title: str,
        document_id: str,
        source_file: str,
        start_sequence: int,
    ) -> list[Chunk]:
        """Split a section's artifacts into token-budget-limited chunks."""
        chunks: list[Chunk] = []
        current_content_parts: list[str] = []
        current_tokens = 0
        current_artifact_ids: list[str] = []
        current_sources: list[SourceLocation] = []
        current_heading_path: list[str] = [section_title]
        contains_tables = False
        contains_images = False
        prev_overlap = ""

        for artifact in artifacts:
            artifact_text = artifact.content
            artifact_tokens = count_tokens(artifact_text)

            # Update heading path if this is a heading
            if artifact.metadata.get("is_heading"):
                level = artifact.metadata.get("heading_level", 1)
                while len(current_heading_path) > level:
                    current_heading_path.pop()
                if len(current_heading_path) < level:
                    current_heading_path.append(artifact_text)
                else:
                    current_heading_path[level - 1] = artifact_text

            # Check if adding this artifact would exceed the budget
            if (
                current_tokens + artifact_tokens > self.token_budget
                and current_content_parts
            ):
                # Finalize current chunk
                chunk = self._create_chunk(
                    content_parts=current_content_parts,
                    overlap_prefix=prev_overlap,
                    document_id=document_id,
                    source_file=source_file,
                    section_title=section_title,
                    heading_path=list(current_heading_path),
                    artifact_ids=current_artifact_ids,
                    source_locations=current_sources,
                    sequence_index=start_sequence + len(chunks),
                    contains_tables=contains_tables,
                    contains_images=contains_images,
                )
                chunks.append(chunk)

                # Prepare overlap for next chunk
                prev_overlap = self._get_overlap_text(current_content_parts)

                # Reset accumulators
                current_content_parts = []
                current_tokens = 0
                current_artifact_ids = []
                current_sources = []
                contains_tables = False
                contains_images = False

            # Add artifact to current chunk
            current_content_parts.append(artifact_text)
            current_tokens += artifact_tokens
            current_artifact_ids.append(artifact.artifact_id)
            current_sources.append(artifact.source)

            if artifact.artifact_type == ArtifactType.TABLE:
                contains_tables = True
            if artifact.artifact_type in (
                ArtifactType.IMAGE,
                ArtifactType.CHART,
                ArtifactType.MAP,
            ):
                contains_images = True

        # Finalize last chunk
        if current_content_parts:
            chunk = self._create_chunk(
                content_parts=current_content_parts,
                overlap_prefix=prev_overlap,
                document_id=document_id,
                source_file=source_file,
                section_title=section_title,
                heading_path=list(current_heading_path),
                artifact_ids=current_artifact_ids,
                source_locations=current_sources,
                sequence_index=start_sequence + len(chunks),
                contains_tables=contains_tables,
                contains_images=contains_images,
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        content_parts: list[str],
        overlap_prefix: str,
        document_id: str,
        source_file: str,
        section_title: str,
        heading_path: list[str],
        artifact_ids: list[str],
        source_locations: list[SourceLocation],
        sequence_index: int,
        contains_tables: bool,
        contains_images: bool,
    ) -> Chunk:
        """Create a Chunk object from accumulated content."""
        # Prepend overlap for context continuity
        full_content = (
            overlap_prefix + "\n\n" + "\n\n".join(content_parts)
            if overlap_prefix
            else "\n\n".join(content_parts)
        )

        return Chunk(
            document_id=document_id,
            source_file=source_file,
            content=full_content.strip(),
            token_count=count_tokens(full_content),
            heading_path=heading_path,
            section_title=section_title,
            contains_tables=contains_tables,
            contains_images=contains_images,
            sequence_index=sequence_index,
            artifact_ids=artifact_ids,
            source_locations=source_locations,
        )

    def _get_overlap_text(self, content_parts: list[str]) -> str:
        """Extract trailing text for overlap with next chunk."""
        full_text = "\n\n".join(content_parts)
        tokens = _get_encoding().encode(full_text)
        if len(tokens) <= self.overlap_tokens:
            return full_text
        overlap_tokens = tokens[-self.overlap_tokens :]
        return _get_encoding().decode(overlap_tokens)

    def _store_embeddings(self, chunks: list[Chunk]) -> None:
        """Store all chunks in the vector store for similarity search."""
        assert self.vector_store is not None

        chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "source_file": c.source_file,
                "section_title": c.section_title,
                "sequence_index": c.sequence_index,
                "token_count": c.token_count,
                "contains_tables": c.contains_tables,
                "contains_images": c.contains_images,
            }
            for c in chunks
        ]

        self.vector_store.add_chunks_batch(chunk_ids, texts, metadatas)

        # Mark chunks as having embeddings
        for chunk in chunks:
            chunk.has_embedding = True

        logger.info("Stored %d chunk embeddings in vector store", len(chunks))
