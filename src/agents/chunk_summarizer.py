"""
Chunk summarization agent (Claude Sonnet 4.6).

Produces faithful, length-limited summaries of individual chunks
while preserving numeric values, key facts, and provenance links.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.models import (
    Chunk,
    ChunkSummary,
    KeyFact,
    NumericEntry,
    PipelineStage,
)

logger = logging.getLogger(__name__)


CHUNK_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Faithful summary of the chunk content. Preserve all numeric values and units.",
        },
        "key_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "demographic",
                            "geographic",
                            "economic",
                            "environmental",
                            "infrastructure",
                            "social",
                            "legal",
                            "administrative",
                            "cultural",
                            "risk",
                            "other",
                        ],
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["fact", "category"],
            },
            "description": "Key facts extracted from the chunk",
        },
        "numeric_table": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["label", "value"],
            },
            "description": "All numeric data points with their units and context",
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any uncertainties, assumptions, or ambiguities in the source",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in summary accuracy (0-1). Lower if source is ambiguous or incomplete.",
        },
    },
    "required": ["summary", "key_facts", "numeric_table", "confidence"],
}


class ChunkSummarizerAgent(BaseAgent):
    """
    Chunk summarization agent using Claude Sonnet 4.6.

    Produces faithful, structured summaries for each chunk while
    preserving numeric data, key facts, and flagging uncertainties.
    """

    role = "chunk_summarization"
    stage = PipelineStage.CHUNK_SUMMARIZATION

    async def summarize_chunk(self, chunk: Chunk) -> ChunkSummary:
        """
        Summarize a single chunk.

        Returns a ChunkSummary with structured data and provenance.
        """
        user_prompt = self._build_prompt(chunk)

        result = await self.call_llm_structured(
            user_prompt=user_prompt,
            schema=CHUNK_SUMMARY_SCHEMA,
        )

        summary = self._parse_result(result, chunk)
        return summary

    async def summarize_chunks(self, chunks: list[Chunk]) -> list[ChunkSummary]:
        """Summarize multiple chunks sequentially."""
        summaries: list[ChunkSummary] = []

        for idx, chunk in enumerate(chunks):
            logger.info(
                "Summarizing chunk %d/%d (%s, %d tokens)",
                idx + 1,
                len(chunks),
                chunk.section_title[:40],
                chunk.token_count,
            )
            try:
                summary = await self.summarize_chunk(chunk)
                summaries.append(summary)
                logger.info(
                    "  -> confidence: %.2f, %d key facts, %d numeric entries",
                    summary.confidence,
                    len(summary.key_facts),
                    len(summary.numeric_table),
                )
            except Exception as exc:
                logger.error(
                    "  -> Failed to summarize chunk %s: %s",
                    chunk.chunk_id,
                    exc,
                )
                # Create a fallback summary with low confidence
                fallback = ChunkSummary(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_file=chunk.source_file,
                    summary=f"[SUMMARIZATION FAILED: {exc}]\n\n{chunk.content[:500]}...",
                    confidence=0.0,
                    section_title=chunk.section_title,
                    heading_path=chunk.heading_path,
                    sequence_index=chunk.sequence_index,
                    provenance=self.create_provenance(
                        chunk_ids=[chunk.chunk_id],
                    ),
                )
                summaries.append(fallback)

        return summaries

    def _build_prompt(self, chunk: Chunk) -> str:
        """Build the user prompt for chunk summarization."""
        context_parts = [
            f"## Source Document: {chunk.source_file}",
            f"## Section: {' > '.join(chunk.heading_path)}"
            if chunk.heading_path
            else "",
            f"## Chunk Position: {chunk.sequence_index}",
            f"## Token Count: {chunk.token_count}",
        ]

        if chunk.contains_tables:
            context_parts.append(
                "## Note: This chunk contains tabular data. Preserve all table values."
            )
        if chunk.contains_images:
            context_parts.append(
                "## Note: This chunk contains image/chart/map descriptions."
            )

        context = "\n".join(p for p in context_parts if p)

        return f"""{context}

## Content to Summarize

{chunk.content}"""

    def _parse_result(self, result: dict[str, Any], chunk: Chunk) -> ChunkSummary:
        """Parse the structured LLM output into a ChunkSummary model."""
        key_facts = [
            KeyFact(
                fact=kf.get("fact", ""),
                category=kf.get("category", "other"),
                entities=kf.get("entities", []),
                source_chunk_id=chunk.chunk_id,
            )
            for kf in result.get("key_facts", [])
        ]

        numeric_table = [
            NumericEntry(
                label=ne.get("label", ""),
                value=ne.get("value", 0),
                unit=ne.get("unit", ""),
                context=ne.get("context", ""),
                source_chunk_id=chunk.chunk_id,
            )
            for ne in result.get("numeric_table", [])
        ]

        return ChunkSummary(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_file=chunk.source_file,
            summary=result.get("summary", ""),
            key_facts=key_facts,
            numeric_table=numeric_table,
            uncertainties=result.get("uncertainties", []),
            confidence=result.get("confidence", 0.5),
            section_title=chunk.section_title,
            heading_path=chunk.heading_path,
            sequence_index=chunk.sequence_index,
            provenance=self.create_provenance(
                chunk_ids=[chunk.chunk_id],
                original_excerpt=chunk.content[:500],
            ),
        )
