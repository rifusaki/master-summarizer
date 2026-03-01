"""
Chunk summarization agent (Claude Sonnet 4.6, with fallback).

Produces faithful, length-limited summaries of individual chunks
while preserving numeric values, key facts, and provenance links.

Resilience design:
- Uses call_llm_structured_resilient for automatic retry + fallback.
- Per-chunk status tracking so failed chunks can be retried without
  re-processing successful ones.
- Raises ModelExhaustionError when all models in the fallback chain
  are exhausted, after saving all progress completed so far.
- on_chunk_done callback for incremental persistence.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from src.agents.base import BaseAgent, ModelExhaustionError
from src.models import (
    Chunk,
    ChunkSummary,
    ItemStatus,
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
    Chunk summarization agent using Claude Sonnet 4.6 (with fallback).

    Produces faithful, structured summaries for each chunk while
    preserving numeric data, key facts, and flagging uncertainties.

    Resilient features:
    - Automatic retry with fallback model chain via call_llm_structured_resilient.
    - Per-chunk status tracking (success/failed) for resume support.
    - Raises ModelExhaustionError when all models are exhausted.
    - on_chunk_done callback for per-chunk incremental saves.
    """

    role = "chunk_summarization"
    stage = PipelineStage.CHUNK_SUMMARIZATION

    async def summarize_chunk(self, chunk: Chunk) -> tuple[ChunkSummary, str]:
        """
        Summarize a single chunk with retry/fallback.

        Returns:
            Tuple of (ChunkSummary, status_string).
            status_string is ItemStatus.SUCCESS or an ItemStatus.FAILED_* value.

        Raises:
            ModelExhaustionError: When all models in the chain are exhausted.
        """
        user_prompt = self._build_prompt(chunk)

        result, model_used = await self.call_llm_structured_resilient(
            user_prompt=user_prompt,
            schema=CHUNK_SUMMARY_SCHEMA,
            item_id=chunk.chunk_id[:12],
        )

        summary = self._parse_result(result, chunk, model_used)
        return summary, ItemStatus.SUCCESS

    async def summarize_chunks(
        self,
        chunks: list[Chunk],
        *,
        on_chunk_done: Callable[[ChunkSummary, str], Awaitable[None]] | None = None,
    ) -> list[ChunkSummary]:
        """
        Summarize multiple chunks sequentially with resilience.

        Args:
            chunks: List of chunks to summarize.
            on_chunk_done: Async callback called after each chunk attempt
                with (summary, status). The caller can persist the summary
                to disk immediately.

        Returns:
            List of ChunkSummary objects for all successfully summarized
            chunks (and fallback summaries for non-exhaustion failures).

        Raises:
            ModelExhaustionError: When all models are exhausted. All progress
                up to this point has already been saved via on_chunk_done.
        """
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
                summary, status = await self.summarize_chunk(chunk)
                summaries.append(summary)
                logger.info(
                    "  -> confidence: %.2f, %d key facts, %d numeric entries",
                    summary.confidence,
                    len(summary.key_facts),
                    len(summary.numeric_table),
                )

                if on_chunk_done is not None:
                    try:
                        await on_chunk_done(summary, status)
                    except Exception as cb_exc:
                        logger.warning("on_chunk_done callback failed: %s", cb_exc)

            except ModelExhaustionError as exc:
                # Update remaining count and re-raise
                exc.items_completed = len(summaries)
                exc.items_remaining = len(chunks) - idx
                logger.error(
                    "Model exhaustion at chunk %d/%d. "
                    "%d summaries completed, %d remaining.",
                    idx + 1,
                    len(chunks),
                    exc.items_completed,
                    exc.items_remaining,
                )
                raise

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
                status = ItemStatus.FAILED_OTHER

                if on_chunk_done is not None:
                    try:
                        await on_chunk_done(fallback, status)
                    except Exception as cb_exc:
                        logger.warning("on_chunk_done callback failed: %s", cb_exc)

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

    def _parse_result(
        self, result: dict[str, Any], chunk: Chunk, model_used: str = ""
    ) -> ChunkSummary:
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

        provenance = self.create_provenance(
            chunk_ids=[chunk.chunk_id],
            original_excerpt=chunk.content[:500],
        )
        # Record actual model used (may differ from primary if fallback was used)
        if model_used:
            provenance.model = model_used

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
            provenance=provenance,
        )
