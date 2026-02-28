"""
Provenance tracking utilities.

Provides functions for building provenance chains across pipeline stages,
validating that every claim links to source chunks, and generating
human-readable provenance reports for the audit trail.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from src.models import (
    AuditEntry,
    ChunkSummary,
    DraftSection,
    MasterDraft,
    PipelineStage,
    ProvenanceRecord,
    ReviewAnnotation,
    ReviewResult,
    SourceLocation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance chain construction
# ---------------------------------------------------------------------------


def build_provenance_chain(
    draft: MasterDraft,
    chunk_summaries: list[ChunkSummary],
) -> dict[str, list[ProvenanceRecord]]:
    """
    Build a full provenance chain from draft sections back to source chunks.

    Returns a mapping of section_id -> list of provenance records showing
    the chain: section -> summary -> chunk -> source file/page.
    """
    # Index summaries by chunk_id for fast lookup
    summary_by_chunk: dict[str, ChunkSummary] = {}
    for s in chunk_summaries:
        summary_by_chunk[s.chunk_id] = s

    chain: dict[str, list[ProvenanceRecord]] = {}

    for section in draft.sections:
        records: list[ProvenanceRecord] = []
        for chunk_id in section.source_chunk_ids:
            summary = summary_by_chunk.get(chunk_id)
            if summary and summary.provenance:
                records.append(summary.provenance)
            else:
                # Create a minimal record even without full provenance
                records.append(
                    ProvenanceRecord(
                        chunk_ids=[chunk_id],
                        agent="unknown",
                        model="unknown",
                    )
                )
        chain[section.section_id] = records

    return chain


def build_section_source_map(
    draft: MasterDraft,
    chunk_summaries: list[ChunkSummary],
) -> dict[str, list[SourceLocation]]:
    """
    Map each draft section back to original source locations (file, page, heading).

    Returns section_id -> list of SourceLocation from the original chunks.
    """
    # Index summaries by chunk_id
    summary_by_chunk: dict[str, ChunkSummary] = {}
    for s in chunk_summaries:
        summary_by_chunk[s.chunk_id] = s

    source_map: dict[str, list[SourceLocation]] = {}

    for section in draft.sections:
        locations: list[SourceLocation] = []
        for chunk_id in section.source_chunk_ids:
            summary = summary_by_chunk.get(chunk_id)
            if summary and summary.provenance:
                locations.extend(summary.provenance.source_locations)
        source_map[section.section_id] = locations

    return source_map


# ---------------------------------------------------------------------------
# Provenance validation
# ---------------------------------------------------------------------------


class ProvenanceValidationResult:
    """Result of provenance validation check."""

    def __init__(self) -> None:
        self.sections_checked: int = 0
        self.sections_with_sources: int = 0
        self.sections_without_sources: list[str] = []
        self.orphan_chunk_ids: list[str] = []  # chunks referenced but no summary found
        self.total_source_links: int = 0

    @property
    def is_valid(self) -> bool:
        """All sections have at least one source link."""
        return len(self.sections_without_sources) == 0

    @property
    def coverage_ratio(self) -> float:
        """Ratio of sections with provenance to total sections."""
        if self.sections_checked == 0:
            return 0.0
        return self.sections_with_sources / self.sections_checked

    def __repr__(self) -> str:
        return (
            f"ProvenanceValidation(valid={self.is_valid}, "
            f"coverage={self.coverage_ratio:.1%}, "
            f"orphans={len(self.orphan_chunk_ids)})"
        )


def validate_provenance(
    draft: MasterDraft,
    chunk_summaries: list[ChunkSummary],
) -> ProvenanceValidationResult:
    """
    Validate that every section in the draft has provenance links
    back to source chunks and summaries.

    Per Project.md: "every claim in the master draft must link to at
    least one chunk ID and show the original text excerpt."
    """
    result = ProvenanceValidationResult()
    summary_ids = {s.chunk_id for s in chunk_summaries}

    for section in draft.sections:
        result.sections_checked += 1

        if not section.source_chunk_ids:
            result.sections_without_sources.append(section.heading)
            continue

        result.sections_with_sources += 1
        result.total_source_links += len(section.source_chunk_ids)

        # Check for orphan references (chunk IDs not in summaries)
        for cid in section.source_chunk_ids:
            if cid not in summary_ids:
                result.orphan_chunk_ids.append(cid)

    if not result.is_valid:
        logger.warning(
            "Provenance validation failed: %d sections without sources: %s",
            len(result.sections_without_sources),
            result.sections_without_sources,
        )
    else:
        logger.info(
            "Provenance validation passed: %d sections, %d total links",
            result.sections_checked,
            result.total_source_links,
        )

    return result


# ---------------------------------------------------------------------------
# Provenance reports
# ---------------------------------------------------------------------------


def generate_provenance_report(
    draft: MasterDraft,
    chunk_summaries: list[ChunkSummary],
    review: ReviewResult | None = None,
) -> str:
    """
    Generate a human-readable provenance report in Markdown.

    Links each draft section to its source chunks, summaries,
    and review annotations.
    """
    summary_by_chunk: dict[str, ChunkSummary] = {}
    for s in chunk_summaries:
        summary_by_chunk[s.chunk_id] = s

    lines = [
        "# Provenance Report",
        f"",
        f"**Draft version:** {draft.version}",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Total sections:** {len(draft.sections)}",
        f"**Total word count:** {draft.total_word_count}",
        "",
    ]

    # Validation summary
    validation = validate_provenance(draft, chunk_summaries)
    lines.extend(
        [
            "## Validation Summary",
            f"- Coverage: {validation.coverage_ratio:.1%}",
            f"- Sections with sources: {validation.sections_with_sources}/{validation.sections_checked}",
            f"- Orphan chunk references: {len(validation.orphan_chunk_ids)}",
            f"- Status: {'PASS' if validation.is_valid else 'FAIL'}",
            "",
        ]
    )

    # Per-section detail
    lines.append("## Section Details\n")

    review_by_section: dict[str, list[ReviewAnnotation]] = defaultdict(list)
    if review:
        for ann in review.annotations:
            review_by_section[ann.section_id].append(ann)

    for section in draft.sections:
        lines.append(f"### {section.heading}")
        lines.append(f"- Confidence: {section.confidence:.2f}")
        lines.append(f"- Source chunks: {len(section.source_chunk_ids)}")
        lines.append(f"- Word count: {len(section.content.split())}")

        # Source files
        source_files: set[str] = set()
        for cid in section.source_chunk_ids:
            s = summary_by_chunk.get(cid)
            if s:
                source_files.add(s.source_file)
        if source_files:
            lines.append(f"- Source files: {', '.join(sorted(source_files))}")

        # Review annotations
        annotations = review_by_section.get(section.section_id, [])
        if annotations:
            lines.append(f"- Review annotations: {len(annotations)}")
            for ann in annotations:
                lines.append(f"  - [{ann.verdict.value.upper()}] {ann.reason[:100]}")

        lines.append("")

    return "\n".join(lines)


def generate_source_file_summary(
    chunk_summaries: list[ChunkSummary],
) -> dict[str, dict[str, Any]]:
    """
    Summarize provenance data grouped by source file.

    Returns per-file stats: chunk count, total confidence,
    key fact count, numeric entry count.
    """
    by_file: dict[str, dict[str, Any]] = {}

    for s in chunk_summaries:
        if s.source_file not in by_file:
            by_file[s.source_file] = {
                "chunk_count": 0,
                "total_confidence": 0.0,
                "key_fact_count": 0,
                "numeric_count": 0,
                "uncertainty_count": 0,
            }
        entry = by_file[s.source_file]
        entry["chunk_count"] += 1
        entry["total_confidence"] += s.confidence
        entry["key_fact_count"] += len(s.key_facts)
        entry["numeric_count"] += len(s.numeric_table)
        entry["uncertainty_count"] += len(s.uncertainties)

    # Compute averages
    for entry in by_file.values():
        count = entry["chunk_count"]
        if count > 0:
            entry["avg_confidence"] = entry["total_confidence"] / count
        else:
            entry["avg_confidence"] = 0.0

    return by_file


# ---------------------------------------------------------------------------
# Audit trail helpers
# ---------------------------------------------------------------------------


def create_stage_audit_entry(
    stage: PipelineStage,
    agent: str,
    model: str,
    action: str,
    input_ids: list[str] | None = None,
    output_ids: list[str] | None = None,
    tokens_input: int = 0,
    tokens_output: int = 0,
    confidence: float | None = None,
    notes: str = "",
) -> AuditEntry:
    """Create an audit entry for a pipeline stage action."""
    return AuditEntry(
        stage=stage,
        agent=agent,
        model=model,
        action=action,
        input_ids=input_ids or [],
        output_ids=output_ids or [],
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        confidence=confidence,
        notes=notes,
    )


def summarize_audit_log(entries: list[AuditEntry]) -> dict[str, Any]:
    """
    Produce aggregate statistics from the audit log.

    Returns per-stage and per-agent token usage and action counts.
    """
    by_stage: dict[str, dict[str, Any]] = {}
    by_agent: dict[str, dict[str, Any]] = {}

    for e in entries:
        stage_key = e.stage.value
        if stage_key not in by_stage:
            by_stage[stage_key] = {
                "actions": 0,
                "tokens_input": 0,
                "tokens_output": 0,
            }
        by_stage[stage_key]["actions"] += 1
        by_stage[stage_key]["tokens_input"] += e.tokens_input
        by_stage[stage_key]["tokens_output"] += e.tokens_output

        if e.agent not in by_agent:
            by_agent[e.agent] = {
                "actions": 0,
                "tokens_input": 0,
                "tokens_output": 0,
            }
        by_agent[e.agent]["actions"] += 1
        by_agent[e.agent]["tokens_input"] += e.tokens_input
        by_agent[e.agent]["tokens_output"] += e.tokens_output

    return {
        "total_entries": len(entries),
        "by_stage": by_stage,
        "by_agent": by_agent,
        "total_tokens": sum(e.tokens_input + e.tokens_output for e in entries),
    }
