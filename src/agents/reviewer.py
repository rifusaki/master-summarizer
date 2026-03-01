"""
Secondary reviewer agent (GPT-5.3 Codex, with fallback).

Performs automated cross-checks on the master draft: fact consistency,
numeric reconciliation, tone verification, and produces a prioritized
list of edits with Accept/Edit/Reject per paragraph.

Resilience design:
- Uses call_llm_structured_resilient for retry + fallback.
- Raises ModelExhaustionError when all models are exhausted,
  so the pipeline can save state and stop cleanly.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent, ModelExhaustionError
from src.models import (
    ChunkSummary,
    MasterDraft,
    PipelineStage,
    ReviewAnnotation,
    ReviewResult,
    ReviewVerdict,
    StyleGuide,
)

logger = logging.getLogger(__name__)


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section_heading": {"type": "string"},
                    "paragraph_index": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": ["accept", "edit", "reject"],
                    },
                    "reason": {"type": "string"},
                    "suggested_replacement": {"type": "string"},
                    "fact_check_passed": {"type": "boolean"},
                    "numeric_check_passed": {"type": "boolean"},
                    "tone_check_passed": {"type": "boolean"},
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": ["section_heading", "verdict", "reason"],
            },
        },
        "risk_register": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "description": {"type": "string"},
                    "affected_sections": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["level", "description"],
            },
        },
        "overall_confidence": {
            "type": "number",
            "description": "Overall confidence in draft quality (0-1)",
        },
        "reviewer_notes": {
            "type": "string",
            "description": "General reviewer observations and recommendations",
        },
    },
    "required": ["annotations", "overall_confidence"],
}


class ReviewerAgent(BaseAgent):
    """
    Secondary reviewer using GPT-5.3 Codex (with fallback).

    Performs systematic cross-checks on the master draft against
    source summaries and style guide, producing annotated feedback.

    Resilient features:
    - Automatic retry with fallback model chain via call_llm_structured_resilient.
    - Raises ModelExhaustionError so the pipeline can stop cleanly.
    """

    role = "reviewer"
    stage = PipelineStage.REVIEW

    async def review_draft(
        self,
        draft: MasterDraft,
        style_guide: StyleGuide,
        chunk_summaries: list[ChunkSummary],
    ) -> ReviewResult:
        """
        Review the master draft against source summaries and style guide.

        Returns annotated review with Accept/Edit/Reject per paragraph.

        Raises:
            ModelExhaustionError: When all models are exhausted.
        """
        logger.info(
            "Reviewing draft v%d (%d sections, %d words)",
            draft.version,
            len(draft.sections),
            draft.total_word_count,
        )

        user_prompt = self._build_prompt(draft, style_guide, chunk_summaries)

        result, model_used = await self.call_llm_structured_resilient(
            user_prompt=user_prompt,
            schema=REVIEW_SCHEMA,
            item_id=f"review_v{draft.version}",
        )

        review = self._parse_result(result, draft, model_used)

        logger.info(
            "Review complete: %d annotations (A:%d E:%d R:%d), "
            "overall confidence: %.2f (model: %s)",
            len(review.annotations),
            review.total_accept,
            review.total_edit,
            review.total_reject,
            review.overall_confidence,
            model_used,
        )

        return review

    def _build_prompt(
        self,
        draft: MasterDraft,
        style_guide: StyleGuide,
        chunk_summaries: list[ChunkSummary],
    ) -> str:
        """Build the comprehensive review prompt."""
        # Draft text
        draft_parts = []
        for section in draft.sections:
            draft_parts.append(f"## {section.heading}")
            draft_parts.append(section.content)
            draft_parts.append(
                f"[Source chunks: {', '.join(section.source_chunk_ids[:5])}]"
            )
            draft_parts.append("")
        draft_text = "\n".join(draft_parts)

        # Style checklist
        checklist = "\n".join(f"  - {item}" for item in style_guide.reviewer_checklist)

        # Key facts from source summaries for cross-checking
        facts_parts = []
        for s in chunk_summaries:
            if s.key_facts:
                for kf in s.key_facts[:3]:  # Limit per summary
                    facts_parts.append(f"- [{s.source_file}] {kf.fact}")
            if s.numeric_table:
                for ne in s.numeric_table[:3]:
                    facts_parts.append(
                        f"- [{s.source_file}] {ne.label}: {ne.value} {ne.unit}"
                    )
        # Limit total facts to stay within context
        facts_text = "\n".join(facts_parts[:200])

        return f"""# Draft Review Task

Review the following executive summary draft for a Colombian municipal 
territorial plan (POT Uribia, La Guajira). Perform these checks:

1. **Fact consistency**: Verify claims against source data
2. **Numeric reconciliation**: Check numbers match source summaries
3. **Tone compliance**: Verify adherence to style guide
4. **Completeness**: Flag any major topics missing
5. **Contradictions**: Identify any internal contradictions
6. **Legal/geographic claims**: Flag any that need human verification

## Style Checklist
{checklist}

## High-Priority Style Rules
Tone: {style_guide.tone_description}
Target Reader: {style_guide.target_reader}

## Source Facts for Cross-Checking
{facts_text}

## Draft to Review (v{draft.version})
{draft_text}

For each paragraph, provide: Accept (correct as-is), Edit (needs changes), 
or Reject (factually wrong or severely off-style).
Include a risk register for any high-risk claims."""

    def _parse_result(
        self,
        result: dict[str, Any],
        draft: MasterDraft,
        model_used: str = "",
    ) -> ReviewResult:
        """Parse structured review output into ReviewResult."""
        # Map section headings to IDs
        heading_to_id = {s.heading: s.section_id for s in draft.sections}

        annotations = []
        for ann_data in result.get("annotations", []):
            heading = ann_data.get("section_heading", "")
            section_id = heading_to_id.get(heading, heading) or ""

            verdict_str = ann_data.get("verdict", "accept")
            try:
                verdict = ReviewVerdict(verdict_str)
            except ValueError:
                verdict = ReviewVerdict.ACCEPT

            annotations.append(
                ReviewAnnotation(
                    section_id=section_id,
                    paragraph_index=ann_data.get("paragraph_index", 0),
                    verdict=verdict,
                    reason=ann_data.get("reason", ""),
                    suggested_replacement=ann_data.get("suggested_replacement", ""),
                    fact_check_passed=ann_data.get("fact_check_passed", True),
                    numeric_check_passed=ann_data.get("numeric_check_passed", True),
                    tone_check_passed=ann_data.get("tone_check_passed", True),
                    risk_level=ann_data.get("risk_level", "low"),
                )
            )

        total_accept = sum(1 for a in annotations if a.verdict == ReviewVerdict.ACCEPT)
        total_edit = sum(1 for a in annotations if a.verdict == ReviewVerdict.EDIT)
        total_reject = sum(1 for a in annotations if a.verdict == ReviewVerdict.REJECT)

        provenance = self.create_provenance()
        if model_used:
            provenance.model = model_used

        return ReviewResult(
            draft_id=draft.draft_id,
            draft_version=draft.version,
            annotations=annotations,
            total_accept=total_accept,
            total_edit=total_edit,
            total_reject=total_reject,
            risk_register=result.get("risk_register", []),
            overall_confidence=result.get("overall_confidence", 0.0),
            reviewer_notes=result.get("reviewer_notes", ""),
            provenance=provenance,
        )
