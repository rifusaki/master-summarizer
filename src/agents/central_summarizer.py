"""
Central summarization agent (Claude Opus 4.6).

Synthesizes all chunk summaries into a coherent master draft,
following the inferred style guide and targeting the structure
needed for an 80-100 slide presentation.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.models import (
    ChunkSummary,
    DraftSection,
    MasterDraft,
    PipelineStage,
    ReviewResult,
    StyleGuide,
)

logger = logging.getLogger(__name__)


class CentralSummarizerAgent(BaseAgent):
    """
    Central summarization agent using Claude Opus 4.6.

    Synthesizes chunk summaries into a master draft organized
    according to the style guide, with provenance linking.
    """

    role = "central_summarization"
    stage = PipelineStage.CENTRAL_SUMMARIZATION

    async def synthesize(
        self,
        chunk_summaries: list[ChunkSummary],
        style_guide: StyleGuide,
        section_groups: dict[str, list[ChunkSummary]] | None = None,
        version: int = 1,
    ) -> MasterDraft:
        """
        Synthesize chunk summaries into a master draft.

        Because the total content may exceed the model's context window,
        we process section-by-section, then do a final coherence pass.

        Args:
            chunk_summaries: All chunk summaries, sorted by sequence.
            style_guide: The inferred style guide to follow.
            section_groups: Optional pre-grouped summaries by section.
            version: Draft version number.

        Returns:
            Complete MasterDraft with sections and provenance.
        """
        logger.info(
            "Starting central synthesis: %d summaries, version %d",
            len(chunk_summaries),
            version,
        )

        # Group summaries by section if not provided
        if section_groups is None:
            section_groups = self._group_by_section(chunk_summaries)

        # Synthesize each section
        sections: list[DraftSection] = []
        for section_title in style_guide.section_order or list(section_groups.keys()):
            group = section_groups.get(section_title, [])
            if not group:
                # Try fuzzy matching
                group = self._fuzzy_match_section(section_title, section_groups)
            if not group:
                continue

            logger.info(
                "  Synthesizing section: %s (%d summaries)",
                section_title,
                len(group),
            )
            section = await self._synthesize_section(
                section_title=section_title,
                summaries=group,
                style_guide=style_guide,
            )
            sections.append(section)

        # Handle any remaining ungrouped summaries
        used_ids = set()
        for section in sections:
            used_ids.update(section.source_summary_ids)
        remaining = [s for s in chunk_summaries if s.summary_id not in used_ids]
        if remaining:
            logger.info(
                "  %d summaries not matched to sections, adding as appendix",
                len(remaining),
            )
            appendix = await self._synthesize_section(
                section_title="Información Complementaria",
                summaries=remaining,
                style_guide=style_guide,
            )
            sections.append(appendix)

        # Build draft
        draft = MasterDraft(
            version=version,
            sections=sections,
            style_guide_id=style_guide.guide_id,
            total_word_count=sum(len(s.content.split()) for s in sections),
        )

        # Build provenance map
        for section in sections:
            for chunk_id in section.source_chunk_ids:
                if chunk_id not in draft.provenance_map:
                    draft.provenance_map[chunk_id] = []
                draft.provenance_map[chunk_id].append(section.section_id)

        logger.info(
            "Draft v%d complete: %d sections, %d words",
            version,
            len(sections),
            draft.total_word_count,
        )

        return draft

    async def refine_with_feedback(
        self,
        draft: MasterDraft,
        review: ReviewResult,
        style_guide: StyleGuide,
        chunk_summaries: list[ChunkSummary],
    ) -> MasterDraft:
        """
        Refine a draft based on reviewer feedback.

        Applies edit/reject annotations to produce an improved version.
        """
        logger.info(
            "Refining draft v%d with %d annotations (%d edits, %d rejects)",
            draft.version,
            len(review.annotations),
            review.total_edit,
            review.total_reject,
        )

        # Build refinement prompt with reviewer feedback
        edits_and_rejects = [
            a for a in review.annotations if a.verdict.value in ("edit", "reject")
        ]

        if not edits_and_rejects:
            logger.info("No edits or rejects to apply, draft unchanged")
            return draft

        # Compile the current draft text
        draft_text = self._draft_to_text(draft)

        # Compile feedback
        feedback_parts = []
        for ann in edits_and_rejects:
            feedback_parts.append(
                f"- [{ann.verdict.value.upper()}] Section {ann.section_id}, "
                f"para {ann.paragraph_index}: {ann.reason}"
            )
            if ann.suggested_replacement:
                feedback_parts.append(f"  Suggested: {ann.suggested_replacement}")

        feedback_text = "\n".join(feedback_parts)

        # Build style context
        style_context = self._style_guide_to_context(style_guide)

        user_prompt = f"""# Draft Refinement Task

Apply the following reviewer feedback to improve the master draft.
Maintain the same structure and style. Only modify sections that have feedback.
Preserve all source references and provenance markers.

## Style Requirements
{style_context}

## Reviewer Feedback
{feedback_text}

## Risk Register
{self._format_risk_register(review)}

## Current Draft
{draft_text}

Produce the refined draft with the same section structure. For each section, include:
1. Section heading
2. Refined content
3. List of source chunk IDs used

Respond with the complete refined draft."""

        response = await self.call_llm(user_prompt=user_prompt)

        # Parse the refined draft
        refined_sections = self._parse_draft_response(response.get("text", ""), draft)

        new_draft = MasterDraft(
            version=draft.version + 1,
            sections=refined_sections or draft.sections,
            style_guide_id=draft.style_guide_id,
            provenance_map=draft.provenance_map,
            total_word_count=sum(
                len(s.content.split()) for s in (refined_sections or draft.sections)
            ),
        )

        logger.info(
            "Refined draft v%d -> v%d (%d words)",
            draft.version,
            new_draft.version,
            new_draft.total_word_count,
        )

        return new_draft

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _synthesize_section(
        self,
        section_title: str,
        summaries: list[ChunkSummary],
        style_guide: StyleGuide,
    ) -> DraftSection:
        """Synthesize a single section from its chunk summaries."""
        # Build input content
        summaries_text = self._format_summaries(summaries)
        style_context = self._style_guide_to_context(style_guide)

        user_prompt = f"""# Section Synthesis: {section_title}

## Style Requirements
{style_context}

## Target
Write this section for an executive summary document (POT - Plan de Ordenamiento Territorial, Uribia, La Guajira).
The content should be dense enough for 80-100 PowerPoint slides total (this is one section of many).
Use inline source references like [Chunk: <chunk_id>] for provenance.
Write in Spanish. Follow the style guide precisely.

## Source Summaries for This Section

{summaries_text}

Write the complete section "{section_title}" following the style guide.
Include all relevant data points and key facts from the summaries."""

        response = await self.call_llm(user_prompt=user_prompt)
        content = response.get("text", "")

        # Extract chunk IDs referenced
        source_chunk_ids = [s.chunk_id for s in summaries]
        source_summary_ids = [s.summary_id for s in summaries]

        # Calculate average confidence
        avg_confidence = (
            sum(s.confidence for s in summaries) / len(summaries) if summaries else 0
        )

        return DraftSection(
            heading=section_title,
            level=1,
            content=content,
            source_chunk_ids=source_chunk_ids,
            source_summary_ids=source_summary_ids,
            confidence=avg_confidence,
        )

    @staticmethod
    def _group_by_section(
        summaries: list[ChunkSummary],
    ) -> dict[str, list[ChunkSummary]]:
        """Group chunk summaries by their section title."""
        groups: dict[str, list[ChunkSummary]] = {}
        for s in summaries:
            key = s.section_title or (
                s.heading_path[0] if s.heading_path else "General"
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(s)
        # Sort each group by sequence index
        for group in groups.values():
            group.sort(key=lambda x: x.sequence_index)
        return groups

    @staticmethod
    def _fuzzy_match_section(
        target: str,
        groups: dict[str, list[ChunkSummary]],
    ) -> list[ChunkSummary]:
        """Try to match a section title to existing groups."""
        target_lower = target.lower()
        for key, group in groups.items():
            if target_lower in key.lower() or key.lower() in target_lower:
                return group
        return []

    @staticmethod
    def _format_summaries(summaries: list[ChunkSummary]) -> str:
        """Format chunk summaries into a readable block."""
        parts = []
        for s in summaries:
            header = f"### Chunk: {s.chunk_id} (confidence: {s.confidence:.2f})"
            parts.append(header)
            parts.append(s.summary)

            if s.key_facts:
                parts.append("\nKey Facts:")
                for kf in s.key_facts:
                    parts.append(f"  - [{kf.category}] {kf.fact}")

            if s.numeric_table:
                parts.append("\nNumeric Data:")
                for ne in s.numeric_table:
                    parts.append(f"  - {ne.label}: {ne.value} {ne.unit}")

            if s.uncertainties:
                parts.append("\nUncertainties:")
                for u in s.uncertainties:
                    parts.append(f"  - {u}")

            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _style_guide_to_context(guide: StyleGuide) -> str:
        """Format style guide into a concise context string."""
        parts = [
            f"Tone: {guide.tone_description}",
            f"Target Reader: {guide.target_reader}",
            f"Bullet Density: {guide.bullet_density}",
            f"Numeric Formatting: {guide.numeric_formatting}",
            f"Citation Style: {guide.citation_style}",
        ]

        if guide.communication_guidelines:
            parts.append(
                f"\n## Directrices de Comunicación (prioridad alta)\n"
                f"{guide.communication_guidelines}"
            )

        if guide.rules:
            parts.append("\nKey Style Rules:")
            for r in guide.rules:
                if r.priority == "high":
                    parts.append(f"  - [{r.category}] {r.rule}")

        return "\n".join(parts)

    @staticmethod
    def _draft_to_text(draft: MasterDraft) -> str:
        """Convert a MasterDraft to plain text."""
        parts = []
        for section in draft.sections:
            parts.append(f"{'#' * section.level} {section.heading}")
            parts.append(section.content)
            parts.append("")
        return "\n\n".join(parts)

    @staticmethod
    def _format_risk_register(review: ReviewResult) -> str:
        """Format risk register from review."""
        if not review.risk_register:
            return "No risks identified."
        parts = []
        for risk in review.risk_register:
            parts.append(
                f"- [{risk.get('level', 'unknown')}] {risk.get('description', '')}"
            )
        return "\n".join(parts)

    @staticmethod
    def _parse_draft_response(
        text: str, original_draft: MasterDraft
    ) -> list[DraftSection]:
        """
        Parse LLM response text back into DraftSection objects.

        Falls back to original sections if parsing fails.
        """
        if not text.strip():
            return list(original_draft.sections)

        sections: list[DraftSection] = []
        current_heading = ""
        current_content_lines: list[str] = []
        current_level = 1

        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                # Save previous section
                if current_heading and current_content_lines:
                    sections.append(
                        DraftSection(
                            heading=current_heading,
                            level=current_level,
                            content="\n".join(current_content_lines).strip(),
                        )
                    )
                    current_content_lines = []

                # Parse heading level
                level = 0
                for ch in stripped:
                    if ch == "#":
                        level += 1
                    else:
                        break
                current_level = level
                current_heading = stripped.lstrip("#").strip()
            else:
                current_content_lines.append(line)

        # Save last section
        if current_heading and current_content_lines:
            sections.append(
                DraftSection(
                    heading=current_heading,
                    level=current_level,
                    content="\n".join(current_content_lines).strip(),
                )
            )

        # Carry over provenance from original sections
        original_map = {s.heading: s for s in original_draft.sections}
        for section in sections:
            if section.heading in original_map:
                orig = original_map[section.heading]
                section.source_chunk_ids = orig.source_chunk_ids
                section.source_summary_ids = orig.source_summary_ids

        return sections if sections else list(original_draft.sections)
