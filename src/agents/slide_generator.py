"""
Slide outline generation agent (GPT-5.3 Codex).

Converts the final master draft into structured slide-by-slide
outlines with titles, bullets, visual suggestions, and source refs.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.models import (
    MasterDraft,
    PipelineStage,
    SlideOutline,
    SlideOutlineSet,
    StyleGuide,
)
from src.config import pipeline_config

logger = logging.getLogger(__name__)


SLIDE_OUTLINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "slides": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slide_number": {"type": "integer"},
                    "title": {"type": "string"},
                    "bullets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 5,
                    },
                    "suggested_visual": {
                        "type": "string",
                        "enum": [
                            "table",
                            "bar_chart",
                            "line_chart",
                            "pie_chart",
                            "map",
                            "photograph",
                            "diagram",
                            "infographic",
                            "none",
                        ],
                    },
                    "visual_description": {"type": "string"},
                    "source_references": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "speaker_notes": {"type": "string"},
                },
                "required": ["slide_number", "title", "bullets"],
            },
        },
    },
    "required": ["slides"],
}


class SlideGeneratorAgent(BaseAgent):
    """
    Slide outline generator using GPT-5.3 Codex.

    Converts the final master draft into a structured set of
    80-100 slide outlines ready for PowerPoint creation.
    """

    role = "slide_generation"
    stage = PipelineStage.SLIDE_GENERATION

    async def generate_outlines(
        self,
        draft: MasterDraft,
        style_guide: StyleGuide,
    ) -> SlideOutlineSet:
        """
        Generate slide outlines from the master draft.

        Processes the draft section by section to stay within
        context limits, then assembles the full outline set.
        """
        logger.info(
            "Generating slide outlines from draft v%d (%d sections)",
            draft.version,
            len(draft.sections),
        )

        config = pipeline_config
        min_slides = config.target_slide_count_min
        max_slides = config.target_slide_count_max

        # Estimate slides per section based on content volume
        total_words = draft.total_word_count or 1
        slide_allocations = self._allocate_slides(draft, min_slides, max_slides)

        all_slides: list[SlideOutline] = []
        slide_counter = 1

        # Title slide
        all_slides.append(
            SlideOutline(
                slide_number=slide_counter,
                title="Plan de Ordenamiento Territorial - Uribia, La Guajira",
                bullets=[
                    "Resumen Ejecutivo",
                    "Documento Técnico de Soporte",
                ],
                suggested_visual="photograph",
                visual_description="Vista panorámica del municipio de Uribia",
                speaker_notes="Presentación del Plan de Ordenamiento Territorial",
            )
        )
        slide_counter += 1

        # Table of contents slide
        all_slides.append(
            SlideOutline(
                slide_number=slide_counter,
                title="Contenido",
                bullets=[s.heading for s in draft.sections[:10]],
                suggested_visual="none",
                speaker_notes="Estructura del documento",
            )
        )
        slide_counter += 1

        # Process each section
        for section, n_slides in zip(draft.sections, slide_allocations):
            if n_slides <= 0:
                continue

            logger.info(
                "  Section '%s': %d slides allocated",
                section.heading[:40],
                n_slides,
            )

            section_slides = await self._generate_section_slides(
                section_heading=section.heading,
                section_content=section.content,
                style_guide=style_guide,
                n_slides=n_slides,
                start_number=slide_counter,
            )

            all_slides.extend(section_slides)
            slide_counter += len(section_slides)

        outline_set = SlideOutlineSet(
            slides=all_slides,
            total_slides=len(all_slides),
            draft_id=draft.draft_id,
            style_guide_id=style_guide.guide_id,
        )

        logger.info(
            "Generated %d slide outlines (target: %d-%d)",
            outline_set.total_slides,
            min_slides,
            max_slides,
        )

        return outline_set

    async def _generate_section_slides(
        self,
        section_heading: str,
        section_content: str,
        style_guide: StyleGuide,
        n_slides: int,
        start_number: int,
    ) -> list[SlideOutline]:
        """Generate slides for a single section."""
        user_prompt = f"""# Slide Generation Task

Convert the following section into exactly {n_slides} PowerPoint slide outlines.

## Constraints
- Maximum {pipeline_config.max_bullets_per_slide} bullets per slide
- Maximum {pipeline_config.max_words_per_slide} words per bullet point
- Each slide should have a clear, descriptive title
- Suggest appropriate visuals (tables, charts, maps) where the content warrants them
- Include speaker notes with additional context
- Slide numbers start at {start_number}
- Write in Spanish

## Style
Tone: {style_guide.tone_description}
Target Reader: {style_guide.target_reader}

## Section: {section_heading}

{section_content}

Generate {n_slides} slides for this section."""

        result = await self.call_llm_structured(
            user_prompt=user_prompt,
            schema=SLIDE_OUTLINE_SCHEMA,
        )

        slides = []
        for slide_data in result.get("slides", []):
            slides.append(
                SlideOutline(
                    slide_number=slide_data.get(
                        "slide_number", start_number + len(slides)
                    ),
                    title=slide_data.get("title", ""),
                    bullets=slide_data.get("bullets", [])[
                        : pipeline_config.max_bullets_per_slide
                    ],
                    suggested_visual=slide_data.get("suggested_visual", "none"),
                    visual_description=slide_data.get("visual_description", ""),
                    source_references=slide_data.get("source_references", []),
                    speaker_notes=slide_data.get("speaker_notes", ""),
                    word_count=sum(
                        len(b.split()) for b in slide_data.get("bullets", [])
                    ),
                )
            )

        return slides

    @staticmethod
    def _allocate_slides(
        draft: MasterDraft,
        min_slides: int,
        max_slides: int,
    ) -> list[int]:
        """
        Allocate slide count per section proportional to content volume.

        Reserves 2 slides for title + TOC, distributes the rest
        proportionally by word count.
        """
        available = max_slides - 2  # title + TOC
        total_words = sum(len(s.content.split()) for s in draft.sections)

        if total_words == 0:
            # Equal distribution
            per_section = available // max(len(draft.sections), 1)
            return [per_section] * len(draft.sections)

        allocations = []
        for section in draft.sections:
            words = len(section.content.split())
            proportion = words / total_words
            n_slides = max(1, round(proportion * available))
            allocations.append(n_slides)

        # Adjust to hit target range
        total = sum(allocations)
        if total > available:
            # Scale down proportionally
            factor = available / total
            allocations = [max(1, round(a * factor)) for a in allocations]
        elif total < min_slides - 2:
            # Scale up
            factor = (min_slides - 2) / max(total, 1)
            allocations = [max(1, round(a * factor)) for a in allocations]

        return allocations
