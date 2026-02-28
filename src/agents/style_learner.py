"""
Style learning agent (Claude Opus 4.6).

Ingests preprocessed example documents and explicit communication
guidelines to infer a comprehensive set of style rules as a
machine-readable JSON style guide.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.models import (
    DocumentParseResult,
    PipelineStage,
    StyleGuide,
    StyleRule,
)

logger = logging.getLogger(__name__)


STYLE_GUIDE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tone_description": {
            "type": "string",
            "description": "Overall tone description (e.g., formal, technical, accessible)",
        },
        "target_reader": {
            "type": "string",
            "description": "Who the target reader is (e.g., municipal officials, planners)",
        },
        "section_order": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Recommended section/chapter ordering",
        },
        "preferred_headings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Standard heading names used in these documents",
        },
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "tone",
                            "structure",
                            "formatting",
                            "vocabulary",
                            "data_presentation",
                            "citation",
                            "length",
                            "visual",
                        ],
                    },
                    "rule": {"type": "string"},
                    "examples_do": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "examples_dont": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": ["category", "rule", "priority"],
            },
            "description": "List of specific style rules organized by category",
        },
        "bullet_density": {
            "type": "string",
            "description": "How frequently bullet points are used (low/moderate/high)",
        },
        "allowed_abbreviations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Abbreviations that are acceptable to use",
        },
        "numeric_formatting": {
            "type": "string",
            "description": "How numbers should be formatted (decimal separators, thousands, etc.)",
        },
        "citation_style": {
            "type": "string",
            "description": "How sources should be cited or referenced",
        },
        "reviewer_checklist": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short checklist for reviewers to verify style compliance",
        },
    },
    "required": [
        "tone_description",
        "target_reader",
        "section_order",
        "rules",
        "reviewer_checklist",
    ],
}


class StyleLearnerAgent(BaseAgent):
    """
    Style learning agent using Claude Opus 4.6.

    Analyzes example executive summary documents to infer
    a comprehensive, machine-readable style guide.
    """

    role = "style_learning"
    stage = PipelineStage.STYLE_LEARNING

    async def learn_style(
        self,
        example_docs: list[DocumentParseResult],
        communication_guidelines: str = "",
    ) -> StyleGuide:
        """
        Analyze example documents and infer style rules.

        Args:
            example_docs: Parsed example executive summary documents.
            communication_guidelines: Additional explicit style instructions.

        Returns:
            A comprehensive StyleGuide with rules, checklist, and rubric.
        """
        logger.info("Learning style from %d example documents", len(example_docs))

        # Build the analysis prompt
        user_prompt = self._build_prompt(example_docs, communication_guidelines)

        result = await self.call_llm_structured(
            user_prompt=user_prompt,
            schema=STYLE_GUIDE_SCHEMA,
        )

        guide = self._parse_result(result, example_docs)

        logger.info(
            "Style guide generated: %d rules, %d checklist items",
            len(guide.rules),
            len(guide.reviewer_checklist),
        )

        return guide

    def _build_prompt(
        self,
        example_docs: list[DocumentParseResult],
        guidelines: str,
    ) -> str:
        """Build the comprehensive style analysis prompt."""
        parts = [
            "# Style Analysis Task\n",
            "You are analyzing example executive summary documents from Colombian "
            "municipal territorial planning (POT - Plan de Ordenamiento Territorial). "
            "Your goal is to infer a comprehensive set of style rules that can be "
            "used to produce new documents in the exact same style.\n",
            "Analyze the following aspects carefully:\n"
            "1. **Tone and register**: Formal/informal, technical density, accessibility level\n"
            "2. **Document structure**: Section ordering, heading hierarchy, chapter organization\n"
            "3. **Formatting**: Bullet point usage, paragraph length, table formatting\n"
            "4. **Vocabulary**: Preferred terms, technical jargon, abbreviations used\n"
            "5. **Data presentation**: How numbers, statistics, and data are presented\n"
            "6. **Citation style**: How sources, laws, and references are cited\n"
            "7. **Visual elements**: How maps, charts, and images are referenced in text\n"
            "8. **Length and density**: Information density per section\n",
        ]

        if guidelines:
            parts.append(f"# Explicit Communication Guidelines\n\n{guidelines}\n")

        # Add example document content (text only, truncated to fit context)
        for doc in example_docs:
            parts.append(f"\n# Example Document: {doc.source_file}\n")
            parts.append(f"## Structure\n")

            # Add heading structure
            for h in doc.heading_structure[:50]:
                indent = "  " * (h.get("level", 1) - 1)
                parts.append(f"{indent}- {h.get('text', '')}")

            parts.append(f"\n## Content Excerpts\n")

            # Add representative text excerpts (avoid sending everything)
            char_budget = 30000  # per document
            chars_used = 0
            for artifact in doc.artifacts:
                if artifact.artifact_type.value == "text" and artifact.content:
                    if chars_used + len(artifact.content) > char_budget:
                        parts.append(
                            "\n[... content truncated for context limit ...]\n"
                        )
                        break
                    parts.append(artifact.content)
                    chars_used += len(artifact.content)

        return "\n".join(parts)

    def _parse_result(
        self, result: dict[str, Any], example_docs: list[DocumentParseResult]
    ) -> StyleGuide:
        """Parse the structured output into a StyleGuide model."""
        rules = [
            StyleRule(
                category=r.get("category", "other"),
                rule=r.get("rule", ""),
                examples_do=r.get("examples_do", []),
                examples_dont=r.get("examples_dont", []),
                priority=r.get("priority", "medium"),
            )
            for r in result.get("rules", [])
        ]

        return StyleGuide(
            section_order=result.get("section_order", []),
            preferred_headings=result.get("preferred_headings", []),
            rules=rules,
            bullet_density=result.get("bullet_density", ""),
            allowed_abbreviations=result.get("allowed_abbreviations", []),
            numeric_formatting=result.get("numeric_formatting", ""),
            citation_style=result.get("citation_style", ""),
            tone_description=result.get("tone_description", ""),
            target_reader=result.get("target_reader", ""),
            reviewer_checklist=result.get("reviewer_checklist", []),
            source_documents=[d.source_file for d in example_docs],
            provenance=self.create_provenance(
                chunk_ids=[d.document_id for d in example_docs],
            ),
        )
