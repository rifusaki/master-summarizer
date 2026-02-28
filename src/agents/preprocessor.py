"""
Preprocessing agent (Gemini 3.1 Pro).

Processes extracted images, charts, maps, and complex tables
through multimodal LLM to generate detailed text descriptions.
No summarization - only canonicalized machine-friendly representations.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.models import (
    ArtifactType,
    DocumentParseResult,
    NormalizedArtifact,
    PipelineStage,
)

logger = logging.getLogger(__name__)


IMAGE_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "Detailed description of the visual content",
        },
        "content_type": {
            "type": "string",
            "enum": ["chart", "map", "photograph", "diagram", "table_image", "other"],
            "description": "Type of visual content",
        },
        "extracted_data": {
            "type": "object",
            "description": "Any data extracted from charts/tables (labels, values, etc.)",
            "properties": {
                "title": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}},
                "values": {"type": "array", "items": {"type": "string"}},
                "units": {"type": "string"},
                "legend": {"type": "array", "items": {"type": "string"}},
                "annotations": {"type": "array", "items": {"type": "string"}},
            },
        },
        "geographic_info": {
            "type": "object",
            "description": "Geographic information if this is a map",
            "properties": {
                "region": {"type": "string"},
                "scale": {"type": "string"},
                "legend_items": {"type": "array", "items": {"type": "string"}},
                "notable_features": {"type": "array", "items": {"type": "string"}},
            },
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the description accuracy (0-1)",
        },
    },
    "required": ["description", "content_type", "confidence"],
}


class PreprocessorAgent(BaseAgent):
    """
    Multimodal preprocessing agent using Gemini 3.1 Pro.

    Processes images, charts, and maps extracted from documents,
    generating detailed text descriptions for downstream processing.
    """

    role = "preprocessing"
    stage = PipelineStage.PREPROCESSING

    async def process_document(
        self, parse_result: DocumentParseResult
    ) -> DocumentParseResult:
        """
        Process all image artifacts in a parsed document through
        multimodal LLM to generate descriptions.

        Modifies artifacts in-place, enriching image content fields
        with detailed descriptions.
        """
        image_artifacts = [
            a
            for a in parse_result.artifacts
            if a.artifact_type == ArtifactType.IMAGE and a.image_base64
        ]

        if not image_artifacts:
            logger.info("No images to preprocess in %s", parse_result.source_file)
            return parse_result

        logger.info(
            "Preprocessing %d images from %s",
            len(image_artifacts),
            parse_result.source_file,
        )

        for idx, artifact in enumerate(image_artifacts):
            try:
                result = await self._describe_image(artifact)
                self._apply_description(artifact, result)
                logger.info(
                    "  [%d/%d] Described image: %s (confidence: %.2f)",
                    idx + 1,
                    len(image_artifacts),
                    result.get("content_type", "unknown"),
                    result.get("confidence", 0),
                )
            except Exception as exc:
                logger.warning(
                    "  [%d/%d] Failed to describe image: %s",
                    idx + 1,
                    len(image_artifacts),
                    exc,
                )
                artifact.confidence = 0.3

        return parse_result

    async def _describe_image(self, artifact: NormalizedArtifact) -> dict[str, Any]:
        """Send an image to Gemini for description."""
        system_prompt = self.load_prompt()

        user_prompt = (
            "Analyze this image from a Colombian municipal planning document "
            "(POT - Plan de Ordenamiento Territorial, Uribia, La Guajira). "
            "Provide a detailed description of what the image contains. "
            "If it's a chart or graph, extract all data values, labels, and units. "
            "If it's a map, describe the geographic features, legend, scale, and annotations. "
            "If it's a table rendered as an image, extract all cell values. "
            "Respond in Spanish if the original content is in Spanish."
        )

        # Determine media type
        fmt = artifact.metadata.get("format", "png")
        media_type_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "tiff": "image/tiff",
        }
        media_type = media_type_map.get(fmt, "image/png")

        result = await self.call_llm_structured(
            user_prompt=user_prompt,
            schema=IMAGE_DESCRIPTION_SCHEMA,
            system_prompt=system_prompt,
            image_base64=artifact.image_base64,
        )

        return result

    @staticmethod
    def _apply_description(
        artifact: NormalizedArtifact, result: dict[str, Any]
    ) -> None:
        """Apply the LLM description to the artifact."""
        description = result.get("description", "")
        content_type = result.get("content_type", "other")
        confidence = result.get("confidence", 0.5)

        # Map content type to artifact type
        type_map = {
            "chart": ArtifactType.CHART,
            "map": ArtifactType.MAP,
            "table_image": ArtifactType.TABLE,
        }
        if content_type in type_map:
            artifact.artifact_type = type_map[content_type]

        # Build rich content string
        parts = [f"[{content_type.upper()}]", description]

        # Add extracted data if available
        extracted = result.get("extracted_data", {})
        if extracted:
            if extracted.get("title"):
                parts.append(f"Title: {extracted['title']}")
            if extracted.get("labels"):
                parts.append(f"Labels: {', '.join(extracted['labels'])}")
            if extracted.get("values"):
                parts.append(f"Values: {', '.join(extracted['values'])}")
            if extracted.get("units"):
                parts.append(f"Units: {extracted['units']}")

        # Add geographic info if available
        geo = result.get("geographic_info", {})
        if geo:
            if geo.get("region"):
                parts.append(f"Region: {geo['region']}")
            if geo.get("scale"):
                parts.append(f"Scale: {geo['scale']}")
            if geo.get("notable_features"):
                parts.append(f"Features: {', '.join(geo['notable_features'])}")

        artifact.content = "\n".join(parts)
        artifact.confidence = confidence
        artifact.metadata["content_type"] = content_type
        artifact.metadata["extracted_data"] = extracted
        if geo:
            artifact.metadata["geographic_info"] = geo
