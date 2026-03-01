"""
Preprocessing agent (Gemini 3.1 Pro, with fallback to Gemini 3 Pro).

Processes extracted images, charts, maps, and complex tables
through multimodal LLM to generate detailed text descriptions.
No summarization — only canonicalized machine-friendly representations.

Resilience design:
- Fresh session per image: prevents session context bloat which causes
  timeouts when a long-lived session accumulates many messages.
- Per-image timeout retries: retries once with a fresh session before
  marking as failed_timeout.
- Rate-limit detection: counts consecutive 429s; once the streak hits
  the configured threshold, prompts the user to switch to the fallback
  Gemini model for the remainder of the run.
- Status metadata on every artifact: preprocess_status, preprocess_model,
  preprocess_attempts, preprocess_error, preprocess_run_id.
- on_image_done callback: allows the caller (main.py) to persist the
  updated DocumentParseResult to disk after every single image, so no
  progress is lost on interruption.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from src.agents.base import BaseAgent, ModelExhaustionError
from src.config import FALLBACK_MODELS, ModelConfig, pipeline_config
from src.models import (
    ArtifactType,
    DocumentParseResult,
    ImagePreprocessStatus,
    NormalizedArtifact,
    PipelineStage,
)
from src.opencode_client import OpenCodeRateLimitError, OpenCodeTimeoutError

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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PreprocessorAgent(BaseAgent):
    """
    Multimodal preprocessing agent using Gemini 3.1 Pro (primary).

    Resilient design:
    - Uses a fresh OpenCode session per image to prevent context bloat.
    - Retries timeouts once with a fresh session before marking failed.
    - Detects consecutive 429 rate-limit errors and prompts the user to
      switch to the configured fallback Gemini model.
    - Stamps every artifact with preprocess_status metadata so the pipeline
      can resume or retry only failed images on the next run.
    - Accepts an optional async on_image_done callback that is called after
      each image (success or failure) so the caller can persist to disk
      immediately.
    """

    role = "preprocessing"
    stage = PipelineStage.PREPROCESSING

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        # Active model may switch to fallback during the run
        self._active_model: ModelConfig = self.model
        self._fallback_models: list[ModelConfig] = FALLBACK_MODELS.get(self.role, [])
        self._fallback_index: int = 0  # which fallback we're currently on
        self._rate_limit_streak: int = 0
        self._fallback_confirmed: bool = False  # user said yes to fallback
        self._exhausted: bool = False  # all models exhausted, stop processing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_document(
        self,
        parse_result: DocumentParseResult,
        *,
        run_id: str = "",
        target_artifact_ids: set[str] | None = None,
        on_image_done: Callable[[DocumentParseResult], Awaitable[None]] | None = None,
    ) -> DocumentParseResult:
        """
        Process image artifacts in a parsed document through multimodal LLM.

        Args:
            parse_result: The parsed document with image artifacts.
            run_id: Optional pipeline run ID stamped on artifact metadata.
            target_artifact_ids: If provided, only process these artifact IDs.
                                  Used for failed-only retry mode.
            on_image_done: Async callback called after each image attempt.
                           Receives the (mutated) parse_result so the caller
                           can persist it to disk immediately.

        Returns:
            The same parse_result with artifact content/metadata updated.

        Raises:
            ModelExhaustionError: When all models (primary + fallbacks) are
                exhausted. Progress up to this point has already been saved
                via on_image_done callbacks.
        """
        image_artifacts = [
            a
            for a in parse_result.artifacts
            if a.artifact_type == ArtifactType.IMAGE
            and a.image_base64
            and (target_artifact_ids is None or a.artifact_id in target_artifact_ids)
        ]

        if not image_artifacts:
            logger.info("No images to preprocess in %s", parse_result.source_file)
            return parse_result

        logger.info(
            "Preprocessing %d images from %s",
            len(image_artifacts),
            parse_result.source_file,
        )

        completed = 0
        for idx, artifact in enumerate(image_artifacts):
            # If models are exhausted, mark remaining images and raise
            if self._exhausted:
                self._stamp_metadata(
                    artifact,
                    status=ImagePreprocessStatus.FAILED_RATE_LIMIT,
                    model=self._active_model.full_id,
                    attempts=0,
                    run_id=run_id,
                    error="Model exhaustion: all models in fallback chain exhausted",
                )
                if on_image_done is not None:
                    try:
                        await on_image_done(parse_result)
                    except Exception as cb_exc:
                        logger.warning("on_image_done callback failed: %s", cb_exc)
                continue

            await self._process_one_image(
                artifact=artifact,
                idx=idx,
                total=len(image_artifacts),
                run_id=run_id,
            )

            if (
                artifact.metadata.get("preprocess_status")
                == ImagePreprocessStatus.SUCCESS
            ):
                completed += 1

            if on_image_done is not None:
                try:
                    await on_image_done(parse_result)
                except Exception as cb_exc:
                    logger.warning("on_image_done callback failed: %s", cb_exc)

        # After processing all images, if we're exhausted, raise so the
        # pipeline can stop cleanly
        if self._exhausted:
            models_tried = [self.model.full_id] + [
                m.full_id for m in self._fallback_models[: self._fallback_index]
            ]
            remaining = len(image_artifacts) - completed
            raise ModelExhaustionError(
                role=self.role,
                models_tried=models_tried,
                last_error="Rate limit / timeout exhaustion during image preprocessing",
                items_completed=completed,
                items_remaining=remaining,
            )

        return parse_result

    # ------------------------------------------------------------------
    # Per-image logic
    # ------------------------------------------------------------------

    async def _process_one_image(
        self,
        artifact: NormalizedArtifact,
        idx: int,
        total: int,
        run_id: str,
    ) -> None:
        """
        Attempt to describe a single image with retry and fallback logic.

        Timeout retry: retries once with a fresh session.
        Rate-limit handling: increments streak; if streak >= threshold,
            prompts user to switch to fallback model (once per run).
        """
        max_timeout_retries = pipeline_config.preprocessing_timeout_retries
        attempts = 0
        last_error: str = ""
        last_status = ImagePreprocessStatus.PENDING

        # --- timeout retry loop ---
        for attempt in range(max_timeout_retries + 1):
            attempts = attempt + 1
            try:
                result = await self._describe_image(artifact, self._active_model)
                self._apply_description(artifact, result)
                self._stamp_metadata(
                    artifact,
                    status=ImagePreprocessStatus.SUCCESS,
                    model=self._active_model.full_id,
                    attempts=attempts,
                    run_id=run_id,
                )
                self._rate_limit_streak = 0  # reset on success
                logger.info(
                    "  [%d/%d] Described image: %s (confidence: %.2f)",
                    idx + 1,
                    total,
                    result.get("content_type", "unknown"),
                    result.get("confidence", 0),
                )
                return

            except OpenCodeTimeoutError as exc:
                last_error = str(exc)
                last_status = ImagePreprocessStatus.FAILED_TIMEOUT
                if attempt < max_timeout_retries:
                    logger.warning(
                        "  [%d/%d] Timeout on attempt %d/%d, retrying with fresh session...",
                        idx + 1,
                        total,
                        attempt + 1,
                        max_timeout_retries + 1,
                    )
                    await asyncio.sleep(2)
                    continue
                # All timeout retries exhausted
                logger.warning(
                    "  [%d/%d] Failed (timeout after %d attempts): %s",
                    idx + 1,
                    total,
                    attempts,
                    exc,
                )
                break

            except OpenCodeRateLimitError as exc:
                last_error = str(exc)
                last_status = ImagePreprocessStatus.FAILED_RATE_LIMIT
                self._rate_limit_streak += 1
                logger.warning(
                    "  [%d/%d] Rate limited (429), streak=%d",
                    idx + 1,
                    total,
                    self._rate_limit_streak,
                )

                # Check if we should switch to fallback model
                if await self._maybe_switch_to_fallback(idx, total):
                    # Try once more with the new fallback model
                    try:
                        result = await self._describe_image(
                            artifact, self._active_model
                        )
                        self._apply_description(artifact, result)
                        self._stamp_metadata(
                            artifact,
                            status=ImagePreprocessStatus.SUCCESS,
                            model=self._active_model.full_id,
                            attempts=attempts + 1,
                            run_id=run_id,
                        )
                        self._rate_limit_streak = 0
                        logger.info(
                            "  [%d/%d] Described via fallback model %s: %s (confidence: %.2f)",
                            idx + 1,
                            total,
                            self._active_model.full_id,
                            result.get("content_type", "unknown"),
                            result.get("confidence", 0),
                        )
                        return
                    except Exception as fallback_exc:
                        last_error = str(fallback_exc)
                        logger.warning(
                            "  [%d/%d] Fallback model also failed: %s",
                            idx + 1,
                            total,
                            fallback_exc,
                        )
                break  # don't retry on rate limit (we've already handled it)

            except Exception as exc:
                last_error = str(exc)
                last_status = ImagePreprocessStatus.FAILED_OTHER
                logger.warning(
                    "  [%d/%d] Failed: %s",
                    idx + 1,
                    total,
                    exc,
                )
                break

        # Mark artifact as failed
        artifact.confidence = 0.3
        self._stamp_metadata(
            artifact,
            status=last_status,
            model=self._active_model.full_id,
            attempts=attempts,
            run_id=run_id,
            error=last_error,
        )

    async def _maybe_switch_to_fallback(self, idx: int, total: int) -> bool:
        """
        Check if we should switch to the next fallback model.

        Returns True if a fallback model was activated (either auto or user-confirmed).
        Returns False if no fallback is available or user declined.
        Sets self._exhausted = True when no more fallback options exist.
        """
        threshold = pipeline_config.preprocessing_rate_limit_streak_threshold
        if self._rate_limit_streak < threshold:
            return False

        if self._fallback_index >= len(self._fallback_models):
            logger.warning(
                "Rate limit streak=%d and no fallback models remaining. "
                "All models exhausted — will stop after saving progress.",
                self._rate_limit_streak,
            )
            self._exhausted = True
            return False

        next_fallback = self._fallback_models[self._fallback_index]

        if (
            not pipeline_config.preprocessing_confirm_fallback
            or self._fallback_confirmed
        ):
            # Auto-switch (or already confirmed for this run)
            self._active_model = next_fallback
            self._fallback_index += 1
            self._rate_limit_streak = 0
            logger.info(
                "  [%d/%d] Switched to fallback model: %s",
                idx + 1,
                total,
                self._active_model.full_id,
            )
            return True

        # Prompt user once
        print(
            f"\n[PREPROCESSOR] Detected {self._rate_limit_streak} consecutive "
            f"rate-limit errors on {self.model.full_id}.\n"
            f"Switch remaining images to fallback model "
            f"'{next_fallback.full_id}'? [y/n]: ",
            end="",
            flush=True,
        )
        try:
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(None, input)
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer.strip().lower() == "y":
            self._active_model = next_fallback
            self._fallback_index += 1
            self._fallback_confirmed = True
            self._rate_limit_streak = 0
            logger.info("User confirmed fallback to %s", self._active_model.full_id)
            return True

        logger.info(
            "User declined fallback. Marking models as exhausted — "
            "will stop after saving progress."
        )
        self._exhausted = True
        return False

    # ------------------------------------------------------------------
    # Core LLM call — always uses a fresh session
    # ------------------------------------------------------------------

    async def _describe_image(
        self,
        artifact: NormalizedArtifact,
        model: ModelConfig,
    ) -> dict[str, Any]:
        """
        Send an image to the LLM for description.

        Always creates a fresh session to prevent context accumulation
        across many image calls (which caused the 300s timeout wall).
        """
        system_prompt = self.load_prompt()

        user_prompt = (
            "Analiza esta imagen de un documento de planificación municipal colombiana "
            "(POT — Plan de Ordenamiento Territorial, Uribia, La Guajira). "
            "Proporciona una descripción detallada del contenido de la imagen. "
            "Si es una gráfica o tabla, extrae todos los valores, etiquetas y unidades. "
            "Si es un mapa, describe los elementos geográficos, la leyenda, la escala y las anotaciones. "
            "Si es una tabla presentada como imagen, extrae todos los valores de cada celda. "
            "Responde en español."
        )

        fmt = artifact.metadata.get("format", "png")
        media_type_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "tiff": "image/tiff",
        }
        media_type = media_type_map.get(fmt, "image/png")

        # Fresh session per image — critical to avoid context bloat
        session_id = await self.client.create_fresh_session(
            title=f"preprocess-{artifact.artifact_id[:8]}"
        )

        body: dict[str, Any] = {
            "model": {
                "providerID": model.provider_id,
                "modelID": model.model_id,
            },
            "parts": [
                {
                    "type": "file",
                    "mime": media_type,
                    "url": f"data:{media_type};base64,{artifact.image_base64}",
                },
                {"type": "text", "text": user_prompt},
            ],
            "tools": {},
            "system": system_prompt,
            "format": {
                "type": "json_schema",
                "schema": IMAGE_DESCRIPTION_SCHEMA,
                "retryCount": 2,
            },
        }

        resp = await self.client._request(
            "POST",
            f"/session/{session_id}/message",
            json=body,
        )
        parsed = self.client._parse_response(resp)

        # Track token usage
        usage = parsed.get("usage", {})
        self._total_input_tokens += usage.get("input_tokens", 0)
        self._total_output_tokens += usage.get("output_tokens", 0)

        if "structured_output" in parsed:
            return parsed["structured_output"]

        if "structured_output_error" in parsed:
            raise RuntimeError(
                f"Structured output failed: {parsed['structured_output_error']}"
            )

        raise RuntimeError(
            f"No structured output in response: {parsed.get('text', '')[:200]}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_description(
        artifact: NormalizedArtifact, result: dict[str, Any]
    ) -> None:
        """Apply LLM description result to the artifact in-place."""
        description = result.get("description", "")
        content_type = result.get("content_type", "other")
        confidence = result.get("confidence", 0.5)

        type_map = {
            "chart": ArtifactType.CHART,
            "map": ArtifactType.MAP,
            "table_image": ArtifactType.TABLE,
        }
        if content_type in type_map:
            artifact.artifact_type = type_map[content_type]

        parts = [f"[{content_type.upper()}]", description]

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

    @staticmethod
    def _stamp_metadata(
        artifact: NormalizedArtifact,
        *,
        status: str,
        model: str,
        attempts: int,
        run_id: str,
        error: str = "",
    ) -> None:
        """Stamp artifact metadata with preprocessing tracking info."""
        artifact.metadata["preprocess_status"] = status
        artifact.metadata["preprocess_model"] = model
        artifact.metadata["preprocess_attempts"] = attempts
        artifact.metadata["preprocess_run_id"] = run_id
        artifact.metadata["preprocess_last_updated"] = datetime.utcnow().isoformat()
        if error:
            artifact.metadata["preprocess_error"] = error[:300]
        elif "preprocess_error" in artifact.metadata:
            del artifact.metadata["preprocess_error"]
