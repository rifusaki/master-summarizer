"""
Base agent class.

Provides common functionality for all pipeline agents: prompt
loading, LLM dispatch via OpenCode, output validation, provenance
attachment, and token budget tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import MODELS, PROMPTS_DIR, ModelConfig, pipeline_config
from src.models import AuditEntry, PipelineStage, ProvenanceRecord
from src.opencode_client import OpenCodeClient

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent errors."""


class LowConfidenceError(AgentError):
    """Output confidence below threshold."""


class BaseAgent:
    """
    Base class for all pipeline agents.

    Subclasses implement `run()` and optionally override `validate_output()`.
    """

    # Subclasses must set these
    role: str = ""  # Maps to MODELS key and prompt file name
    stage: PipelineStage = PipelineStage.PREPROCESSING

    def __init__(self, client: OpenCodeClient) -> None:
        self.client = client
        self.model: ModelConfig = MODELS[self.role]
        self._prompt_cache: dict[str, str] = {}
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    # ------------------------------------------------------------------
    # Prompt management
    # ------------------------------------------------------------------

    def load_prompt(self, name: str | None = None) -> str:
        """
        Load a prompt template from the prompts/ directory.

        Args:
            name: Prompt filename without extension. Defaults to self.role.
        """
        key = name or self.role
        if key in self._prompt_cache:
            return self._prompt_cache[key]

        filepath = PROMPTS_DIR / f"{key}.md"
        if not filepath.exists():
            logger.warning("Prompt file not found: %s", filepath)
            return ""

        content = filepath.read_text(encoding="utf-8")
        self._prompt_cache[key] = content
        return content

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    async def call_llm(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        image_base64: str | None = None,
        image_media_type: str = "image/png",
        json_schema: dict[str, Any] | None = None,
        model_override: ModelConfig | None = None,
    ) -> dict[str, Any]:
        """
        Send a prompt to the LLM assigned to this agent's role.

        Returns the raw response dict with 'text', 'usage', and
        optionally 'structured_output'.
        """
        model = model_override or self.model

        # Use system prompt from file if not provided
        if system_prompt is None:
            system_prompt = self.load_prompt()

        response = await self.client.send_prompt_for_role(
            role=self.role,
            user_prompt=user_prompt,
            system_prompt=system_prompt if system_prompt else None,
            image_base64=image_base64,
            image_media_type=image_media_type,
            json_schema=json_schema,
            model_override=model_override,
        )

        # Track token usage
        usage = response.get("usage", {})
        self._total_input_tokens += usage.get("input_tokens", 0)
        self._total_output_tokens += usage.get("output_tokens", 0)

        return response

    async def call_llm_structured(
        self,
        user_prompt: str,
        schema: dict[str, Any],
        system_prompt: str | None = None,
        image_base64: str | None = None,
    ) -> dict[str, Any]:
        """
        Call LLM expecting structured JSON output.

        Returns the parsed structured output dict, or raises on failure.
        """
        response = await self.call_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            image_base64=image_base64,
            json_schema=schema,
        )

        # Check for structured output
        if "structured_output" in response:
            return response["structured_output"]

        if "structured_output_error" in response:
            raise AgentError(
                f"Structured output failed: {response['structured_output_error']}"
            )

        # Try to parse text as JSON as fallback
        text = response.get("text", "")
        try:
            # Try to find JSON in the response text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        raise AgentError(
            f"Expected structured output but got plain text: {text[:200]}..."
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_output(self, output: Any) -> bool:
        """
        Validate agent output. Override in subclasses for
        agent-specific validation logic.

        Returns True if output is acceptable.
        """
        return True

    def check_confidence(self, confidence: float) -> None:
        """Raise if confidence is below the configured threshold."""
        if confidence < pipeline_config.confidence_threshold:
            raise LowConfidenceError(
                f"Confidence {confidence:.2f} below threshold "
                f"{pipeline_config.confidence_threshold}"
            )

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def create_provenance(
        self,
        chunk_ids: list[str] | None = None,
        original_excerpt: str = "",
    ) -> ProvenanceRecord:
        """Create a provenance record for this agent's output."""
        return ProvenanceRecord(
            chunk_ids=chunk_ids or [],
            agent=self.role,
            model=self.model.full_id,
            timestamp=datetime.now(),
            original_excerpt=original_excerpt[:500],
        )

    def create_audit_entry(
        self,
        action: str,
        input_ids: list[str] | None = None,
        output_ids: list[str] | None = None,
        confidence: float | None = None,
        notes: str = "",
    ) -> AuditEntry:
        """Create an audit log entry for this agent's action."""
        return AuditEntry(
            stage=self.stage,
            agent=self.role,
            model=self.model.full_id,
            action=action,
            input_ids=input_ids or [],
            output_ids=output_ids or [],
            tokens_input=self._total_input_tokens,
            tokens_output=self._total_output_tokens,
            confidence=confidence,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        input_cost = (
            self._total_input_tokens / 1_000_000
        ) * self.model.cost_input_per_m
        output_cost = (
            self._total_output_tokens / 1_000_000
        ) * self.model.cost_output_per_m
        return input_cost + output_cost
