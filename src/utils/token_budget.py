"""
Token counting and cost tracking utilities.

Provides token counting using tiktoken, cost estimation per model,
budget enforcement, and usage reporting across pipeline stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from src.config import MODELS, ModelConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting (shared encoding)
# ---------------------------------------------------------------------------

_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    """Lazy-load the cl100k_base tokenizer."""
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Count tokens in a text string using cl100k_base."""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    enc = _get_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def estimate_token_count(text: str) -> int:
    """
    Fast approximate token count without full tokenization.

    Uses the ~4 chars per token heuristic for cl100k_base.
    Useful for rough budget checks before expensive tokenization.
    """
    return max(1, len(text) // 4)


def fits_context_window(
    text: str,
    model: ModelConfig,
    reserve_output: int = 0,
) -> bool:
    """Check if text fits within a model's context window."""
    available = model.context_window - (reserve_output or model.max_output)
    return count_tokens(text) <= available


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: ModelConfig,
) -> float:
    """Estimate cost in USD for a given token usage."""
    input_cost = (input_tokens / 1_000_000) * model.cost_input_per_m
    output_cost = (output_tokens / 1_000_000) * model.cost_output_per_m
    return input_cost + output_cost


def estimate_cost_for_role(
    role: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost for a specific agent role."""
    model = MODELS.get(role)
    if model is None:
        return 0.0
    return estimate_cost(input_tokens, output_tokens, model)


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------


@dataclass
class RoleUsage:
    """Token usage for a single role/model."""

    role: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class BudgetTracker:
    """
    Tracks token usage and costs across all models/roles.

    Per Project.md: "enforce token budgets and cost caps"
    and "Reserve Opus calls for style learning and final synthesis only."
    """

    # Per-role usage
    usage: dict[str, RoleUsage] = field(default_factory=dict)
    # Optional cost cap
    cost_cap_usd: float | None = None
    # Optional per-role call limits
    call_limits: dict[str, int] = field(default_factory=dict)

    def record_usage(
        self,
        role: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record token usage for a role."""
        if role not in self.usage:
            model = MODELS.get(role)
            self.usage[role] = RoleUsage(
                role=role,
                model_id=model.full_id if model else "unknown",
            )

        entry = self.usage[role]
        entry.input_tokens += input_tokens
        entry.output_tokens += output_tokens
        entry.call_count += 1

        # Calculate cost
        model = MODELS.get(role)
        if model:
            entry.cost_usd = estimate_cost(
                entry.input_tokens, entry.output_tokens, model
            )

    def check_budget(self, role: str | None = None) -> bool:
        """
        Check if we're within budget.

        Returns True if within limits, False if budget exceeded.
        """
        # Check total cost cap
        if self.cost_cap_usd is not None:
            if self.total_cost_usd > self.cost_cap_usd:
                logger.warning(
                    "Cost cap exceeded: $%.2f > $%.2f",
                    self.total_cost_usd,
                    self.cost_cap_usd,
                )
                return False

        # Check per-role call limits
        if role and role in self.call_limits:
            entry = self.usage.get(role)
            if entry and entry.call_count >= self.call_limits[role]:
                logger.warning(
                    "Call limit for %s exceeded: %d >= %d",
                    role,
                    entry.call_count,
                    self.call_limits[role],
                )
                return False

        return True

    @property
    def total_input_tokens(self) -> int:
        return sum(e.input_tokens for e in self.usage.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(e.output_tokens for e in self.usage.values())

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.usage.values())

    @property
    def total_calls(self) -> int:
        return sum(e.call_count for e in self.usage.values())

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all usage."""
        return {
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_calls": self.total_calls,
            "cost_cap_usd": self.cost_cap_usd,
            "within_budget": self.check_budget(),
            "by_role": {
                role: {
                    "model": entry.model_id,
                    "input_tokens": entry.input_tokens,
                    "output_tokens": entry.output_tokens,
                    "total_tokens": entry.total_tokens,
                    "calls": entry.call_count,
                    "cost_usd": entry.cost_usd,
                }
                for role, entry in self.usage.items()
            },
        }

    def format_report(self) -> str:
        """Format a human-readable usage report."""
        lines = [
            "# Token Budget Report",
            "",
            f"**Total tokens:** {self.total_tokens:,}",
            f"**Total cost:** ${self.total_cost_usd:.4f}",
            f"**Total calls:** {self.total_calls}",
        ]

        if self.cost_cap_usd is not None:
            remaining = self.cost_cap_usd - self.total_cost_usd
            lines.append(
                f"**Budget remaining:** ${remaining:.4f} / ${self.cost_cap_usd:.4f}"
            )

        lines.append("")
        lines.append("| Role | Model | Calls | Input | Output | Cost |")
        lines.append("|------|-------|-------|-------|--------|------|")

        for role, entry in sorted(self.usage.items()):
            lines.append(
                f"| {role} | {entry.model_id} | {entry.call_count} | "
                f"{entry.input_tokens:,} | {entry.output_tokens:,} | "
                f"${entry.cost_usd:.4f} |"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline-level budget estimation
# ---------------------------------------------------------------------------


@dataclass
class PipelineBudgetEstimate:
    """Estimated token usage and cost for the full pipeline."""

    estimated_chunks: int = 0
    preprocessing_calls: int = 0
    chunk_summary_calls: int = 0
    style_learning_calls: int = 1
    central_synthesis_calls: int = 5
    review_calls: int = 3
    slide_generation_calls: int = 10
    estimated_total_cost: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)


def estimate_pipeline_cost(
    total_text_tokens: int,
    n_images: int = 0,
    chunk_budget: int = 10_000,
) -> PipelineBudgetEstimate:
    """
    Estimate total pipeline cost based on input size.

    Uses Project.md guidance:
    - Preprocessing: Gemini for images ($2/$12 per M tokens)
    - Chunk summarization: Sonnet ($0, subscription)
    - Style learning: 1-2 Opus calls ($0)
    - Central synthesis: ~5-10 Opus calls ($0)
    - Review: GPT-5.3 ($0)
    - Slides: GPT-5.3 ($0)

    Only Gemini incurs actual per-token costs.
    """
    est = PipelineBudgetEstimate()

    # Estimate chunk count
    est.estimated_chunks = max(1, total_text_tokens // chunk_budget)

    # Preprocessing (Gemini) - mainly for images
    est.preprocessing_calls = n_images
    gemini = MODELS.get("preprocessing")
    if gemini and n_images > 0:
        # Estimate ~2000 tokens per image description
        img_input = n_images * 3000  # image + prompt tokens
        img_output = n_images * 2000
        gemini_cost = estimate_cost(img_input, img_output, gemini)
        est.breakdown["preprocessing"] = gemini_cost
    else:
        est.breakdown["preprocessing"] = 0.0

    # Chunk summarization (Sonnet) - $0
    est.chunk_summary_calls = est.estimated_chunks
    est.breakdown["chunk_summarization"] = 0.0

    # Style learning (Opus) - $0
    est.breakdown["style_learning"] = 0.0

    # Central synthesis (Opus) - $0
    est.central_synthesis_calls = min(est.estimated_chunks // 5, 20) + 1
    est.breakdown["central_summarization"] = 0.0

    # Review (GPT-5.3) - $0
    est.breakdown["reviewer"] = 0.0

    # Slide generation (GPT-5.3) - $0
    est.breakdown["slide_generation"] = 0.0

    est.estimated_total_cost = sum(est.breakdown.values())

    return est


def create_default_budget_tracker(
    cost_cap: float | None = None,
) -> BudgetTracker:
    """
    Create a BudgetTracker with sensible defaults per Project.md.

    Limits Opus calls and sets a cost cap if specified.
    """
    return BudgetTracker(
        cost_cap_usd=cost_cap,
        call_limits={
            # Per Project.md: "3-6 Opus calls (style learning + central synth passes)"
            "style_learning": 3,
            "central_summarization": 20,
            # "300-500 Sonnet calls for chunk summaries"
            "chunk_summarization": 500,
            # Multiple GPT-5.3 passes
            "reviewer": 10,
            "slide_generation": 30,
            # Gemini for images only
            "preprocessing": 200,
        },
    )
