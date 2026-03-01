"""
Central configuration for the summarization pipeline.

Loads settings from environment variables and provides
model mappings, paths, and pipeline parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "input"
RAW_DATA_DIR = INPUT_DIR / "raw_data"
STYLE_EXAMPLES_DIR = INPUT_DIR / "style_examples"

OUTPUT_DIR = PROJECT_ROOT / "output"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
CHUNKS_DIR = OUTPUT_DIR / "chunks"
CHUNK_SUMMARIES_DIR = OUTPUT_DIR / "chunk_summaries"
STYLE_GUIDE_DIR = OUTPUT_DIR / "style_guide"
DRAFTS_DIR = OUTPUT_DIR / "drafts"
REVIEWS_DIR = OUTPUT_DIR / "reviews"
SLIDES_DIR = OUTPUT_DIR / "slides"

REVIEW_DIR = PROJECT_ROOT / "review"  # human review files
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Pipeline state persistence
STATE_FILE = OUTPUT_DIR / "pipeline_state.json"
AUDIT_LOG_FILE = OUTPUT_DIR / "audit_log.jsonl"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific LLM model."""

    provider_id: str
    model_id: str
    context_window: int  # tokens
    max_output: int  # tokens
    supports_images: bool = False
    supports_pdf: bool = False
    cost_input_per_m: float = 0.0  # USD per 1M input tokens
    cost_output_per_m: float = 0.0  # USD per 1M output tokens

    @property
    def full_id(self) -> str:
        return f"{self.provider_id}/{self.model_id}"


# Available models mapped by role
MODELS: dict[str, ModelConfig] = {
    "preprocessing": ModelConfig(
        provider_id="google",
        model_id="gemini-3.1-pro-preview",
        context_window=1_048_576,
        max_output=65_536,
        supports_images=True,
        supports_pdf=True,
        cost_input_per_m=2.0,
        cost_output_per_m=12.0,
    ),
    "chunk_summarization": ModelConfig(
        provider_id="azure-anthropic",
        model_id="claude-sonnet-4-6",
        context_window=200_000,
        max_output=64_000,
        supports_images=True,
    ),
    "style_learning": ModelConfig(
        provider_id="github-copilot",
        model_id="claude-opus-4.6",
        context_window=128_000,
        max_output=64_000,
        supports_images=True,
    ),
    "central_summarization": ModelConfig(
        provider_id="github-copilot",
        model_id="claude-opus-4.6",
        context_window=128_000,
        max_output=64_000,
        supports_images=True,
    ),
    "reviewer": ModelConfig(
        provider_id="azure-gpt",
        model_id="gpt-5.3-codex",
        context_window=200_000,  # not specified, estimate
        max_output=100_000,
    ),
    "slide_generation": ModelConfig(
        provider_id="azure-gpt",
        model_id="gpt-5.3-codex",
        context_window=200_000,
        max_output=100_000,
    ),
}


# ---------------------------------------------------------------------------
# Fallback model chains (tried in order when primary model fails)
# ---------------------------------------------------------------------------

# NOTE: gemini-3-pro model id must be validated against your OpenCode instance.
# Run: opencode models | grep gemini  to find the exact identifier.
FALLBACK_MODELS: dict[str, list[ModelConfig]] = {
    "preprocessing": [
        ModelConfig(
            provider_id="google",
            model_id="gemini-3-pro-preview",
            context_window=1_048_576,
            max_output=65_536,
            supports_images=True,
            supports_pdf=True,
            cost_input_per_m=1.0,
            cost_output_per_m=6.0,
        ),
    ],
    "chunk_summarization": [
        ModelConfig(
            provider_id="azure-anthropic",
            model_id="claude-sonnet-4-5",
            context_window=200_000,
            max_output=64_000,
            supports_images=True,
        ),
    ],
}


# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Runtime configuration for the pipeline."""

    # OpenCode server
    opencode_host: str = field(
        default_factory=lambda: os.getenv("OPENCODE_SERVER_HOST", "127.0.0.1")
    )
    opencode_port: int = field(
        default_factory=lambda: int(os.getenv("OPENCODE_SERVER_PORT", "4096"))
    )

    # Quality gates
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
    )
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))

    # Chunking
    chunk_token_budget: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_TOKEN_BUDGET", "10000"))
    )
    chunk_overlap_tokens: int = 500  # overlap between adjacent chunks

    # Batching
    chunk_summary_batch_size: int = 5  # parallel chunk summaries
    max_review_iterations: int = 3  # max refinement loops

    # Slide constraints
    target_slide_count_min: int = 80
    target_slide_count_max: int = 100
    max_words_per_slide: int = 50
    max_bullets_per_slide: int = 5

    # Timeouts (seconds)
    llm_timeout: int = 300  # 5 minutes per LLM call
    server_startup_timeout: int = 30

    # Preprocessing resilience
    # How many consecutive 429s before prompting user to switch to fallback model
    preprocessing_rate_limit_streak_threshold: int = field(
        default_factory=lambda: int(
            os.getenv("PREPROCESSING_RATE_LIMIT_STREAK_THRESHOLD", "3")
        )
    )
    # Retries per image on timeout (each retry uses a fresh session)
    preprocessing_timeout_retries: int = field(
        default_factory=lambda: int(os.getenv("PREPROCESSING_TIMEOUT_RETRIES", "1"))
    )
    # Ask user before switching to fallback Gemini model (set to 0 to auto-switch)
    preprocessing_confirm_fallback: bool = field(
        default_factory=lambda: os.getenv("PREPROCESSING_CONFIRM_FALLBACK", "1") != "0"
    )

    # Chunk summarization resilience
    # Retries using fallback model before accepting failure
    chunk_summary_fallback_retries: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SUMMARY_FALLBACK_RETRIES", "1"))
    )

    @property
    def opencode_base_url(self) -> str:
        return f"http://{self.opencode_host}:{self.opencode_port}"


# Singleton config instance
pipeline_config = PipelineConfig()


# ---------------------------------------------------------------------------
# Input file discovery
# ---------------------------------------------------------------------------


def get_raw_documents() -> list[Path]:
    """Return all DOCX files in the raw data directory."""
    if not RAW_DATA_DIR.exists():
        return []
    return sorted(RAW_DATA_DIR.glob("*.docx"))


def get_style_examples() -> list[Path]:
    """Return all PDF files in the style examples directory."""
    if not STYLE_EXAMPLES_DIR.exists():
        return []
    return sorted(STYLE_EXAMPLES_DIR.glob("*.pdf"))


def ensure_output_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in [
        PREPROCESSED_DIR,
        CHUNKS_DIR,
        CHUNK_SUMMARIES_DIR,
        STYLE_GUIDE_DIR,
        DRAFTS_DIR,
        DRAFTS_DIR / "sections",  # incremental draft-section saves
        REVIEWS_DIR,
        SLIDES_DIR,
        SLIDES_DIR / "sections",  # incremental slide-section saves
        REVIEW_DIR,
        CHROMA_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
