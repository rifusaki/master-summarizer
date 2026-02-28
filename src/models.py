"""
Pydantic data models for inter-agent communication.

All agents communicate via these JSON-serializable models, ensuring
machine-parseable messages with provenance metadata attached.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ArtifactType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    MAP = "map"
    CAPTION = "caption"


class ReviewVerdict(str, Enum):
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"


class PipelineStage(str, Enum):
    PREPROCESSING = "preprocessing"
    CHUNKING = "chunking"
    CHUNK_SUMMARIZATION = "chunk_summarization"
    STYLE_LEARNING = "style_learning"
    CENTRAL_SUMMARIZATION = "central_summarization"
    REVIEW = "review"
    REFINEMENT = "refinement"
    SLIDE_GENERATION = "slide_generation"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


# ---------------------------------------------------------------------------
# Provenance & metadata
# ---------------------------------------------------------------------------


class SourceLocation(BaseModel):
    """Tracks exactly where a piece of content came from."""

    source_file: str
    page: int | None = None
    section: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    paragraph_index: int | None = None
    table_index: int | None = None
    image_index: int | None = None


class ProvenanceRecord(BaseModel):
    """Links a claim or artifact to its source."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_ids: list[str] = Field(default_factory=list)
    source_locations: list[SourceLocation] = Field(default_factory=list)
    agent: str = ""
    model: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    original_excerpt: str = ""


# ---------------------------------------------------------------------------
# Preprocessing outputs
# ---------------------------------------------------------------------------


class NormalizedArtifact(BaseModel):
    """A single preprocessed artifact from a source document."""

    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: ArtifactType
    content: str  # text content or description
    raw_content: str = ""  # original unprocessed content
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: SourceLocation
    confidence: float = 1.0
    # For tables: structured data
    table_data: list[dict[str, Any]] | None = None
    # For images: path to saved image file
    image_path: str | None = None
    # For images: base64 encoded data
    image_base64: str | None = None


class DocumentParseResult(BaseModel):
    """Complete parse result for a single document."""

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_file: str
    title: str = ""
    artifacts: list[NormalizedArtifact] = Field(default_factory=list)
    heading_structure: list[dict[str, Any]] = Field(default_factory=list)
    total_text_length: int = 0
    total_tables: int = 0
    total_images: int = 0
    parse_timestamp: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A semantic chunk of content ready for summarization."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    source_file: str
    content: str
    token_count: int = 0
    # Semantic boundaries
    heading_path: list[str] = Field(default_factory=list)
    section_title: str = ""
    # What types of content are in this chunk
    contains_tables: bool = False
    contains_images: bool = False
    # Ordering
    sequence_index: int = 0
    # Source artifacts that compose this chunk
    artifact_ids: list[str] = Field(default_factory=list)
    source_locations: list[SourceLocation] = Field(default_factory=list)
    # Embedding (stored separately in ChromaDB, but tracked here)
    has_embedding: bool = False


# ---------------------------------------------------------------------------
# Chunk summaries
# ---------------------------------------------------------------------------


class KeyFact(BaseModel):
    """A single key fact extracted from a chunk."""

    fact: str
    category: str = ""  # e.g. "demographic", "geographic", "economic"
    entities: list[str] = Field(default_factory=list)
    numeric_value: float | None = None
    unit: str = ""
    source_chunk_id: str = ""


class NumericEntry(BaseModel):
    """A numeric data point preserved from source."""

    label: str
    value: float
    unit: str = ""
    context: str = ""
    source_chunk_id: str = ""


class ChunkSummary(BaseModel):
    """Summary of a single chunk produced by the summarization agent."""

    summary_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_id: str
    document_id: str
    source_file: str
    # Core summary
    summary: str
    key_facts: list[KeyFact] = Field(default_factory=list)
    numeric_table: list[NumericEntry] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    # Quality
    confidence: float = 0.0
    token_count: int = 0
    # Provenance
    provenance: ProvenanceRecord | None = None
    # Outline position
    section_title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    sequence_index: int = 0


# ---------------------------------------------------------------------------
# Style guide
# ---------------------------------------------------------------------------


class StyleRule(BaseModel):
    """A single inferred style rule."""

    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: str  # e.g. "tone", "structure", "formatting", "vocabulary"
    rule: str
    examples_do: list[str] = Field(default_factory=list)
    examples_dont: list[str] = Field(default_factory=list)
    priority: str = "medium"  # high, medium, low


class StyleGuide(BaseModel):
    """Machine-readable style guide inferred from example documents."""

    guide_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # Structural rules
    section_order: list[str] = Field(default_factory=list)
    preferred_headings: list[str] = Field(default_factory=list)
    # Style rules organized by category
    rules: list[StyleRule] = Field(default_factory=list)
    # Specific formatting
    bullet_density: str = ""  # e.g. "moderate", "high"
    allowed_abbreviations: list[str] = Field(default_factory=list)
    numeric_formatting: str = ""
    citation_style: str = ""
    # Tone
    tone_description: str = ""
    target_reader: str = ""
    # Human-readable rubric
    reviewer_checklist: list[str] = Field(default_factory=list)
    # Generation metadata
    source_documents: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord | None = None


# ---------------------------------------------------------------------------
# Master draft
# ---------------------------------------------------------------------------


class DraftSection(BaseModel):
    """A section of the master draft with provenance."""

    section_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    heading: str
    level: int = 1  # heading level (1-4)
    content: str
    # Provenance: which chunks/summaries contributed
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_summary_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    # Subsections
    subsections: list[DraftSection] = Field(default_factory=list)


class MasterDraft(BaseModel):
    """The synthesized master draft document."""

    draft_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    sections: list[DraftSection] = Field(default_factory=list)
    # Global provenance mapping
    provenance_map: dict[str, list[str]] = Field(default_factory=dict)
    # Metadata
    total_word_count: int = 0
    style_guide_id: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Review annotations
# ---------------------------------------------------------------------------


class ReviewAnnotation(BaseModel):
    """A single review annotation on a draft paragraph."""

    annotation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    section_id: str
    paragraph_index: int = 0
    verdict: ReviewVerdict
    reason: str = ""
    suggested_replacement: str = ""
    # Specific checks
    fact_check_passed: bool = True
    numeric_check_passed: bool = True
    tone_check_passed: bool = True
    # Risk level
    risk_level: str = "low"  # low, medium, high


class ReviewResult(BaseModel):
    """Complete review result from the secondary reviewer."""

    review_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    draft_id: str
    draft_version: int
    annotations: list[ReviewAnnotation] = Field(default_factory=list)
    # Summary metrics
    total_accept: int = 0
    total_edit: int = 0
    total_reject: int = 0
    # Risk register
    risk_register: list[dict[str, str]] = Field(default_factory=list)
    # Overall assessment
    overall_confidence: float = 0.0
    reviewer_notes: str = ""
    provenance: ProvenanceRecord | None = None


# ---------------------------------------------------------------------------
# Slide outlines
# ---------------------------------------------------------------------------


class SlideOutline(BaseModel):
    """Outline for a single slide."""

    slide_number: int
    title: str
    bullets: list[str] = Field(default_factory=list)
    suggested_visual: str = ""  # "table", "graph", "map", "photo", "none"
    visual_description: str = ""
    source_references: list[str] = Field(default_factory=list)
    speaker_notes: str = ""
    word_count: int = 0


class SlideOutlineSet(BaseModel):
    """Complete set of slide outlines."""

    outline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    slides: list[SlideOutline] = Field(default_factory=list)
    total_slides: int = 0
    draft_id: str = ""
    style_guide_id: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Pipeline state & task tracking
# ---------------------------------------------------------------------------


class PipelineTask(BaseModel):
    """A single task in the orchestrator's task queue."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stage: PipelineStage
    status: TaskStatus = TaskStatus.PENDING
    description: str = ""
    input_ids: list[str] = Field(default_factory=list)
    output_ids: list[str] = Field(default_factory=list)
    # Token usage
    tokens_input: int = 0
    tokens_output: int = 0
    estimated_cost_usd: float = 0.0
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    # Error handling
    error: str | None = None
    retry_count: int = 0


class PipelineState(BaseModel):
    """Overall state of the pipeline for checkpointing and resume."""

    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    current_stage: PipelineStage = PipelineStage.PREPROCESSING
    stages_completed: list[PipelineStage] = Field(default_factory=list)
    tasks: list[PipelineTask] = Field(default_factory=list)
    # Aggregate stats
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    total_documents: int = 0
    total_chunks: int = 0
    total_summaries: int = 0
    # Checkpoints
    last_checkpoint: datetime | None = None
    started_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    """Immutable audit trail entry."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    stage: PipelineStage
    agent: str
    model: str
    action: str  # e.g. "summarize_chunk", "generate_draft", "review"
    input_ids: list[str] = Field(default_factory=list)
    output_ids: list[str] = Field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    confidence: float | None = None
    notes: str = ""
