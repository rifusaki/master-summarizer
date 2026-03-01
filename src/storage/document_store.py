"""
File-based document store for pipeline artifacts.

Manages storage and retrieval of parsed documents, chunks,
summaries, and other artifacts with JSON metadata sidecars.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from src.config import (
    AUDIT_LOG_FILE,
    CHUNK_SUMMARIES_DIR,
    CHUNKS_DIR,
    DRAFTS_DIR,
    OUTPUT_DIR,
    PREPROCESSED_DIR,
    REVIEWS_DIR,
    SLIDES_DIR,
    STATE_FILE,
    STYLE_GUIDE_DIR,
    ensure_output_dirs,
)

# Sub-directories for incremental section persistence
DRAFT_SECTIONS_DIR = DRAFTS_DIR / "sections"
SLIDE_SECTIONS_DIR = SLIDES_DIR / "sections"
from src.models import (
    AuditEntry,
    ChunkSummary,
    Chunk,
    DocumentParseResult,
    DraftSection,
    MasterDraft,
    PipelineState,
    ReviewResult,
    SlideOutline,
    SlideOutlineSet,
    StyleGuide,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class DocumentStore:
    """
    File-based artifact store with JSON serialization.

    Each artifact type is stored in its designated directory
    as a JSON file named by its ID.
    """

    def __init__(self) -> None:
        ensure_output_dirs()

    # ------------------------------------------------------------------
    # Generic save / load
    # ------------------------------------------------------------------

    # Map each model class to its primary ID field name.
    # This avoids ambiguity when a model has multiple *_id fields
    # (e.g. Chunk has both chunk_id and document_id).
    _PRIMARY_ID_FIELD: dict[type, str] = {
        DocumentParseResult: "document_id",
        Chunk: "chunk_id",
        ChunkSummary: "summary_id",
        DraftSection: "section_id",
        StyleGuide: "guide_id",
        MasterDraft: "draft_id",
        ReviewResult: "review_id",
        SlideOutlineSet: "outline_id",
    }

    @staticmethod
    def _atomic_write(content: str, filepath: Path) -> None:
        """Write content atomically: write to tmp, fsync, rename.

        Prevents partial writes that could corrupt pipeline state on crash.
        """
        tmp_fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, filepath)
        except Exception:
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
            raise

    @staticmethod
    def _save(obj: BaseModel, directory: Path, filename: str | None = None) -> Path:
        """Save a Pydantic model to a JSON file atomically."""
        # Use object's primary ID field for filename if not specified
        if filename is None:
            id_field = DocumentStore._PRIMARY_ID_FIELD.get(type(obj))
            if id_field is not None:
                filename = f"{getattr(obj, id_field)}.json"
            else:
                raise ValueError(
                    f"Cannot determine filename for {type(obj).__name__}: "
                    f"add it to DocumentStore._PRIMARY_ID_FIELD"
                )

        filepath = directory / filename
        DocumentStore._atomic_write(obj.model_dump_json(indent=2), filepath)
        logger.debug("Saved %s to %s", type(obj).__name__, filepath)
        return filepath

    @staticmethod
    def _load(model_class: type[T], filepath: Path) -> T:
        """Load a Pydantic model from a JSON file."""
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return model_class.model_validate(data)

    @staticmethod
    def _load_all(model_class: type[T], directory: Path) -> list[T]:
        """Load all JSON files in a directory as Pydantic models."""
        results = []
        if not directory.exists():
            return results
        for filepath in sorted(directory.glob("*.json")):
            try:
                results.append(DocumentStore._load(model_class, filepath))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", filepath, exc)
        return results

    # ------------------------------------------------------------------
    # Parsed documents
    # ------------------------------------------------------------------

    def save_parse_result(self, result: DocumentParseResult) -> Path:
        return self._save(result, PREPROCESSED_DIR)

    def load_parse_result(self, document_id: str) -> DocumentParseResult:
        return self._load(DocumentParseResult, PREPROCESSED_DIR / f"{document_id}.json")

    def load_all_parse_results(self) -> list[DocumentParseResult]:
        return self._load_all(DocumentParseResult, PREPROCESSED_DIR)

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def save_chunk(self, chunk: Chunk) -> Path:
        return self._save(chunk, CHUNKS_DIR)

    def save_chunks(self, chunks: list[Chunk]) -> list[Path]:
        return [self.save_chunk(c) for c in chunks]

    def load_chunk(self, chunk_id: str) -> Chunk:
        return self._load(Chunk, CHUNKS_DIR / f"{chunk_id}.json")

    def load_all_chunks(self) -> list[Chunk]:
        return self._load_all(Chunk, CHUNKS_DIR)

    def load_chunks_for_document(self, document_id: str) -> list[Chunk]:
        all_chunks = self.load_all_chunks()
        return [c for c in all_chunks if c.document_id == document_id]

    # ------------------------------------------------------------------
    # Chunk summaries
    # ------------------------------------------------------------------

    def save_chunk_summary(self, summary: ChunkSummary) -> Path:
        return self._save(summary, CHUNK_SUMMARIES_DIR)

    def save_chunk_summaries(self, summaries: list[ChunkSummary]) -> list[Path]:
        return [self.save_chunk_summary(s) for s in summaries]

    def load_chunk_summary(self, summary_id: str) -> ChunkSummary:
        return self._load(ChunkSummary, CHUNK_SUMMARIES_DIR / f"{summary_id}.json")

    def load_all_chunk_summaries(self) -> list[ChunkSummary]:
        return self._load_all(ChunkSummary, CHUNK_SUMMARIES_DIR)

    # ------------------------------------------------------------------
    # Style guide
    # ------------------------------------------------------------------

    def save_style_guide(self, guide: StyleGuide) -> Path:
        return self._save(guide, STYLE_GUIDE_DIR)

    def load_style_guide(self) -> StyleGuide | None:
        """Load the latest style guide (there should be only one)."""
        guides = self._load_all(StyleGuide, STYLE_GUIDE_DIR)
        return guides[-1] if guides else None

    # ------------------------------------------------------------------
    # Master draft
    # ------------------------------------------------------------------

    def save_draft(self, draft: MasterDraft) -> Path:
        return self._save(draft, DRAFTS_DIR, f"draft_v{draft.version}.json")

    def load_draft(self, version: int) -> MasterDraft:
        return self._load(MasterDraft, DRAFTS_DIR / f"draft_v{version}.json")

    def load_latest_draft(self) -> MasterDraft | None:
        drafts = self._load_all(MasterDraft, DRAFTS_DIR)
        return max(drafts, key=lambda d: d.version) if drafts else None

    # ------------------------------------------------------------------
    # Draft sections (incremental saves for Stage 5 resume)
    # ------------------------------------------------------------------

    @staticmethod
    def _heading_slug(heading: str) -> str:
        """Convert a section heading to a safe filename slug."""
        import re

        slug = heading.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "_", slug)
        slug = slug[:80]  # cap length
        return slug or "section"

    def save_draft_section(self, section: DraftSection) -> Path:
        """Save a single DraftSection to disk immediately (for resume support)."""
        DRAFT_SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        slug = self._heading_slug(section.heading)
        filename = f"{slug}__{section.section_id[:8]}.json"
        filepath = DRAFT_SECTIONS_DIR / filename
        self._atomic_write(section.model_dump_json(indent=2), filepath)
        logger.debug("Saved draft section '%s' to %s", section.heading, filepath)
        return filepath

    def load_all_draft_sections(self) -> list[DraftSection]:
        """Load all incrementally-saved draft sections."""
        return self._load_all(DraftSection, DRAFT_SECTIONS_DIR)

    def clear_draft_sections(self) -> None:
        """Delete all incremental draft section files (called after full draft saved)."""
        if DRAFT_SECTIONS_DIR.exists():
            for f in DRAFT_SECTIONS_DIR.glob("*.json"):
                try:
                    f.unlink()
                except Exception as exc:
                    logger.warning("Could not delete draft section file %s: %s", f, exc)

    # ------------------------------------------------------------------
    # Slide sections (incremental saves for Stage 7 resume)
    # ------------------------------------------------------------------

    def save_slide_section(self, heading: str, slides: list[SlideOutline]) -> Path:
        """
        Save slides for one draft section immediately (for resume support).

        Stores a JSON object with {"heading": ..., "slides": [...]}.
        """
        SLIDE_SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        slug = self._heading_slug(heading)
        # Use a stable filename per heading so re-runs overwrite safely
        filename = f"{slug}.json"
        filepath = SLIDE_SECTIONS_DIR / filename
        payload = {
            "heading": heading,
            "slides": [s.model_dump() for s in slides],
        }
        self._atomic_write(json.dumps(payload, indent=2, default=str), filepath)
        logger.debug("Saved %d slides for section '%s'", len(slides), heading)
        return filepath

    def load_all_slide_sections(self) -> dict[str, list[SlideOutline]]:
        """
        Load all incrementally-saved slide sections.

        Returns a dict of {heading: [SlideOutline, ...]}.
        """
        result: dict[str, list[SlideOutline]] = {}
        if not SLIDE_SECTIONS_DIR.exists():
            return result
        for filepath in sorted(SLIDE_SECTIONS_DIR.glob("*.json")):
            try:
                payload = json.loads(filepath.read_text(encoding="utf-8"))
                heading = payload.get("heading", filepath.stem)
                slides = [
                    SlideOutline.model_validate(s) for s in payload.get("slides", [])
                ]
                result[heading] = slides
            except Exception as exc:
                logger.warning("Failed to load slide section %s: %s", filepath, exc)
        return result

    def clear_slide_sections(self) -> None:
        """Delete all incremental slide section files (called after full set saved)."""
        if SLIDE_SECTIONS_DIR.exists():
            for f in SLIDE_SECTIONS_DIR.glob("*.json"):
                try:
                    f.unlink()
                except Exception as exc:
                    logger.warning("Could not delete slide section file %s: %s", f, exc)

    # ------------------------------------------------------------------
    # Review results
    # ------------------------------------------------------------------

    def save_review(self, review: ReviewResult) -> Path:
        return self._save(review, REVIEWS_DIR)

    def load_all_reviews(self) -> list[ReviewResult]:
        return self._load_all(ReviewResult, REVIEWS_DIR)

    # ------------------------------------------------------------------
    # Slide outlines
    # ------------------------------------------------------------------

    def save_slide_outlines(self, outlines: SlideOutlineSet) -> Path:
        return self._save(outlines, SLIDES_DIR)

    def load_slide_outlines(self) -> SlideOutlineSet | None:
        results = self._load_all(SlideOutlineSet, SLIDES_DIR)
        return results[-1] if results else None

    # ------------------------------------------------------------------
    # Pipeline state (checkpointing)
    # ------------------------------------------------------------------

    def save_pipeline_state(self, state: PipelineState) -> Path:
        filepath = STATE_FILE
        self._atomic_write(state.model_dump_json(indent=2), filepath)
        logger.info("Pipeline state saved (stage: %s)", state.current_stage)
        return filepath

    def load_pipeline_state(self) -> PipelineState | None:
        if not STATE_FILE.exists():
            return None
        return self._load(PipelineState, STATE_FILE)

    # ------------------------------------------------------------------
    # Audit log (append-only)
    # ------------------------------------------------------------------

    def append_audit_entry(self, entry: AuditEntry) -> None:
        """Append an entry to the immutable audit log."""
        AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def load_audit_log(self) -> list[AuditEntry]:
        """Load all audit entries."""
        if not AUDIT_LOG_FILE.exists():
            return []
        entries = []
        for line in AUDIT_LOG_FILE.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                entries.append(AuditEntry.model_validate_json(line))
        return entries

    # ------------------------------------------------------------------
    # Human review files
    # ------------------------------------------------------------------

    def write_review_file(self, filename: str, content: str) -> Path:
        """Write a markdown file for human review."""
        from src.config import REVIEW_DIR

        filepath = REVIEW_DIR / filename
        filepath.write_text(content, encoding="utf-8")
        logger.info("Review file written: %s", filepath)
        return filepath

    def check_review_complete(self, filename: str) -> bool:
        """Check if a review file has been marked as reviewed."""
        from src.config import REVIEW_DIR

        filepath = REVIEW_DIR / filename
        if not filepath.exists():
            return False
        content = filepath.read_text(encoding="utf-8")
        return "REVIEWED: YES" in content.upper() or "APPROVED" in content.upper()
