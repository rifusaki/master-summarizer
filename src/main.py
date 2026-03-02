"""
Main orchestrator pipeline.

Coordinates all agents through the full summarization pipeline:
parse → preprocess → chunk → summarize → style learn → synthesize → review → slides.

Supports stage-by-stage execution with manual checkpoints,
resumable state, quality gates, retry logic, human review
escalation, and graceful model exhaustion handling.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from src.config import (
    PROMPTS_DIR,
    REVIEW_DIR,
    ensure_output_dirs,
    get_raw_documents,
    get_style_examples,
    pipeline_config,
)
from src.models import (
    ArtifactType,
    ChunkSummary,
    DocumentParseResult,
    ImagePreprocessStatus,
    ItemStatus,
    MasterDraft,
    PipelineStage,
    PipelineState,
    PipelineTask,
    ReviewResult,
    SlideOutlineSet,
    StyleGuide,
    TaskStatus,
)
from src.opencode_client import OpenCodeClient
from src.storage.document_store import DocumentStore
from src.storage.vector_store import VectorStore

# Agents
from src.agents.base import ModelExhaustionError
from src.agents.preprocessor import PreprocessorAgent
from src.agents.chunker import Chunker
from src.agents.chunk_summarizer import ChunkSummarizerAgent
from src.agents.style_learner import StyleLearnerAgent
from src.agents.central_summarizer import CentralSummarizerAgent
from src.agents.reviewer import ReviewerAgent
from src.agents.slide_generator import SlideGeneratorAgent

# Parsers
from src.parsers.docx_parser import parse_docx
from src.parsers.pdf_parser import parse_pdf

# Utilities
from src.utils.provenance import (
    generate_provenance_report,
    validate_provenance,
)
from src.utils.quality import (
    QualityGateResult,
    check_review_quality,
    check_summary_confidence,
    generate_quality_report,
    items_needing_human_review,
    reconcile_numerics,
    run_all_quality_gates,
)
from src.utils.token_budget import (
    BudgetTracker,
    create_default_budget_tracker,
    estimate_pipeline_cost,
)

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class Pipeline:
    """
    Main pipeline orchestrator.

    Manages the full document summarization workflow with:
    - Stage-by-stage execution with manual checkpoints
    - Resumable state persistence
    - Quality gates with retry logic
    - Human review escalation
    - Token budget tracking
    - Graceful model exhaustion handling (save and stop)
    - Rich console progress display
    """

    def __init__(self) -> None:
        self.store = DocumentStore()
        self.vector_store = VectorStore()
        self.client = OpenCodeClient()
        self.budget = create_default_budget_tracker()

        # Pipeline state
        self.state: PipelineState = PipelineState()

        # Agents (initialized after client starts)
        self._preprocessor: PreprocessorAgent | None = None
        self._chunker: Chunker | None = None
        self._chunk_summarizer: ChunkSummarizerAgent | None = None
        self._style_learner: StyleLearnerAgent | None = None
        self._central_summarizer: CentralSummarizerAgent | None = None
        self._reviewer: ReviewerAgent | None = None
        self._slide_generator: SlideGeneratorAgent | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize all components."""
        console.print(Panel("Starting Pipeline", style="bold cyan"))

        ensure_output_dirs()
        self.vector_store.initialize()

        # Try to resume from checkpoint
        saved_state = self.store.load_pipeline_state()
        if saved_state:
            self.state = saved_state
            console.print(
                f"[yellow]Resumed from checkpoint: "
                f"stage={self.state.current_stage.value}, "
                f"completed={[s.value for s in self.state.stages_completed]}[/]"
            )
        else:
            console.print("[green]Starting fresh pipeline run[/]")

        # Start OpenCode server
        console.print("Connecting to OpenCode server...")
        await self.client.start()
        console.print("[green]OpenCode server ready[/]")

        # Initialize agents
        self._preprocessor = PreprocessorAgent(self.client)
        self._chunker = Chunker(vector_store=self.vector_store)
        self._chunk_summarizer = ChunkSummarizerAgent(self.client)
        self._style_learner = StyleLearnerAgent(self.client)
        self._central_summarizer = CentralSummarizerAgent(self.client)
        self._reviewer = ReviewerAgent(self.client)
        self._slide_generator = SlideGeneratorAgent(self.client)

    async def shutdown(self) -> None:
        """Clean up resources."""
        await self.client.stop()
        self._save_checkpoint()
        console.print(Panel("Pipeline Stopped", style="bold red"))

    def _save_checkpoint(self) -> None:
        """Save current pipeline state."""
        self.state.last_checkpoint = datetime.now()
        self.state.total_tokens_used = self.budget.total_tokens
        self.state.total_cost_usd = self.budget.total_cost_usd
        self.store.save_pipeline_state(self.state)

    def _mark_stage_complete(self, stage: PipelineStage) -> None:
        """Mark a stage as completed and save checkpoint."""
        if stage not in self.state.stages_completed:
            self.state.stages_completed.append(stage)
        self._save_checkpoint()

    def _is_stage_complete(self, stage: PipelineStage) -> bool:
        """Check if a stage has already been completed."""
        return stage in self.state.stages_completed

    def _handle_exhaustion(self, exc: ModelExhaustionError, stage_name: str) -> None:
        """
        Handle a ModelExhaustionError: save state, display status, and stop.

        This is called when all models in a fallback chain are exhausted.
        The pipeline saves all progress and stops cleanly so the user can
        adjust models/quotas and resume later.
        """
        self._save_checkpoint()
        console.print(
            Panel(
                f"[bold red]Model Exhaustion — Pipeline Stopped[/]\n\n"
                f"Stage: [bold]{stage_name}[/]\n"
                f"Role: {exc.role}\n"
                f"Models tried: {', '.join(exc.models_tried)}\n"
                f"Items completed: [green]{exc.items_completed}[/]\n"
                f"Items remaining: [red]{exc.items_remaining}[/]\n"
                f"Last error: {exc.last_error[:200]}\n\n"
                f"All progress has been saved. To resume:\n"
                f"  1. Check model quotas / API keys\n"
                f"  2. Re-run the pipeline (it will resume from checkpoint)\n"
                f"  3. Or use [bold]--retry-failed[/] to retry only failed items",
                title="Model Exhaustion",
                style="bold red",
            )
        )

    # ------------------------------------------------------------------
    # Manual checkpoint (user confirmation)
    # ------------------------------------------------------------------

    async def _checkpoint(self, stage_name: str, next_stage: str) -> bool:
        """
        Pause for manual checkpoint between stages.

        Saves state and prompts user to continue.
        Returns True to continue, False to stop.
        """
        self._save_checkpoint()

        # Display budget summary
        self._display_budget()

        console.print(
            Panel(
                f"[bold green]Stage '{stage_name}' complete.[/]\n"
                f"Next: [bold]{next_stage}[/]\n\n"
                f"Type [bold]'c'[/] to continue, [bold]'q'[/] to quit and save state.",
                title="Checkpoint",
                style="cyan",
            )
        )

        # Read input in a non-blocking way
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: input("Continue? [c/q]: ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            return False

        if response == "q":
            console.print("[yellow]Pipeline paused. State saved. Resume later.[/]")
            return False

        return True

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def stage_parse_and_preprocess(self) -> list[DocumentParseResult]:
        """
        Stage 1: Parse DOCX and PDF files, then preprocess images via Gemini.

        Resilient design:
        - Each document is saved to disk immediately after parsing (before
          image preprocessing), so a crash never loses parsed text.
        - Each image description is persisted to disk immediately via the
          on_image_done callback, so no Gemini tokens are wasted on crash.
        - On restart, already-saved documents are reloaded from disk and only
          images that are not yet successfully described are re-processed.
        """
        stage = PipelineStage.PREPROCESSING

        if self._is_stage_complete(stage):
            console.print("[dim]Preprocessing already complete, loading results...[/]")
            return self.store.load_all_parse_results()

        self.state.current_stage = stage
        console.print(Panel("Stage 1: Parse & Preprocess", style="bold magenta"))

        raw_docs = get_raw_documents()
        style_docs = get_style_examples()
        console.print(f"Found {len(raw_docs)} DOCX files, {len(style_docs)} PDF files")

        # Load any parse results already on disk (from previous partial runs)
        existing = {r.source_file: r for r in self.store.load_all_parse_results()}
        results: list[DocumentParseResult] = []

        # ----------------------------------------------------------------
        # Parse DOCX files (skip already saved ones)
        # ----------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing DOCX files...", total=len(raw_docs))
            for doc_path in raw_docs:
                if doc_path.name in existing:
                    logger.info("Skipping already-parsed DOCX: %s", doc_path.name)
                    results.append(existing[doc_path.name])
                    progress.update(
                        task, advance=1, description=f"Loaded: {doc_path.name[:40]}"
                    )
                    continue
                try:
                    logger.info("Parsing DOCX: %s", doc_path.name)
                    result = parse_docx(doc_path)
                    # Save immediately so a crash later doesn't lose parse work
                    self.store.save_parse_result(result)
                    results.append(result)
                    progress.update(
                        task, advance=1, description=f"Parsed: {doc_path.name[:40]}"
                    )
                except Exception as exc:
                    logger.error("Failed to parse %s: %s", doc_path.name, exc)
                    console.print(f"[red]Failed to parse {doc_path.name}: {exc}[/]")
                    progress.update(task, advance=1)

        # ----------------------------------------------------------------
        # Parse PDF style examples (skip already saved ones)
        # ----------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing PDF examples...", total=len(style_docs))
            for pdf_path in style_docs:
                if pdf_path.name in existing:
                    logger.info("Skipping already-parsed PDF: %s", pdf_path.name)
                    results.append(existing[pdf_path.name])
                    progress.update(
                        task, advance=1, description=f"Loaded: {pdf_path.name[:40]}"
                    )
                    continue
                try:
                    logger.info("Parsing PDF: %s", pdf_path.name)
                    result = parse_pdf(pdf_path)
                    self.store.save_parse_result(result)
                    results.append(result)
                    progress.update(
                        task, advance=1, description=f"Parsed: {pdf_path.name[:40]}"
                    )
                except Exception as exc:
                    logger.error("Failed to parse %s: %s", pdf_path.name, exc)
                    console.print(f"[red]Failed to parse {pdf_path.name}: {exc}[/]")
                    progress.update(task, advance=1)

        # ----------------------------------------------------------------
        # Preprocess images — only those not yet successfully described
        # ----------------------------------------------------------------
        assert self._preprocessor is not None

        pending_images = sum(
            1
            for r in results
            for a in r.artifacts
            if a.artifact_type == ArtifactType.IMAGE
            and a.image_base64
            and a.metadata.get("preprocess_status") != ImagePreprocessStatus.SUCCESS
        )

        if pending_images > 0:
            console.print(
                f"Preprocessing {pending_images} images via Gemini "
                f"(fresh session per image to prevent context bloat)..."
            )
            run_id = self.state.active_run_id

            for result in results:
                needs_preprocess = any(
                    a.artifact_type == ArtifactType.IMAGE
                    and a.image_base64
                    and a.metadata.get("preprocess_status")
                    != ImagePreprocessStatus.SUCCESS
                    for a in result.artifacts
                )
                if not needs_preprocess:
                    continue

                async def _save_callback(r: DocumentParseResult) -> None:
                    self.store.save_parse_result(r)
                    self.budget.set_cumulative_usage(
                        "preprocessing",
                        self._preprocessor._total_input_tokens,  # type: ignore[union-attr]
                        self._preprocessor._total_output_tokens,  # type: ignore[union-attr]
                    )
                    # Also persist pipeline state so we can see progress
                    self._save_checkpoint()

                try:
                    await self._preprocessor.process_document(
                        result,
                        run_id=run_id,
                        on_image_done=_save_callback,
                    )
                except ModelExhaustionError as exc:
                    self._handle_exhaustion(exc, "Image Preprocessing")
                    # Save what we have so far and return partial results
                    self._display_parse_summary(results)
                    self.state.total_documents = len(results)
                    self._save_checkpoint()
                    raise
        else:
            console.print("[dim]All images already preprocessed.[/]")

        self._display_parse_summary(results)
        self.state.total_documents = len(results)
        self._mark_stage_complete(stage)

        return results

    async def stage_chunk(self, parse_results: list[DocumentParseResult]) -> list[Any]:
        """
        Stage 2: Chunk parsed documents.

        Deterministic semantic chunking with heading boundaries and
        token budgets. Stores embeddings in ChromaDB.
        """
        stage = PipelineStage.CHUNKING

        if self._is_stage_complete(stage):
            console.print("[dim]Chunking already complete, loading results...[/]")
            return self.store.load_all_chunks()

        self.state.current_stage = stage
        console.print(Panel("Stage 2: Chunking", style="bold magenta"))

        assert self._chunker is not None
        all_chunks = []

        # Only chunk raw data documents (not style examples)
        raw_docs = get_raw_documents()
        raw_filenames = {p.name for p in raw_docs}
        data_results = [r for r in parse_results if r.source_file in raw_filenames]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Chunking documents...", total=len(data_results))
            for result in data_results:
                chunks = self._chunker.chunk_document(result)
                all_chunks.extend(chunks)
                self.store.save_chunks(chunks)
                progress.update(
                    task,
                    advance=1,
                    description=f"Chunked: {result.source_file[:35]} -> {len(chunks)} chunks",
                )

        self.state.total_chunks = len(all_chunks)
        console.print(f"[green]Created {len(all_chunks)} chunks total[/]")

        # Display chunk stats
        self._display_chunk_summary(all_chunks)

        self._mark_stage_complete(stage)
        return all_chunks

    async def stage_summarize_chunks(self, chunks: list[Any]) -> list[ChunkSummary]:
        """
        Stage 3: Summarize each chunk via Sonnet (with fallback).

        Processes chunks with per-chunk incremental saves and
        automatic retry/fallback. Raises ModelExhaustionError on
        exhaustion (after saving all progress).
        """
        stage = PipelineStage.CHUNK_SUMMARIZATION

        if self._is_stage_complete(stage):
            console.print("[dim]Chunk summarization already complete, loading...[/]")
            return self.store.load_all_chunk_summaries()

        self.state.current_stage = stage
        console.print(Panel("Stage 3: Chunk Summarization", style="bold magenta"))

        assert self._chunk_summarizer is not None

        # Load already-completed summaries so we can resume mid-stage
        existing_summaries = self.store.load_all_chunk_summaries()
        completed_chunk_ids = {s.chunk_id for s in existing_summaries}
        if completed_chunk_ids:
            console.print(
                f"[dim]Resuming: {len(completed_chunk_ids)} chunks already summarized, "
                f"skipping those.[/]"
            )

        pending_chunks = [c for c in chunks if c.chunk_id not in completed_chunk_ids]
        all_summaries: list[ChunkSummary] = list(existing_summaries)

        if not pending_chunks:
            console.print("[dim]All chunks already summarized.[/]")
        else:
            # Per-chunk save callback
            async def _on_chunk_done(summary: ChunkSummary, status: str) -> None:
                self.store.save_chunk_summary(summary)
                self.budget.set_cumulative_usage(
                    "chunk_summarization",
                    self._chunk_summarizer._total_input_tokens,  # type: ignore[union-attr]
                    self._chunk_summarizer._total_output_tokens,  # type: ignore[union-attr]
                )
                self._save_checkpoint()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Summarizing chunks...", total=len(pending_chunks)
                )

                # Wrap callback to also update progress bar
                chunk_counter = 0

                async def _on_chunk_done_with_progress(
                    summary: ChunkSummary, status: str
                ) -> None:
                    nonlocal chunk_counter
                    chunk_counter += 1
                    await _on_chunk_done(summary, status)
                    progress.update(
                        task,
                        advance=1,
                        description=f"Chunk {chunk_counter}/{len(pending_chunks)}: "
                        f"{summary.section_title[:30]}",
                    )

                new_summaries = await self._chunk_summarizer.summarize_chunks(
                    pending_chunks,
                    on_chunk_done=_on_chunk_done_with_progress,
                )
                all_summaries.extend(new_summaries)

        # Quality gate: confidence check
        confidence_gate = check_summary_confidence(all_summaries)
        self._display_gate_result(confidence_gate)

        if not confidence_gate.passed:
            # Write low-confidence items for human review
            review_items = confidence_gate.failed_items
            if review_items:
                self._write_review_file(
                    "low_confidence_summaries.md",
                    "Low Confidence Summaries",
                    review_items,
                )

        self.state.total_summaries = len(all_summaries)
        self._mark_stage_complete(stage)

        return all_summaries

    async def stage_learn_style(
        self, parse_results: list[DocumentParseResult]
    ) -> StyleGuide:
        """
        Stage 4: Learn style from example PDFs via Opus (with fallback).

        Analyzes style example documents to produce a machine-readable
        style guide for downstream synthesis.

        Raises:
            ModelExhaustionError: When all models are exhausted.
        """
        stage = PipelineStage.STYLE_LEARNING

        if self._is_stage_complete(stage):
            console.print("[dim]Style learning already complete, loading...[/]")
            guide = self.store.load_style_guide()
            if guide:
                return guide

        self.state.current_stage = stage
        console.print(Panel("Stage 4: Style Learning", style="bold magenta"))

        assert self._style_learner is not None

        # Get style example documents (PDF parse results)
        style_examples = get_style_examples()
        style_filenames = {p.name for p in style_examples}
        example_docs = [r for r in parse_results if r.source_file in style_filenames]

        if not example_docs:
            console.print(
                "[yellow]No style example parse results found. "
                "Using all parse results as fallback.[/]"
            )
            example_docs = parse_results[:2]

        console.print(f"Learning style from {len(example_docs)} example documents...")

        # Load manual style guide if available
        manual_guide_path = PROMPTS_DIR / "manual_style_guide.md"
        communication_guidelines = ""
        if manual_guide_path.exists():
            communication_guidelines = manual_guide_path.read_text(encoding="utf-8")
            console.print(
                f"[green]Loaded manual style guide ({len(communication_guidelines)} chars)[/]"
            )

        guide = await self._style_learner.learn_style(
            example_docs, communication_guidelines=communication_guidelines
        )

        # Store the manual guidelines in the guide for downstream use
        if communication_guidelines:
            guide.communication_guidelines = communication_guidelines

        # Track budget
        self.budget.set_cumulative_usage(
            "style_learning",
            self._style_learner._total_input_tokens,
            self._style_learner._total_output_tokens,
        )

        self.store.save_style_guide(guide)

        # Display style guide summary
        self._display_style_guide(guide)

        self._mark_stage_complete(stage)
        return guide

    async def stage_synthesize(
        self,
        chunk_summaries: list[ChunkSummary],
        style_guide: StyleGuide,
    ) -> MasterDraft:
        """
        Stage 5: Central synthesis via Opus (with fallback).

        Synthesizes all chunk summaries into a coherent master draft
        following the style guide. Saves incrementally per section.

        Raises:
            ModelExhaustionError: When all models are exhausted.
        """
        stage = PipelineStage.CENTRAL_SUMMARIZATION

        if self._is_stage_complete(stage):
            console.print("[dim]Synthesis already complete, loading...[/]")
            # Always load v1 — the synthesis output. Refined drafts (v2, v3, ...)
            # are produced by stage_review and must not be returned here, or the
            # review loop would restart from a mid-refinement state on resume.
            draft = self.store.load_synthesis_draft()
            if draft:
                if draft.version > 1:
                    logger.warning(
                        "draft_v1.json not found on resume; "
                        "using lowest available version (v%d)",
                        draft.version,
                    )
                return draft

        self.state.current_stage = stage
        console.print(Panel("Stage 5: Central Synthesis", style="bold magenta"))

        assert self._central_summarizer is not None

        # Load any sections already persisted from a prior partial run
        existing_sections = self.store.load_all_draft_sections()
        if existing_sections:
            console.print(
                f"[dim]Resuming synthesis: {len(existing_sections)} sections already "
                f"completed, skipping those.[/]"
            )

        console.print(
            f"Synthesizing {len(chunk_summaries)} summaries into master draft..."
        )

        # Per-section callback: save section to disk immediately
        section_count = len(existing_sections)

        async def _on_section_done(section: Any, status: str) -> None:
            nonlocal section_count
            section_count += 1
            self.store.save_draft_section(section)
            self.budget.set_cumulative_usage(
                "central_summarization",
                self._central_summarizer._total_input_tokens,  # type: ignore[union-attr]
                self._central_summarizer._total_output_tokens,  # type: ignore[union-attr]
            )
            self._save_checkpoint()
            console.print(
                f"  [dim]Section {section_count} complete: {section.heading[:40]}[/]"
            )

        draft = await self._central_summarizer.synthesize(
            chunk_summaries=chunk_summaries,
            style_guide=style_guide,
            completed_sections=existing_sections,
            on_section_done=_on_section_done,
        )

        # Final budget snapshot
        self.budget.set_cumulative_usage(
            "central_summarization",
            self._central_summarizer._total_input_tokens,
            self._central_summarizer._total_output_tokens,
        )

        self.store.save_draft(draft)
        # Full draft saved — discard incremental section files
        self.store.clear_draft_sections()

        # Provenance validation
        prov_result = validate_provenance(draft, chunk_summaries)
        if not prov_result.is_valid:
            console.print(
                f"[yellow]Provenance validation: {prov_result.coverage_ratio:.0%} coverage "
                f"({len(prov_result.sections_without_sources)} sections without sources)[/]"
            )
        else:
            console.print(
                f"[green]Provenance validation passed: "
                f"{prov_result.total_source_links} source links[/]"
            )

        # Display draft summary
        console.print(
            f"Draft v{draft.version}: {len(draft.sections)} sections, "
            f"{draft.total_word_count} words"
        )

        self._mark_stage_complete(stage)
        return draft

    async def stage_review(
        self,
        draft: MasterDraft,
        style_guide: StyleGuide,
        chunk_summaries: list[ChunkSummary],
    ) -> tuple[MasterDraft, ReviewResult]:
        """
        Stage 6: Review and refinement loop.

        The reviewer checks the draft, then the synthesizer refines it.
        Repeats up to max_review_iterations or until quality gate passes.

        Raises:
            ModelExhaustionError: When all models are exhausted.
        """
        stage = PipelineStage.REVIEW

        if self._is_stage_complete(stage):
            console.print("[dim]Review already complete, loading...[/]")
            latest_draft = self.store.load_latest_draft() or draft
            reviews = self.store.load_all_reviews()
            latest_review = (
                reviews[-1]
                if reviews
                else ReviewResult(draft_id=draft.draft_id, draft_version=draft.version)
            )
            return latest_draft, latest_review

        self.state.current_stage = stage
        console.print(Panel("Stage 6: Review & Refinement", style="bold magenta"))

        assert self._reviewer is not None
        assert self._central_summarizer is not None

        current_draft = draft
        latest_review: ReviewResult | None = None
        max_iterations = pipeline_config.max_review_iterations

        for iteration in range(1, max_iterations + 1):
            console.print(f"\n[bold]Review iteration {iteration}/{max_iterations}[/]")

            # Review
            console.print("Running reviewer...")
            review = await self._reviewer.review_draft(
                draft=current_draft,
                style_guide=style_guide,
                chunk_summaries=chunk_summaries,
            )
            latest_review = review

            # Track budget
            self.budget.set_cumulative_usage(
                "reviewer",
                self._reviewer._total_input_tokens,
                self._reviewer._total_output_tokens,
            )

            self.store.save_review(review)

            # Display review summary
            console.print(
                f"Review: Accept={review.total_accept}, "
                f"Edit={review.total_edit}, Reject={review.total_reject}, "
                f"Confidence={review.overall_confidence:.2f}"
            )

            # Quality gate
            review_gate = check_review_quality(review)
            self._display_gate_result(review_gate)

            if review_gate.passed:
                console.print("[green]Review quality gate passed![/]")
                break

            # Check for items needing human review
            conf_gate = check_summary_confidence(chunk_summaries)
            human_items = items_needing_human_review(conf_gate, review)
            if human_items:
                self._write_review_file(
                    f"human_review_iter{iteration}.md",
                    f"Items Requiring Human Review (Iteration {iteration})",
                    human_items,
                )
                console.print(
                    f"[yellow]{len(human_items)} items written to review/ "
                    f"for human review[/]"
                )

            if iteration < max_iterations:
                # Refine
                console.print("Refining draft based on feedback...")
                pre_refine_words = current_draft.total_word_count
                refined_draft = await self._central_summarizer.refine_with_feedback(
                    draft=current_draft,
                    review=review,
                    style_guide=style_guide,
                    chunk_summaries=chunk_summaries,
                )

                self.budget.set_cumulative_usage(
                    "central_summarization",
                    self._central_summarizer._total_input_tokens,
                    self._central_summarizer._total_output_tokens,
                )

                # Guard against catastrophic collapse: if the refined draft lost
                # more than 40% of words the refiner over-corrected. Keep the
                # previous draft in that case and stop iterating.
                collapse_ratio = refined_draft.total_word_count / max(
                    pre_refine_words, 1
                )
                if collapse_ratio < 0.60:
                    console.print(
                        f"[yellow]Warning: refined draft shrank by "
                        f"{(1 - collapse_ratio):.0%} "
                        f"({pre_refine_words} -> {refined_draft.total_word_count} words). "
                        f"Discarding refinement and keeping v{current_draft.version}.[/]"
                    )
                    logger.warning(
                        "Refinement collapsed draft by %.0f%% (%d -> %d words); "
                        "keeping v%d",
                        (1 - collapse_ratio) * 100,
                        pre_refine_words,
                        refined_draft.total_word_count,
                        current_draft.version,
                    )
                    break

                current_draft = refined_draft
                self.store.save_draft(current_draft)
                console.print(
                    f"Refined to draft v{current_draft.version}: "
                    f"{current_draft.total_word_count} words"
                )

        # Generate provenance report
        assert latest_review is not None
        prov_report = generate_provenance_report(
            current_draft, chunk_summaries, latest_review
        )
        self.store.write_review_file("provenance_report.md", prov_report)

        # Run full quality gates
        gate_results = run_all_quality_gates(
            current_draft, chunk_summaries, style_guide, latest_review
        )
        quality_report = generate_quality_report(gate_results)
        self.store.write_review_file("quality_report.md", quality_report)

        self._mark_stage_complete(stage)
        return current_draft, latest_review

    async def stage_generate_slides(
        self,
        draft: MasterDraft,
        style_guide: StyleGuide,
    ) -> SlideOutlineSet:
        """
        Stage 7: Generate slide outlines (with fallback).

        Converts the final master draft into 80-100 slide outlines
        ready for PowerPoint creation. Saves incrementally per section.

        Raises:
            ModelExhaustionError: When all models are exhausted.
        """
        stage = PipelineStage.SLIDE_GENERATION

        if self._is_stage_complete(stage):
            console.print("[dim]Slide generation already complete, loading...[/]")
            outlines = self.store.load_slide_outlines()
            if outlines:
                return outlines

        self.state.current_stage = stage
        console.print(Panel("Stage 7: Slide Generation", style="bold magenta"))

        assert self._slide_generator is not None

        # Load any slide sections already persisted from a prior partial run
        existing_slide_sections = self.store.load_all_slide_sections()
        if existing_slide_sections:
            console.print(
                f"[dim]Resuming slide generation: {len(existing_slide_sections)} "
                f"sections already done, skipping those.[/]"
            )

        console.print("Generating slide outlines...")

        # Per-section callback: save slides to disk immediately
        async def _on_section_done(
            slides: list[Any], heading: str, status: str
        ) -> None:
            self.store.save_slide_section(heading, slides)
            self.budget.set_cumulative_usage(
                "slide_generation",
                self._slide_generator._total_input_tokens,  # type: ignore[union-attr]
                self._slide_generator._total_output_tokens,  # type: ignore[union-attr]
            )
            self._save_checkpoint()
            console.print(f"  [dim]{len(slides)} slides for: {heading[:40]}[/]")

        outlines = await self._slide_generator.generate_outlines(
            draft=draft,
            style_guide=style_guide,
            completed_sections=existing_slide_sections,
            on_section_done=_on_section_done,
        )

        # Final budget snapshot
        self.budget.set_cumulative_usage(
            "slide_generation",
            self._slide_generator._total_input_tokens,
            self._slide_generator._total_output_tokens,
        )

        self.store.save_slide_outlines(outlines)
        # Full outline set saved — discard incremental section files
        self.store.clear_slide_sections()

        # Display slide summary
        console.print(
            f"[green]Generated {outlines.total_slides} slide outlines "
            f"(target: {pipeline_config.target_slide_count_min}-"
            f"{pipeline_config.target_slide_count_max})[/]"
        )

        self._mark_stage_complete(stage)
        return outlines

    # ------------------------------------------------------------------
    # Full pipeline execution
    # ------------------------------------------------------------------

    async def run(self, preprocess_model: str | None = None) -> None:
        """Run the full pipeline."""
        try:
            await self.startup()
            if preprocess_model and self._preprocessor:
                from src.config import MODELS, ModelConfig

                # Build a ModelConfig from the full_id string "provider/model"
                parts = preprocess_model.split("/", 1)
                if len(parts) == 2:
                    override = ModelConfig(
                        provider_id=parts[0],
                        model_id=parts[1],
                        context_window=1_048_576,
                        max_output=65_536,
                        supports_images=True,
                        supports_pdf=True,
                    )
                    self._preprocessor.set_active_model(override)
                    console.print(
                        f"[yellow]Preprocessing model overridden: {preprocess_model}[/]"
                    )

            # Stage 1: Parse & Preprocess
            parse_results = await self.stage_parse_and_preprocess()
            if not await self._checkpoint("Parse & Preprocess", "Chunking"):
                return

            # Stage 2: Chunk
            chunks = await self.stage_chunk(parse_results)
            if not await self._checkpoint("Chunking", "Chunk Summarization"):
                return

            # Stage 3: Summarize Chunks
            chunk_summaries = await self.stage_summarize_chunks(chunks)
            if not await self._checkpoint("Chunk Summarization", "Style Learning"):
                return

            # Stage 4: Learn Style
            style_guide = await self.stage_learn_style(parse_results)
            if not await self._checkpoint("Style Learning", "Central Synthesis"):
                return

            # Stage 5: Central Synthesis
            draft = await self.stage_synthesize(chunk_summaries, style_guide)
            if not await self._checkpoint("Central Synthesis", "Review & Refinement"):
                return

            # Stage 6: Review & Refinement
            final_draft, review = await self.stage_review(
                draft, style_guide, chunk_summaries
            )
            if not await self._checkpoint("Review & Refinement", "Slide Generation"):
                return

            # Stage 7: Slide Generation
            slides = await self.stage_generate_slides(final_draft, style_guide)

            # Final report
            self._display_final_report(final_draft, review, slides)

        except ModelExhaustionError as exc:
            self._handle_exhaustion(exc, self.state.current_stage.value)
        except KeyboardInterrupt:
            console.print("\n[yellow]Pipeline interrupted. Saving state...[/]")
        except Exception as exc:
            console.print(f"\n[red]Pipeline error: {exc}[/]")
            logger.exception("Pipeline failed")
        finally:
            await self.shutdown()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _display_parse_summary(self, results: list[DocumentParseResult]) -> None:
        """Display a summary table of parsed documents."""
        table = Table(title="Parsed Documents")
        table.add_column("File", style="cyan")
        table.add_column("Artifacts", justify="right")
        table.add_column("Text", justify="right")
        table.add_column("Tables", justify="right")
        table.add_column("Images", justify="right")

        for r in results:
            table.add_row(
                r.source_file[:45],
                str(len(r.artifacts)),
                str(r.total_text_length),
                str(r.total_tables),
                str(r.total_images),
            )
        console.print(table)

    def _display_chunk_summary(self, chunks: list[Any]) -> None:
        """Display chunk statistics."""
        if not chunks:
            return
        total_tokens = sum(c.token_count for c in chunks)
        avg_tokens = total_tokens // len(chunks)
        console.print(
            f"  Total tokens: {total_tokens:,} | "
            f"Avg per chunk: {avg_tokens:,} | "
            f"Min: {min(c.token_count for c in chunks):,} | "
            f"Max: {max(c.token_count for c in chunks):,}"
        )

    def _display_style_guide(self, guide: StyleGuide) -> None:
        """Display style guide summary."""
        console.print(f"  Tone: {guide.tone_description[:80]}")
        console.print(f"  Target reader: {guide.target_reader[:80]}")
        console.print(f"  Sections: {len(guide.section_order)}")
        console.print(f"  Rules: {len(guide.rules)}")
        console.print(f"  Checklist items: {len(guide.reviewer_checklist)}")

    def _display_gate_result(self, gate: QualityGateResult) -> None:
        """Display a quality gate result."""
        color = "green" if gate.passed else "red"
        console.print(
            f"  [{color}]Quality Gate [{gate.gate_name}]: "
            f"{'PASS' if gate.passed else 'FAIL'} "
            f"({gate.items_passed}/{gate.items_checked}, "
            f"score={gate.score:.2f})[/{color}]"
        )

    def _display_budget(self) -> None:
        """Display current budget/usage summary."""
        table = Table(title="Token Budget")
        table.add_column("Role", style="cyan")
        table.add_column("Calls", justify="right")
        table.add_column("Input", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("Cost", justify="right")

        for role, entry in sorted(self.budget.usage.items()):
            table.add_row(
                role,
                str(entry.call_count),
                f"{entry.input_tokens:,}",
                f"{entry.output_tokens:,}",
                f"${entry.cost_usd:.4f}",
            )

        table.add_row(
            "[bold]Total[/]",
            str(self.budget.total_calls),
            f"{self.budget.total_input_tokens:,}",
            f"{self.budget.total_output_tokens:,}",
            f"[bold]${self.budget.total_cost_usd:.4f}[/]",
        )

        console.print(table)

    def _display_final_report(
        self,
        draft: MasterDraft,
        review: ReviewResult,
        slides: SlideOutlineSet,
    ) -> None:
        """Display the final pipeline report."""
        console.print(
            Panel(
                f"[bold green]Pipeline Complete![/]\n\n"
                f"Draft: v{draft.version}, {len(draft.sections)} sections, "
                f"{draft.total_word_count} words\n"
                f"Review: confidence {review.overall_confidence:.2f}, "
                f"{review.total_accept} accepted, "
                f"{review.total_edit} edited, "
                f"{review.total_reject} rejected\n"
                f"Slides: {slides.total_slides} outlines "
                f"(target: {pipeline_config.target_slide_count_min}-"
                f"{pipeline_config.target_slide_count_max})\n\n"
                f"Total tokens: {self.budget.total_tokens:,}\n"
                f"Total cost: ${self.budget.total_cost_usd:.4f}\n\n"
                f"Outputs saved to: output/\n"
                f"Review files: review/",
                title="Final Report",
                style="bold green",
            )
        )

        # Save budget report
        budget_report = self.budget.format_report()
        self.store.write_review_file("budget_report.md", budget_report)

    # ------------------------------------------------------------------
    # Human review file generation
    # ------------------------------------------------------------------

    def _write_review_file(
        self,
        filename: str,
        title: str,
        items: list[dict[str, Any]],
    ) -> None:
        """Write items needing human review to a markdown file."""
        lines = [
            f"# {title}",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "---",
            "",
            "Mark this file as reviewed by adding `REVIEWED: YES` at the bottom.",
            "",
        ]

        for i, item in enumerate(items, 1):
            lines.append(f"## Item {i}")
            for key, value in item.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        content = "\n".join(lines)
        self.store.write_review_file(filename, content)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Set up rich logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                show_path=False,
                rich_tracebacks=True,
            )
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _is_failed_image(artifact: Any) -> bool:
    """
    Return True if an image artifact needs (re-)preprocessing.

    Primary check: preprocess_status metadata stamped by the new code.
    Legacy fallback: no content_type in metadata means it was never described
    (prior-run artifacts before status tracking was added).
    """
    status = artifact.metadata.get("preprocess_status")
    if status == ImagePreprocessStatus.SUCCESS:
        return False
    if status and status.startswith("failed_"):
        return True
    # Legacy heuristic: image artifact with no description yet
    if artifact.artifact_type == ArtifactType.IMAGE and not artifact.metadata.get(
        "content_type"
    ):
        return True
    # Low confidence with no status = likely a silent failure from old code
    if artifact.confidence <= 0.3 and not status:
        return True
    return False


def _is_low_confidence_image(artifact: Any, threshold: float) -> bool:
    """
    Return True if an image was described successfully but with confidence
    below the given threshold.

    Only targets artifacts with preprocess_status == success, so it is
    completely disjoint from _is_failed_image — running both flags on the
    same dataset will never double-process the same image.
    """
    status = artifact.metadata.get("preprocess_status")
    if status != ImagePreprocessStatus.SUCCESS:
        return False
    if not artifact.image_base64:
        return False
    return artifact.confidence < threshold


async def _retry_failed_images_async(
    all_runs: bool = False,
    preprocess_model: str | None = None,
) -> None:
    """
    Async impl of --retry-failed-images mode.

    Loads all saved parse results, identifies failed/unprocessed image artifacts,
    and runs preprocessing only on those using the current model config (with
    fallback support). Persists results after each image.

    Args:
        all_runs: If True, retry images failed in any prior run, not just the
            most recent one.
        preprocess_model: Optional "provider/model" string to override the
            default preprocessing model for this retry run only. Successfully
            described images are never re-processed regardless of this setting.
    """
    from src.config import PREPROCESSED_DIR
    from src.storage.vector_store import VectorStore

    model_note = f" (model: {preprocess_model})" if preprocess_model else ""
    console.print(
        Panel(
            f"[bold]Retry Failed Images Mode[/]\n"
            f"Reprocessing only failed/pending image descriptions{model_note}.",
            style="bold yellow",
        )
    )

    store = DocumentStore()
    client = OpenCodeClient()
    await client.start()
    preprocessor = PreprocessorAgent(client)

    if preprocess_model:
        parts = preprocess_model.split("/", 1)
        if len(parts) == 2:
            from src.config import ModelConfig

            override = ModelConfig(
                provider_id=parts[0],
                model_id=parts[1],
                context_window=1_048_576,
                max_output=65_536,
                supports_images=True,
                supports_pdf=True,
            )
            preprocessor.set_active_model(override)
            console.print(
                f"[yellow]Preprocessing model overridden: {preprocess_model}[/]"
            )

    try:
        results = store.load_all_parse_results()
        if not results:
            console.print(
                "[red]No parse results found in output/preprocessed/. "
                "Run the full pipeline first.[/]"
            )
            return

        total_failed = sum(
            1
            for r in results
            for a in r.artifacts
            if a.image_base64 and _is_failed_image(a)
        )

        if total_failed == 0:
            console.print("[green]No failed images found. Nothing to retry.[/]")
            return

        console.print(
            f"Found [bold]{total_failed}[/] failed/pending images across "
            f"{len(results)} documents."
        )

        recovered = 0
        still_failed = 0

        for result in results:
            failed_ids = {
                a.artifact_id
                for a in result.artifacts
                if a.image_base64 and _is_failed_image(a)
            }
            if not failed_ids:
                continue

            logger.info("Retrying %d images in %s", len(failed_ids), result.source_file)

            before_success = sum(
                1
                for a in result.artifacts
                if a.metadata.get("preprocess_status") == ImagePreprocessStatus.SUCCESS
            )

            async def _save(r: DocumentParseResult) -> None:
                store.save_parse_result(r)

            try:
                await preprocessor.process_document(
                    result,
                    target_artifact_ids=failed_ids,
                    on_image_done=_save,
                )
            except ModelExhaustionError as exc:
                console.print(
                    f"[red]Model exhaustion during retry: {exc}[/]\n"
                    f"Partial progress saved."
                )
                break

            after_success = sum(
                1
                for a in result.artifacts
                if a.metadata.get("preprocess_status") == ImagePreprocessStatus.SUCCESS
            )
            doc_recovered = after_success - before_success
            doc_still_failed = len(failed_ids) - doc_recovered
            recovered += doc_recovered
            still_failed += doc_still_failed

            logger.info(
                "%s: recovered=%d, still_failed=%d",
                result.source_file,
                doc_recovered,
                doc_still_failed,
            )

        console.print(
            Panel(
                f"Retry complete.\n"
                f"  Recovered: [bold green]{recovered}[/]\n"
                f"  Still failed: [bold red]{still_failed}[/]\n"
                f"  Total token cost: "
                f"input={preprocessor._total_input_tokens:,}, "
                f"output={preprocessor._total_output_tokens:,}",
                title="Retry Summary",
                style="cyan",
            )
        )

    finally:
        await client.stop()


async def _retry_low_confidence_async(
    threshold: float = 0.5,
    preprocess_model: str | None = None,
) -> None:
    """
    Async impl of --retry-low-confidence mode.

    Loads all saved parse results, identifies image artifacts that were
    successfully described but with confidence below ``threshold``, and
    re-runs preprocessing only on those. Never touches failed/pending
    images (use --retry-failed-images for those).

    Args:
        threshold: Re-process images whose confidence is strictly below
            this value. Default 0.5.
        preprocess_model: Optional "provider/model" override.
    """
    model_note = f" (model: {preprocess_model})" if preprocess_model else ""
    console.print(
        Panel(
            f"[bold]Retry Low-Confidence Images Mode[/]\n"
            f"Re-describing images with confidence < {threshold}{model_note}.",
            style="bold yellow",
        )
    )

    store = DocumentStore()
    client = OpenCodeClient()
    await client.start()
    preprocessor = PreprocessorAgent(client)

    if preprocess_model:
        parts = preprocess_model.split("/", 1)
        if len(parts) == 2:
            from src.config import ModelConfig

            override = ModelConfig(
                provider_id=parts[0],
                model_id=parts[1],
                context_window=1_048_576,
                max_output=65_536,
                supports_images=True,
                supports_pdf=True,
            )
            preprocessor.set_active_model(override)
            console.print(
                f"[yellow]Preprocessing model overridden: {preprocess_model}[/]"
            )

    try:
        results = store.load_all_parse_results()
        if not results:
            console.print(
                "[red]No parse results found in output/preprocessed/. "
                "Run the full pipeline first.[/]"
            )
            return

        total_low_conf = sum(
            1
            for r in results
            for a in r.artifacts
            if _is_low_confidence_image(a, threshold)
        )

        if total_low_conf == 0:
            console.print(
                f"[green]No images found with confidence < {threshold}. Nothing to retry.[/]"
            )
            return

        console.print(
            f"Found [bold]{total_low_conf}[/] low-confidence images "
            f"(confidence < {threshold}) across {len(results)} documents."
        )

        recovered = 0
        still_low = 0

        for result in results:
            low_conf_ids = {
                a.artifact_id
                for a in result.artifacts
                if _is_low_confidence_image(a, threshold)
            }
            if not low_conf_ids:
                continue

            logger.info(
                "Retrying %d low-confidence images in %s",
                len(low_conf_ids),
                result.source_file,
            )

            before_high_conf = sum(
                1
                for a in result.artifacts
                if a.metadata.get("preprocess_status") == ImagePreprocessStatus.SUCCESS
                and a.confidence >= threshold
            )

            async def _save(r: DocumentParseResult) -> None:
                store.save_parse_result(r)

            try:
                await preprocessor.process_document(
                    result,
                    target_artifact_ids=low_conf_ids,
                    on_image_done=_save,
                )
            except ModelExhaustionError as exc:
                console.print(
                    f"[red]Model exhaustion during retry: {exc}[/]\n"
                    f"Partial progress saved."
                )
                break

            after_high_conf = sum(
                1
                for a in result.artifacts
                if a.metadata.get("preprocess_status") == ImagePreprocessStatus.SUCCESS
                and a.confidence >= threshold
            )
            doc_recovered = after_high_conf - before_high_conf
            doc_still_low = len(low_conf_ids) - doc_recovered
            recovered += doc_recovered
            still_low += doc_still_low

            logger.info(
                "%s: improved=%d, still_low_conf=%d",
                result.source_file,
                doc_recovered,
                doc_still_low,
            )

        console.print(
            Panel(
                f"Retry complete.\n"
                f"  Improved (now ≥ {threshold}): [bold green]{recovered}[/]\n"
                f"  Still below threshold: [bold yellow]{still_low}[/]\n"
                f"  Total token cost: "
                f"input={preprocessor._total_input_tokens:,}, "
                f"output={preprocessor._total_output_tokens:,}",
                title="Low-Confidence Retry Summary",
                style="cyan",
            )
        )

    finally:
        await client.stop()


async def _retry_failed_chunks_async() -> None:
    """
    Retry only failed/low-confidence chunk summaries.

    Loads all saved chunks and summaries, identifies chunks whose
    summaries are missing or failed (confidence == 0.0), and re-runs
    summarization only on those.
    """
    console.print(
        Panel(
            "[bold]Retry Failed Chunks Mode[/]\n"
            "Re-summarizing only failed/missing chunk summaries.",
            style="bold yellow",
        )
    )

    store = DocumentStore()
    client = OpenCodeClient()
    await client.start()
    summarizer = ChunkSummarizerAgent(client)

    try:
        all_chunks = store.load_all_chunks()
        if not all_chunks:
            console.print("[red]No chunks found. Run the full pipeline first.[/]")
            return

        existing_summaries = store.load_all_chunk_summaries()
        successful_ids = {
            s.chunk_id
            for s in existing_summaries
            if s.confidence > 0.0 and not s.summary.startswith("[SUMMARIZATION FAILED")
        }

        pending_chunks = [c for c in all_chunks if c.chunk_id not in successful_ids]

        if not pending_chunks:
            console.print("[green]No failed chunks found. Nothing to retry.[/]")
            return

        console.print(
            f"Found [bold]{len(pending_chunks)}[/] failed/missing chunks "
            f"out of {len(all_chunks)} total."
        )

        recovered = 0
        still_failed = 0

        async def _on_chunk_done(summary: ChunkSummary, status: str) -> None:
            nonlocal recovered, still_failed
            store.save_chunk_summary(summary)
            if status == ItemStatus.SUCCESS:
                recovered += 1
            else:
                still_failed += 1

        try:
            await summarizer.summarize_chunks(
                pending_chunks,
                on_chunk_done=_on_chunk_done,
            )
        except ModelExhaustionError as exc:
            console.print(
                f"[red]Model exhaustion during retry: {exc}[/]\nPartial progress saved."
            )

        console.print(
            Panel(
                f"Retry complete.\n"
                f"  Recovered: [bold green]{recovered}[/]\n"
                f"  Still failed: [bold red]{still_failed}[/]\n"
                f"  Total token cost: "
                f"input={summarizer._total_input_tokens:,}, "
                f"output={summarizer._total_output_tokens:,}",
                title="Retry Summary",
                style="cyan",
            )
        )

    finally:
        await client.stop()


def main() -> None:
    """CLI entry point for the summarization pipeline."""
    parser = argparse.ArgumentParser(
        prog="summarizer",
        description="Master Summarizer — multi-agent POT document pipeline",
    )
    # ------------------------------------------------------------------
    # Archive commands
    # ------------------------------------------------------------------
    parser.add_argument(
        "--export-run",
        action="store_true",
        help=(
            "Export the current run as a tar.gz archive for backup or sharing. "
            "By default excludes preprocessed/ (918 MB+). Use --full to include it."
        ),
    )
    parser.add_argument(
        "--import-run",
        metavar="ARCHIVE",
        default=None,
        help=(
            "Import a previously exported run archive (.tar.gz), restoring "
            "output/ and review/ directories. Requires --force if output/ "
            "already contains data."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="With --export-run: include output/preprocessed/ in the archive.",
    )
    parser.add_argument(
        "--output-path",
        metavar="PATH",
        default=None,
        help="With --export-run: explicit path for the archive file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --import-run: overwrite existing output/ data. "
        "With --clean: skip the confirmation prompt.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete all run data: output/, generated review/ files, and data/chroma/. "
            "Preserves input/ and git-tracked files. "
            "Prompts for confirmation unless --force is also passed."
        ),
    )

    # ------------------------------------------------------------------
    # Retry commands
    # ------------------------------------------------------------------
    parser.add_argument(
        "--retry-failed-images",
        action="store_true",
        help=(
            "Retry only failed/pending image descriptions using saved parse results. "
            "Does not re-run parsing or other pipeline stages."
        ),
    )
    parser.add_argument(
        "--retry-failed-chunks",
        action="store_true",
        help=(
            "Retry only failed/low-confidence chunk summaries. "
            "Does not re-run parsing, chunking, or other stages."
        ),
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="With --retry-failed-images: include images failed in any prior run, "
        "not just the most recent one.",
    )
    parser.add_argument(
        "--preprocess-model",
        metavar="PROVIDER/MODEL",
        default=None,
        help=(
            "Override the preprocessing model for this run only. "
            "Example: --preprocess-model google/gemini-3-pro-preview"
        ),
    )
    parser.add_argument(
        "--retry-low-confidence",
        action="store_true",
        help=(
            "Retry image descriptions that succeeded but with confidence below "
            "--confidence-threshold (default 0.5). Only targets images with "
            "preprocess_status=success — never overlaps with --retry-failed-images."
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Confidence threshold for --retry-low-confidence (default: 0.5).",
    )
    args = parser.parse_args()

    _configure_logging()

    console.print(
        Panel(
            "[bold]Master Summarizer[/]\n"
            "Multi-agent document summarization pipeline\n"
            "POT Uribia, La Guajira",
            style="bold blue",
        )
    )

    if args.clean:
        from src.storage.archiver import clean_run

        try:
            clean_run(force=args.force)
        except SystemExit:
            raise
        except Exception as exc:
            console.print(f"[red]Clean failed: {exc}[/]")
            sys.exit(1)
        return

    if args.export_run:
        from src.storage.archiver import export_run

        try:
            output_path = Path(args.output_path) if args.output_path else None
            export_run(full=args.full, output_path=output_path)
        except (FileNotFoundError, RuntimeError) as exc:
            console.print(f"[red]Export failed: {exc}[/]")
            sys.exit(1)
        return

    if args.import_run:
        from src.storage.archiver import import_run

        try:
            import_run(archive_path=Path(args.import_run), force=args.force)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            console.print(f"[red]Import failed: {exc}[/]")
            sys.exit(1)
        return

    if args.retry_failed_images:
        asyncio.run(
            _retry_failed_images_async(
                all_runs=args.all_runs,
                preprocess_model=args.preprocess_model,
            )
        )
        return

    if args.retry_failed_chunks:
        asyncio.run(_retry_failed_chunks_async())
        return

    if args.retry_low_confidence:
        asyncio.run(
            _retry_low_confidence_async(
                threshold=args.confidence_threshold,
                preprocess_model=args.preprocess_model,
            )
        )
        return

    pipeline = Pipeline()
    asyncio.run(pipeline.run(preprocess_model=args.preprocess_model))


if __name__ == "__main__":
    main()
