"""
Main orchestrator pipeline.

Coordinates all agents through the full summarization pipeline:
parse → preprocess → chunk → summarize → style learn → synthesize → review → slides.

Supports stage-by-stage execution with manual checkpoints,
resumable state, quality gates, retry logic, and human review
escalation.
"""

from __future__ import annotations

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
    REVIEW_DIR,
    ensure_output_dirs,
    get_raw_documents,
    get_style_examples,
    pipeline_config,
)
from src.models import (
    ChunkSummary,
    DocumentParseResult,
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
        loop = asyncio.get_event_loop()
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
        Stage 1: Parse DOCX and PDF files, then preprocess images.

        - Parse all DOCX files from input/raw_data/
        - Parse all PDF files from input/style_examples/
        - Run multimodal preprocessing on images via Gemini
        """
        stage = PipelineStage.PREPROCESSING

        if self._is_stage_complete(stage):
            console.print("[dim]Preprocessing already complete, loading results...[/]")
            return self.store.load_all_parse_results()

        self.state.current_stage = stage
        console.print(Panel("Stage 1: Parse & Preprocess", style="bold magenta"))

        # Discover input files
        raw_docs = get_raw_documents()
        style_docs = get_style_examples()
        console.print(f"Found {len(raw_docs)} DOCX files, {len(style_docs)} PDF files")

        results: list[DocumentParseResult] = []

        # Parse DOCX files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing DOCX files...", total=len(raw_docs))
            for doc_path in raw_docs:
                try:
                    logger.info("Parsing DOCX: %s", doc_path.name)
                    result = parse_docx(doc_path)
                    results.append(result)
                    progress.update(
                        task, advance=1, description=f"Parsed: {doc_path.name[:40]}"
                    )
                except Exception as exc:
                    logger.error("Failed to parse %s: %s", doc_path.name, exc)
                    console.print(f"[red]Failed to parse {doc_path.name}: {exc}[/]")
                    progress.update(task, advance=1)

        # Parse PDF style examples
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing PDF examples...", total=len(style_docs))
            for pdf_path in style_docs:
                try:
                    logger.info("Parsing PDF: %s", pdf_path.name)
                    result = parse_pdf(pdf_path)
                    results.append(result)
                    progress.update(
                        task, advance=1, description=f"Parsed: {pdf_path.name[:40]}"
                    )
                except Exception as exc:
                    logger.error("Failed to parse %s: %s", pdf_path.name, exc)
                    console.print(f"[red]Failed to parse {pdf_path.name}: {exc}[/]")
                    progress.update(task, advance=1)

        # Preprocess images through Gemini
        assert self._preprocessor is not None
        total_images = sum(
            1
            for r in results
            for a in r.artifacts
            if a.artifact_type.value == "image" and a.image_base64
        )

        if total_images > 0:
            console.print(f"Preprocessing {total_images} images via Gemini...")
            for result in results:
                result = await self._preprocessor.process_document(result)
                # Track budget
                self.budget.record_usage(
                    "preprocessing",
                    self._preprocessor._total_input_tokens,
                    self._preprocessor._total_output_tokens,
                )
        else:
            console.print("[dim]No images to preprocess[/]")

        # Save all results
        for result in results:
            self.store.save_parse_result(result)

        # Display summary
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
        Stage 3: Summarize each chunk via Sonnet.

        Processes chunks in batches with retry logic for
        low-confidence outputs.
        """
        stage = PipelineStage.CHUNK_SUMMARIZATION

        if self._is_stage_complete(stage):
            console.print("[dim]Chunk summarization already complete, loading...[/]")
            return self.store.load_all_chunk_summaries()

        self.state.current_stage = stage
        console.print(Panel("Stage 3: Chunk Summarization", style="bold magenta"))

        assert self._chunk_summarizer is not None
        all_summaries: list[ChunkSummary] = []
        batch_size = pipeline_config.chunk_summary_batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Summarizing chunks...", total=len(chunks))

            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                summaries = await self._chunk_summarizer.summarize_chunks(batch)
                all_summaries.extend(summaries)
                self.store.save_chunk_summaries(summaries)

                # Track budget
                self.budget.record_usage(
                    "chunk_summarization",
                    self._chunk_summarizer._total_input_tokens,
                    self._chunk_summarizer._total_output_tokens,
                )

                progress.update(
                    task,
                    advance=len(batch),
                    description=f"Batch {i // batch_size + 1}: {len(summaries)} summaries",
                )

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
        Stage 4: Learn style from example PDFs via Opus.

        Analyzes style example documents to produce a machine-readable
        style guide for downstream synthesis.
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

        guide = await self._style_learner.learn_style(example_docs)

        # Track budget
        self.budget.record_usage(
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
        Stage 5: Central synthesis via Opus.

        Synthesizes all chunk summaries into a coherent master draft
        following the style guide.
        """
        stage = PipelineStage.CENTRAL_SUMMARIZATION

        if self._is_stage_complete(stage):
            console.print("[dim]Synthesis already complete, loading...[/]")
            draft = self.store.load_latest_draft()
            if draft:
                return draft

        self.state.current_stage = stage
        console.print(Panel("Stage 5: Central Synthesis", style="bold magenta"))

        assert self._central_summarizer is not None

        console.print(
            f"Synthesizing {len(chunk_summaries)} summaries into master draft..."
        )

        draft = await self._central_summarizer.synthesize(
            chunk_summaries=chunk_summaries,
            style_guide=style_guide,
        )

        # Track budget
        self.budget.record_usage(
            "central_summarization",
            self._central_summarizer._total_input_tokens,
            self._central_summarizer._total_output_tokens,
        )

        self.store.save_draft(draft)

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
            self.budget.record_usage(
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
                current_draft = await self._central_summarizer.refine_with_feedback(
                    draft=current_draft,
                    review=review,
                    style_guide=style_guide,
                    chunk_summaries=chunk_summaries,
                )

                self.budget.record_usage(
                    "central_summarization",
                    self._central_summarizer._total_input_tokens,
                    self._central_summarizer._total_output_tokens,
                )

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
        Stage 7: Generate slide outlines.

        Converts the final master draft into 80-100 slide outlines
        ready for PowerPoint creation.
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

        console.print("Generating slide outlines...")
        outlines = await self._slide_generator.generate_outlines(
            draft=draft,
            style_guide=style_guide,
        )

        # Track budget
        self.budget.record_usage(
            "slide_generation",
            self._slide_generator._total_input_tokens,
            self._slide_generator._total_output_tokens,
        )

        self.store.save_slide_outlines(outlines)

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

    async def run(self) -> None:
        """
        Execute the full pipeline with checkpoints between stages.

        Stages:
        1. Parse & Preprocess
        2. Chunk
        3. Summarize Chunks
        4. Learn Style
        5. Central Synthesis
        6. Review & Refinement
        7. Slide Generation
        """
        try:
            await self.startup()

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


def main() -> None:
    """CLI entry point for the summarization pipeline."""
    # Configure logging
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
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    console.print(
        Panel(
            "[bold]Master Summarizer[/]\n"
            "Multi-agent document summarization pipeline\n"
            "POT Uribia, La Guajira",
            style="bold blue",
        )
    )

    pipeline = Pipeline()
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
