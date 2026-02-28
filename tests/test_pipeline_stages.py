"""
Test pipeline stages 1-2 (non-LLM): Parse & Preprocess, then Chunk.

This verifies the orchestrator wiring and data flow without making
any LLM calls (image preprocessing is skipped if no server).

Run with: uv run python tests/test_pipeline_stages.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.logging import RichHandler

from src.config import (
    ensure_output_dirs,
    get_raw_documents,
    get_style_examples,
    pipeline_config,
    PREPROCESSED_DIR,
    CHUNKS_DIR,
)
from src.models import DocumentParseResult, PipelineStage
from src.parsers.docx_parser import parse_docx
from src.parsers.pdf_parser import parse_pdf
from src.agents.chunker import Chunker
from src.storage.document_store import DocumentStore
from src.storage.vector_store import VectorStore

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("test_pipeline")


async def main() -> None:
    ensure_output_dirs()

    store = DocumentStore()
    vector_store = VectorStore()
    vector_store.initialize()

    # ----------------------------------------------------------------
    # Stage 1: Parse
    # ----------------------------------------------------------------
    console.print("\n[bold magenta]═══ Stage 1: Parse & Preprocess ═══[/]")

    raw_docs = get_raw_documents()
    style_docs = get_style_examples()
    console.print(f"Found {len(raw_docs)} DOCX files, {len(style_docs)} PDF files")

    results: list[DocumentParseResult] = []

    # Parse DOCX
    for doc_path in raw_docs:
        console.print(
            f"  Parsing DOCX: {doc_path.name} ({doc_path.stat().st_size / 1e6:.1f} MB)..."
        )
        try:
            result = parse_docx(doc_path)
            results.append(result)
            console.print(
                f"    [green]OK[/]: {len(result.artifacts)} artifacts, "
                f"{result.total_text_length} chars, "
                f"{result.total_tables} tables, {result.total_images} images"
            )
        except Exception as exc:
            console.print(f"    [red]FAIL[/]: {exc}")

    # Parse PDFs
    for pdf_path in style_docs:
        console.print(
            f"  Parsing PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1e6:.1f} MB)..."
        )
        try:
            result = parse_pdf(pdf_path)
            results.append(result)
            console.print(
                f"    [green]OK[/]: {len(result.artifacts)} artifacts, "
                f"{result.total_text_length} chars, "
                f"{result.total_tables} tables, {result.total_images} images"
            )
        except Exception as exc:
            console.print(f"    [red]FAIL[/]: {exc}")

    # Save parse results
    for result in results:
        store.save_parse_result(result)
    console.print(
        f"\n[green]Saved {len(results)} parse results to {PREPROCESSED_DIR}[/]"
    )

    # Skip image preprocessing (would need LLM)
    total_images = sum(
        1
        for r in results
        for a in r.artifacts
        if a.artifact_type.value == "image" and a.image_base64
    )
    console.print(
        f"[yellow]Skipping image preprocessing ({total_images} images would be processed by Gemini)[/]"
    )

    # ----------------------------------------------------------------
    # Stage 2: Chunk
    # ----------------------------------------------------------------
    console.print("\n[bold magenta]═══ Stage 2: Chunking ═══[/]")

    chunker = Chunker(vector_store=vector_store)

    # Only chunk raw data documents (not style PDFs)
    raw_filenames = {p.name for p in raw_docs}
    data_results = [r for r in results if r.source_file in raw_filenames]
    console.print(
        f"Chunking {len(data_results)} data documents (skipping {len(results) - len(data_results)} style examples)"
    )

    all_chunks = []
    for result in data_results:
        chunks = chunker.chunk_document(result)
        all_chunks.extend(chunks)
        store.save_chunks(chunks)

        token_counts = [c.token_count for c in chunks]
        console.print(
            f"  {result.source_file[:50]}: {len(chunks)} chunks, "
            f"tokens: min={min(token_counts)}, max={max(token_counts)}, "
            f"avg={sum(token_counts) // len(token_counts)}"
        )

    console.print(
        f"\n[green]Created {len(all_chunks)} chunks total, saved to {CHUNKS_DIR}[/]"
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    console.print("\n[bold cyan]═══ Pipeline Stage 1-2 Summary ═══[/]")
    console.print(
        f"  Documents parsed: {len(results)} ({len(raw_docs)} DOCX + {len(style_docs)} PDF)"
    )
    console.print(f"  Total artifacts: {sum(len(r.artifacts) for r in results)}")
    console.print(f"  Total text chars: {sum(r.total_text_length for r in results):,}")
    console.print(f"  Total tables: {sum(r.total_tables for r in results)}")
    console.print(f"  Total images: {sum(r.total_images for r in results)}")
    console.print(f"  Chunks created: {len(all_chunks)}")
    console.print(f"  Total chunk tokens: {sum(c.token_count for c in all_chunks):,}")
    console.print(f"  ChromaDB entries: {vector_store.count()}")

    # Verify store reload works
    console.print("\n[dim]Verifying store reload...[/]")
    loaded_results = store.load_all_parse_results()
    loaded_chunks = store.load_all_chunks()
    console.print(
        f"  Reloaded {len(loaded_results)} parse results, {len(loaded_chunks)} chunks"
    )

    if len(loaded_results) == len(results) and len(loaded_chunks) == len(all_chunks):
        console.print("[green]Store reload: PASS[/]")
    else:
        console.print("[red]Store reload: MISMATCH[/]")

    console.print(
        "\n[bold green]Stages 1-2 complete. Ready for manual review before LLM stages.[/]"
    )


if __name__ == "__main__":
    asyncio.run(main())
