"""
Run archiver — export and import pipeline run snapshots.

Creates self-contained tar.gz archives of completed (or partial) runs
so they can be stored externally, shared, or restored later.

Archive layout::

    run_YYYY-MM-DD_<short-id>.tar.gz
    ├── manifest.json              ← run metadata + file listing
    ├── output/
    │   ├── pipeline_state.json
    │   ├── summary.md
    │   ├── slides.md
    │   ├── audit_log.jsonl
    │   ├── grd_all_facts_by_threat.*
    │   ├── chunks/
    │   ├── chunk_summaries/
    │   ├── drafts/
    │   ├── style_guide/
    │   ├── reviews/
    │   ├── slides/
    │   └── preprocessed/          ← only with --full
    └── review/
        └── *.md

No external dependencies — uses stdlib tarfile, json, pathlib.
"""

from __future__ import annotations

import json
import logging
import tarfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import OUTPUT_DIR, REVIEW_DIR, PREPROCESSED_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)
console = Console()


# Subdirectories of output/ that are always included in a selective export.
_SELECTIVE_DIRS = [
    "chunks",
    "chunk_summaries",
    "drafts",
    "style_guide",
    "reviews",
    "slides",
]

# Top-level files inside output/ that are always included.
_SELECTIVE_FILES = [
    "pipeline_state.json",
    "summary.md",
    "slides.md",
    "audit_log.jsonl",
]

# Top-level files matched by glob (for files like grd_all_facts_by_threat.*).
_SELECTIVE_GLOBS = [
    "grd_all_facts_by_threat.*",
]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _build_manifest(
    *,
    archive_type: str,
    file_listing: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a manifest dict from the current pipeline state and file listing.

    Reads pipeline_state.json for run metadata. If absent (e.g. no run has
    been executed), uses sensible defaults.
    """
    state_path = OUTPUT_DIR / "pipeline_state.json"
    state: dict[str, Any] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    # Try to read project version from pyproject.toml
    project_version = "unknown"
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version"):
                project_version = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

    return {
        "run_id": state.get("active_run_id", "unknown"),
        "state_id": state.get("state_id", "unknown"),
        "started_at": state.get("started_at"),
        "last_checkpoint": state.get("last_checkpoint"),
        "current_stage": state.get("current_stage"),
        "stages_completed": state.get("stages_completed", []),
        "total_documents": state.get("total_documents", 0),
        "total_chunks": state.get("total_chunks", 0),
        "total_summaries": state.get("total_summaries", 0),
        "total_tokens_used": state.get("total_tokens_used", 0),
        "total_cost_usd": state.get("total_cost_usd", 0.0),
        "archive_type": archive_type,
        "archive_created_at": datetime.now().isoformat(),
        "project_version": project_version,
        "total_files": len(file_listing),
        "total_bytes": sum(f["size"] for f in file_listing),
        "files": file_listing,
    }


def _default_archive_name(manifest: dict[str, Any]) -> str:
    """Generate default archive filename from run metadata."""
    started = manifest.get("started_at")
    if started:
        try:
            dt = datetime.fromisoformat(started)
            date_part = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_part = datetime.now().strftime("%Y-%m-%d")
    else:
        date_part = datetime.now().strftime("%Y-%m-%d")

    run_id = manifest.get("run_id", "unknown")
    short_id = run_id[:8] if run_id != "unknown" else "norun"

    return f"run_{date_part}_{short_id}.tar.gz"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _collect_files(full: bool) -> list[Path]:
    """
    Collect all files to include in the archive.

    Returns absolute paths. Skips empty directories.
    """
    files: list[Path] = []

    if not OUTPUT_DIR.exists():
        return files

    # Selective top-level files
    for name in _SELECTIVE_FILES:
        path = OUTPUT_DIR / name
        if path.is_file():
            files.append(path)

    # Selective globs
    for pattern in _SELECTIVE_GLOBS:
        files.extend(sorted(OUTPUT_DIR.glob(pattern)))

    # Selective subdirectories
    for dirname in _SELECTIVE_DIRS:
        dirpath = OUTPUT_DIR / dirname
        if dirpath.is_dir():
            for filepath in sorted(dirpath.rglob("*")):
                if filepath.is_file():
                    files.append(filepath)

    # Full mode: include preprocessed/
    if full and PREPROCESSED_DIR.is_dir():
        for filepath in sorted(PREPROCESSED_DIR.rglob("*")):
            if filepath.is_file():
                files.append(filepath)

    # Review directory
    if REVIEW_DIR.is_dir():
        for filepath in sorted(REVIEW_DIR.rglob("*")):
            if filepath.is_file() and filepath.name != ".DS_Store":
                files.append(filepath)

    return files


def export_run(
    full: bool = False,
    output_path: Path | None = None,
) -> Path:
    """
    Export the current run as a tar.gz archive.

    Args:
        full: If True, include output/preprocessed/ (large).
        output_path: Explicit path for the archive file. If None,
            auto-generates a name in the project root.

    Returns:
        Path to the created archive.

    Raises:
        FileNotFoundError: If output/ does not exist or has no pipeline state.
        RuntimeError: If no files are found to archive.
    """
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {OUTPUT_DIR}\n"
            "Run the pipeline first to generate artifacts."
        )

    # Collect files
    files = _collect_files(full)
    if not files:
        raise RuntimeError("No files found to archive.")

    # Build file listing for manifest
    file_listing: list[dict[str, Any]] = []
    for filepath in files:
        rel = filepath.relative_to(PROJECT_ROOT)
        file_listing.append(
            {
                "path": str(rel),
                "size": filepath.stat().st_size,
            }
        )

    archive_type = "full" if full else "selective"
    manifest = _build_manifest(archive_type=archive_type, file_listing=file_listing)

    # Determine output path
    if output_path is None:
        archive_name = _default_archive_name(manifest)
        output_path = PROJECT_ROOT / archive_name

    # Warn if partial run
    stages = manifest.get("stages_completed", [])
    all_stages = [
        "preprocessing",
        "chunking",
        "chunk_summarization",
        "style_learning",
        "central_summarization",
        "review",
        "slide_generation",
    ]
    if stages and set(stages) != set(all_stages):
        missing = [s for s in all_stages if s not in stages]
        console.print(
            f"[yellow]Warning: Partial run — stages not completed: "
            f"{', '.join(missing)}[/]"
        )

    # Create archive
    console.print(f"Creating archive: {output_path.name}")

    with tarfile.open(output_path, "w:gz") as tar:
        # Add manifest as the first entry
        manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode(
            "utf-8"
        )
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = int(datetime.now().timestamp())
        tar.addfile(info, BytesIO(manifest_bytes))

        # Add all collected files
        for filepath in files:
            arcname = str(filepath.relative_to(PROJECT_ROOT))
            tar.add(filepath, arcname=arcname)

    archive_size = output_path.stat().st_size
    _display_export_summary(manifest, archive_size, output_path)

    return output_path


def _display_export_summary(
    manifest: dict[str, Any],
    archive_size: int,
    archive_path: Path,
) -> None:
    """Print a summary of the exported archive."""
    # Size formatting
    if archive_size >= 1_000_000_000:
        size_str = f"{archive_size / 1_000_000_000:.1f} GB"
    elif archive_size >= 1_000_000:
        size_str = f"{archive_size / 1_000_000:.1f} MB"
    elif archive_size >= 1_000:
        size_str = f"{archive_size / 1_000:.1f} KB"
    else:
        size_str = f"{archive_size} B"

    stages = manifest.get("stages_completed", [])
    started = manifest.get("started_at", "unknown")
    run_id = manifest.get("run_id", "unknown")

    console.print(
        Panel(
            f"[bold green]Archive created successfully[/]\n\n"
            f"File: [bold]{archive_path}[/]\n"
            f"Size: [bold]{size_str}[/] "
            f"({manifest['total_files']} files)\n"
            f"Type: {manifest['archive_type']}\n\n"
            f"Run ID: {run_id[:8]}\n"
            f"Started: {started}\n"
            f"Stages: {len(stages)}/7 complete\n"
            f"Documents: {manifest.get('total_documents', 0)} | "
            f"Chunks: {manifest.get('total_chunks', 0)} | "
            f"Summaries: {manifest.get('total_summaries', 0)}",
            title="Export Complete",
            style="bold green",
        )
    )


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def import_run(
    archive_path: Path,
    force: bool = False,
) -> None:
    """
    Import a run archive, restoring output/ and review/ directories.

    Args:
        archive_path: Path to a .tar.gz archive created by export_run.
        force: If True, overwrite existing output/ data without prompting.

    Raises:
        FileNotFoundError: If the archive file does not exist.
        RuntimeError: If output/ already has data and force is False.
        ValueError: If the archive has no manifest.json.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # Read manifest first (without extracting everything)
    manifest = _read_manifest(archive_path)
    _display_import_preview(manifest, archive_path)

    # Check for existing data
    has_existing = _has_existing_output()
    if has_existing and not force:
        raise RuntimeError(
            "output/ directory already contains data.\n"
            "Use --force to overwrite, or move/delete existing output first."
        )
    if has_existing and force:
        console.print("[yellow]--force specified: overwriting existing output data[/]")

    # Extract
    console.print("Extracting archive...")

    # Security: only extract paths that start with output/ or review/ or manifest.json
    allowed_prefixes = ("output/", "review/", "manifest.json")

    with tarfile.open(archive_path, "r:gz") as tar:
        members_to_extract: list[tarfile.TarInfo] = []
        for member in tar.getmembers():
            # Security check: skip absolute paths and path traversal
            if member.name.startswith("/") or ".." in member.name:
                logger.warning("Skipping unsafe path in archive: %s", member.name)
                continue
            if not member.name.startswith(allowed_prefixes):
                logger.warning("Skipping unexpected path in archive: %s", member.name)
                continue
            members_to_extract.append(member)

        tar.extractall(path=PROJECT_ROOT, members=members_to_extract)

    _display_import_summary(manifest, len(members_to_extract))


def _read_manifest(archive_path: Path) -> dict[str, Any]:
    """Read the manifest.json from an archive without full extraction."""
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            manifest_file = tar.extractfile("manifest.json")
            if manifest_file is None:
                raise ValueError("manifest.json is empty in archive")
            return json.loads(manifest_file.read().decode("utf-8"))
        except KeyError:
            raise ValueError(
                f"No manifest.json found in archive: {archive_path}\n"
                "This does not appear to be a valid run archive."
            )


def _has_existing_output() -> bool:
    """Check if output/ contains any pipeline data."""
    if not OUTPUT_DIR.exists():
        return False
    state = OUTPUT_DIR / "pipeline_state.json"
    if state.exists():
        return True
    # Check for any JSON files in subdirectories
    for subdir in _SELECTIVE_DIRS:
        dirpath = OUTPUT_DIR / subdir
        if dirpath.is_dir() and any(dirpath.glob("*.json")):
            return True
    return False


def _display_import_preview(
    manifest: dict[str, Any],
    archive_path: Path,
) -> None:
    """Display run metadata before extracting."""
    stages = manifest.get("stages_completed", [])
    archive_type = manifest.get("archive_type", "unknown")

    total_bytes = manifest.get("total_bytes", 0)
    if total_bytes >= 1_000_000_000:
        uncompressed_str = f"{total_bytes / 1_000_000_000:.1f} GB"
    elif total_bytes >= 1_000_000:
        uncompressed_str = f"{total_bytes / 1_000_000:.1f} MB"
    else:
        uncompressed_str = f"{total_bytes / 1_000:.1f} KB"

    console.print(
        Panel(
            f"[bold]Importing run from:[/] {archive_path.name}\n\n"
            f"Run ID: {manifest.get('run_id', 'unknown')[:8]}\n"
            f"Started: {manifest.get('started_at', 'unknown')}\n"
            f"Last checkpoint: {manifest.get('last_checkpoint', 'unknown')}\n"
            f"Stages: {len(stages)}/7 — {', '.join(stages) if stages else 'none'}\n"
            f"Archive type: {archive_type}\n"
            f"Files: {manifest.get('total_files', 0)} "
            f"({uncompressed_str} uncompressed)\n\n"
            f"Documents: {manifest.get('total_documents', 0)} | "
            f"Chunks: {manifest.get('total_chunks', 0)} | "
            f"Summaries: {manifest.get('total_summaries', 0)}",
            title="Run Archive",
            style="bold cyan",
        )
    )


def _display_import_summary(manifest: dict[str, Any], files_extracted: int) -> None:
    """Print a summary after import completes."""
    stages = manifest.get("stages_completed", [])
    all_stages = [
        "preprocessing",
        "chunking",
        "chunk_summarization",
        "style_learning",
        "central_summarization",
        "review",
        "slide_generation",
    ]
    is_complete = set(stages) == set(all_stages)

    status_msg = (
        "[bold green]Complete run[/] — all 7 stages"
        if is_complete
        else f"[yellow]Partial run[/] — {len(stages)}/7 stages. "
        f"Resume with: [bold]summarizer[/]"
    )

    console.print(
        Panel(
            f"[bold green]Import complete[/]\n\n"
            f"Extracted: {files_extracted} files\n"
            f"Status: {status_msg}\n\n"
            f"Output restored to: output/\n"
            f"Review files restored to: review/",
            title="Import Complete",
            style="bold green",
        )
    )
