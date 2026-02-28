"""
PDF document parser.

Extracts text, tables, and images from PDF files (primarily the
style example documents) for style learning and reference.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image

from src.models import (
    ArtifactType,
    DocumentParseResult,
    NormalizedArtifact,
    SourceLocation,
)

logger = logging.getLogger(__name__)


def _extract_page_text(
    page: fitz.Page, source_file: str, page_num: int
) -> list[NormalizedArtifact]:
    """Extract text blocks from a single PDF page."""
    artifacts: list[NormalizedArtifact] = []

    # Get text blocks: (x0, y0, x1, y1, "text", block_no, block_type)
    blocks = page.get_text("blocks")

    for block_idx, block in enumerate(blocks):
        if block[6] == 0:  # text block (not image)
            text = block[4].strip()
            if not text:
                continue

            # Heuristic: detect headings by font size analysis
            is_heading = False
            heading_level = 0
            text_dict = page.get_text("dict", clip=fitz.Rect(block[:4]))
            if text_dict.get("blocks"):
                for b in text_dict["blocks"]:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            size = span.get("size", 12)
                            flags = span.get("flags", 0)
                            is_bold = bool(flags & 2**4)
                            if size >= 16 or (size >= 14 and is_bold):
                                is_heading = True
                                if size >= 20:
                                    heading_level = 1
                                elif size >= 16:
                                    heading_level = 2
                                else:
                                    heading_level = 3

            artifact = NormalizedArtifact(
                artifact_type=ArtifactType.TEXT,
                content=text,
                raw_content=text,
                source=SourceLocation(
                    source_file=source_file,
                    page=page_num + 1,
                    paragraph_index=block_idx,
                ),
                metadata={
                    "is_heading": is_heading,
                    "heading_level": heading_level,
                    "bbox": list(block[:4]),
                },
            )
            artifacts.append(artifact)

    return artifacts


def _extract_page_images(
    page: fitz.Page,
    doc: fitz.Document,
    source_file: str,
    page_num: int,
    output_dir: Path,
    image_counter: int,
) -> tuple[list[NormalizedArtifact], int]:
    """Extract images from a single PDF page."""
    artifacts: list[NormalizedArtifact] = []

    image_list = page.get_images(full=True)

    for img_info in image_list:
        xref = img_info[0]
        try:
            pix = fitz.Pixmap(doc, xref)

            # Convert CMYK to RGB if needed
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            # Skip tiny images (likely decorative)
            if pix.width < 50 or pix.height < 50:
                continue

            # Save image
            img_filename = f"{Path(source_file).stem}_img_{image_counter:04d}.png"
            img_path = output_dir / img_filename

            # Convert to PNG bytes
            img_data = pix.tobytes("png")
            img_path.write_bytes(img_data)

            # Base64 encode
            b64 = base64.b64encode(img_data).decode("utf-8")

            artifact = NormalizedArtifact(
                artifact_type=ArtifactType.IMAGE,
                content=f"[Image: {img_filename}, {pix.width}x{pix.height}px, page {page_num + 1}]",
                source=SourceLocation(
                    source_file=source_file,
                    page=page_num + 1,
                    image_index=image_counter,
                ),
                image_path=str(img_path),
                image_base64=b64,
                metadata={
                    "width": pix.width,
                    "height": pix.height,
                    "xref": xref,
                    "size_bytes": len(img_data),
                },
            )
            artifacts.append(artifact)
            image_counter += 1

        except Exception as exc:
            logger.warning(
                "Failed to extract image xref=%d from %s page %d: %s",
                xref,
                source_file,
                page_num + 1,
                exc,
            )

    return artifacts, image_counter


def _extract_tables(
    page: fitz.Page, source_file: str, page_num: int
) -> list[NormalizedArtifact]:
    """
    Attempt to extract tables from a PDF page.

    Uses PyMuPDF's table detection when available, with fallback
    to text-based heuristics.
    """
    artifacts: list[NormalizedArtifact] = []

    try:
        tables = page.find_tables()
        for table_idx, table in enumerate(tables):
            # Extract table data
            table_data_raw = table.extract()
            if not table_data_raw or len(table_data_raw) < 2:
                continue

            # First row as headers
            headers = [
                str(h).strip() if h else f"col_{i}"
                for i, h in enumerate(table_data_raw[0])
            ]
            table_data = []
            for row in table_data_raw[1:]:
                row_dict = {
                    h: str(v).strip() if v else "" for h, v in zip(headers, row)
                }
                table_data.append(row_dict)

            # Text representation
            lines = [" | ".join(headers)]
            lines.append("-" * len(lines[0]))
            for row in table_data_raw[1:]:
                lines.append(" | ".join(str(v).strip() if v else "" for v in row))
            table_text = "\n".join(lines)

            artifact = NormalizedArtifact(
                artifact_type=ArtifactType.TABLE,
                content=table_text,
                raw_content=table_text,
                table_data=table_data,
                source=SourceLocation(
                    source_file=source_file,
                    page=page_num + 1,
                    table_index=table_idx,
                ),
                metadata={
                    "rows": len(table_data),
                    "columns": len(headers),
                },
            )
            artifacts.append(artifact)

    except Exception as exc:
        logger.debug(
            "Table extraction failed for %s page %d: %s", source_file, page_num + 1, exc
        )

    return artifacts


def parse_pdf(
    filepath: Path, images_output_dir: Path | None = None
) -> DocumentParseResult:
    """
    Parse a PDF file into normalized artifacts.

    Extracts text blocks, tables, and images with structural metadata.

    Args:
        filepath: Path to the PDF file.
        images_output_dir: Directory to save extracted images.

    Returns:
        DocumentParseResult with all extracted artifacts.
    """
    source_file = filepath.name
    logger.info("Parsing PDF: %s (%d bytes)", source_file, filepath.stat().st_size)

    if images_output_dir is None:
        from src.config import PREPROCESSED_DIR

        images_output_dir = PREPROCESSED_DIR / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(filepath))
    artifacts: list[NormalizedArtifact] = []
    heading_structure: list[dict[str, Any]] = []
    total_text_length = 0
    total_tables = 0
    image_counter = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract text
        text_artifacts = _extract_page_text(page, source_file, page_num)
        for a in text_artifacts:
            if a.metadata.get("is_heading"):
                heading_structure.append(
                    {
                        "level": a.metadata.get("heading_level", 1),
                        "text": a.content,
                        "page": page_num + 1,
                    }
                )
            total_text_length += len(a.content)
        artifacts.extend(text_artifacts)

        # Extract tables
        table_artifacts = _extract_tables(page, source_file, page_num)
        total_tables += len(table_artifacts)
        artifacts.extend(table_artifacts)

        # Extract images
        image_artifacts, image_counter = _extract_page_images(
            page,
            doc,
            source_file,
            page_num,
            images_output_dir,
            image_counter,
        )
        artifacts.extend(image_artifacts)

    page_count = len(doc)
    doc.close()

    result = DocumentParseResult(
        source_file=source_file,
        title=filepath.stem,
        artifacts=artifacts,
        heading_structure=heading_structure,
        total_text_length=total_text_length,
        total_tables=total_tables,
        total_images=image_counter,
    )

    logger.info(
        "Parsed %s: %d pages, %d artifacts (%d text, %d tables, %d images)",
        source_file,
        page_count,
        len(artifacts),
        len([a for a in artifacts if a.artifact_type == ArtifactType.TEXT]),
        total_tables,
        image_counter,
    )

    return result
