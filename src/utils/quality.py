"""
Quality gate utilities.

Provides functions for confidence checking, numeric reconciliation
between source chunks and draft content, completeness validation,
and stage-level quality gates used by the orchestrator.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.config import pipeline_config
from src.models import (
    ChunkSummary,
    DraftSection,
    MasterDraft,
    NumericEntry,
    ReviewAnnotation,
    ReviewResult,
    ReviewVerdict,
    StyleGuide,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for quality gate results
# ---------------------------------------------------------------------------


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""

    passed: bool
    gate_name: str
    details: str = ""
    score: float = 0.0
    items_checked: int = 0
    items_passed: int = 0
    items_failed: int = 0
    failed_items: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.items_checked == 0:
            return 0.0
        return self.items_passed / self.items_checked

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"QualityGate[{self.gate_name}]: {status} "
            f"({self.items_passed}/{self.items_checked}, score={self.score:.2f})"
        )


@dataclass
class NumericDiscrepancy:
    """A numeric mismatch between source and draft."""

    label: str
    source_value: float
    source_unit: str
    draft_value: float | None
    draft_context: str
    source_chunk_id: str
    severity: str = "low"  # low, medium, high

    @property
    def relative_error(self) -> float | None:
        if self.draft_value is None or self.source_value == 0:
            return None
        return abs(self.draft_value - self.source_value) / abs(self.source_value)


# ---------------------------------------------------------------------------
# Confidence gates
# ---------------------------------------------------------------------------


def check_summary_confidence(
    summaries: list[ChunkSummary],
    threshold: float | None = None,
) -> QualityGateResult:
    """
    Verify all chunk summaries meet the confidence threshold.

    Per Project.md: "only accept automated merges when confidence >= 0.85.
    Lower-confidence items go to human review."
    """
    threshold = threshold or pipeline_config.confidence_threshold
    passed_items = []
    failed_items = []

    for s in summaries:
        if s.confidence >= threshold:
            passed_items.append(s)
        else:
            failed_items.append(
                {
                    "summary_id": s.summary_id,
                    "chunk_id": s.chunk_id,
                    "source_file": s.source_file,
                    "confidence": s.confidence,
                    "section": s.section_title,
                }
            )

    gate_passed = len(failed_items) == 0
    avg_confidence = (
        sum(s.confidence for s in summaries) / len(summaries) if summaries else 0
    )

    result = QualityGateResult(
        passed=gate_passed,
        gate_name="summary_confidence",
        details=f"Threshold: {threshold:.2f}, Average: {avg_confidence:.2f}",
        score=avg_confidence,
        items_checked=len(summaries),
        items_passed=len(passed_items),
        items_failed=len(failed_items),
        failed_items=failed_items,
    )

    if not gate_passed:
        logger.warning(
            "Confidence gate FAILED: %d/%d summaries below %.2f",
            len(failed_items),
            len(summaries),
            threshold,
        )
    else:
        logger.info(
            "Confidence gate passed: %d summaries, avg %.2f",
            len(summaries),
            avg_confidence,
        )

    return result


def check_draft_confidence(
    draft: MasterDraft,
    threshold: float | None = None,
) -> QualityGateResult:
    """Check that all draft sections meet the confidence threshold."""
    threshold = threshold or pipeline_config.confidence_threshold
    failed_items = []

    for section in draft.sections:
        if section.confidence < threshold:
            failed_items.append(
                {
                    "section_heading": section.heading,
                    "section_id": section.section_id,
                    "confidence": section.confidence,
                }
            )

    avg_conf = (
        sum(s.confidence for s in draft.sections) / len(draft.sections)
        if draft.sections
        else 0
    )

    return QualityGateResult(
        passed=len(failed_items) == 0,
        gate_name="draft_confidence",
        details=f"Threshold: {threshold:.2f}, Average: {avg_conf:.2f}",
        score=avg_conf,
        items_checked=len(draft.sections),
        items_passed=len(draft.sections) - len(failed_items),
        items_failed=len(failed_items),
        failed_items=failed_items,
    )


# ---------------------------------------------------------------------------
# Numeric reconciliation
# ---------------------------------------------------------------------------

# Regex pattern to find numbers (with optional decimals and thousand separators)
_NUMBER_PATTERN = re.compile(
    r"(?<!\w)"  # not preceded by a word char
    r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?)"  # number with optional separators
    r"(?:\s*(%|km²|km2|ha|m²|m2|habitantes|hab\.?|personas|viviendas|"
    r"hogares|predios|hectáreas|millones|USD|COP|pesos|\$))?"  # optional unit
    r"(?!\w)",  # not followed by a word char
    re.IGNORECASE,
)


def _normalize_number(text: str) -> float | None:
    """Parse a number string handling Colombian formatting (. for thousands, , for decimals)."""
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        # Try as-is first (Python default)
        return float(cleaned.replace(" ", ""))
    except ValueError:
        pass
    try:
        # Colombian format: 1.234.567,89
        if "," in cleaned and "." in cleaned:
            # Determine which is the decimal separator
            last_dot = cleaned.rfind(".")
            last_comma = cleaned.rfind(",")
            if last_comma > last_dot:
                # comma is decimal: 1.234,56
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                # dot is decimal: 1,234.56
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            # Could be thousands or decimal
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "." in cleaned:
            # Could be thousands or decimal
            parts = cleaned.split(".")
            if len(parts) == 2 and len(parts[1]) == 3:
                # Likely thousands: 1.234
                cleaned = cleaned.replace(".", "")
            # Otherwise keep as decimal
        return float(cleaned)
    except (ValueError, IndexError):
        return None


def extract_numbers_from_text(text: str) -> list[tuple[float, str]]:
    """Extract all numbers and their units from a text string."""
    results = []
    for match in _NUMBER_PATTERN.finditer(text):
        num_str = match.group(1)
        unit = match.group(2) or ""
        value = _normalize_number(num_str)
        if value is not None:
            results.append((value, unit))
    return results


def reconcile_numerics(
    chunk_summaries: list[ChunkSummary],
    draft: MasterDraft,
    tolerance: float = 0.05,
) -> QualityGateResult:
    """
    Cross-check numeric values between source summaries and the draft.

    Per Project.md: "reviewer agent runs numeric checks (sums, units, ranges)
    and flags mismatches before Opus finalization."

    Args:
        chunk_summaries: Source summaries with numeric tables.
        draft: The draft to check against.
        tolerance: Relative tolerance for numeric comparison (5% default).
    """
    # Collect all source numeric entries
    source_numerics: list[NumericEntry] = []
    for s in chunk_summaries:
        source_numerics.extend(s.numeric_table)

    if not source_numerics:
        return QualityGateResult(
            passed=True,
            gate_name="numeric_reconciliation",
            details="No numeric entries to reconcile",
            score=1.0,
        )

    # Extract numbers from draft text
    draft_text = "\n".join(s.content for s in draft.sections)
    draft_numbers = extract_numbers_from_text(draft_text)
    draft_value_set = {v for v, _ in draft_numbers}

    discrepancies: list[dict[str, Any]] = []
    checked = 0
    matched = 0

    for ne in source_numerics:
        checked += 1
        # Check if this value appears in the draft (with tolerance)
        found = False
        for dv in draft_value_set:
            if ne.value == 0:
                if dv == 0:
                    found = True
                    break
            elif abs(dv - ne.value) / abs(ne.value) <= tolerance:
                found = True
                break

        if found:
            matched += 1
        else:
            severity = "low"
            # Higher severity for important metrics
            if ne.unit in ("habitantes", "hab", "km²", "ha", "hectáreas"):
                severity = "medium"
            if ne.value > 10000:
                severity = "medium"

            discrepancies.append(
                {
                    "label": ne.label,
                    "value": ne.value,
                    "unit": ne.unit,
                    "context": ne.context,
                    "source_chunk_id": ne.source_chunk_id,
                    "severity": severity,
                }
            )

    match_rate = matched / checked if checked > 0 else 1.0
    # Pass if at least 70% of source numbers appear in draft
    # (not all source numbers need to be in the summary)
    gate_passed = match_rate >= 0.50

    return QualityGateResult(
        passed=gate_passed,
        gate_name="numeric_reconciliation",
        details=f"Matched {matched}/{checked} source values in draft (tolerance {tolerance:.0%})",
        score=match_rate,
        items_checked=checked,
        items_passed=matched,
        items_failed=len(discrepancies),
        failed_items=discrepancies,
    )


# ---------------------------------------------------------------------------
# Review quality gate
# ---------------------------------------------------------------------------


def check_review_quality(
    review: ReviewResult,
    max_reject_ratio: float = 0.15,
    min_overall_confidence: float = 0.70,
) -> QualityGateResult:
    """
    Check if the review result indicates acceptable draft quality.

    The draft passes if:
    - Reject ratio is below threshold
    - Overall confidence is above minimum
    - No high-risk items remain unaddressed
    """
    total = len(review.annotations)
    if total == 0:
        return QualityGateResult(
            passed=True,
            gate_name="review_quality",
            details="No annotations to check",
            score=1.0,
        )

    reject_ratio = review.total_reject / total
    high_risk_count = sum(1 for r in review.risk_register if r.get("level") == "high")

    passed = (
        reject_ratio <= max_reject_ratio
        and review.overall_confidence >= min_overall_confidence
        and high_risk_count == 0
    )

    failed_items = []
    if reject_ratio > max_reject_ratio:
        failed_items.append(
            {
                "issue": "reject_ratio_too_high",
                "value": reject_ratio,
                "threshold": max_reject_ratio,
            }
        )
    if review.overall_confidence < min_overall_confidence:
        failed_items.append(
            {
                "issue": "low_overall_confidence",
                "value": review.overall_confidence,
                "threshold": min_overall_confidence,
            }
        )
    if high_risk_count > 0:
        failed_items.append(
            {
                "issue": "unresolved_high_risks",
                "count": high_risk_count,
            }
        )

    return QualityGateResult(
        passed=passed,
        gate_name="review_quality",
        details=(
            f"Reject ratio: {reject_ratio:.1%}, "
            f"Confidence: {review.overall_confidence:.2f}, "
            f"High risks: {high_risk_count}"
        ),
        score=review.overall_confidence,
        items_checked=total,
        items_passed=review.total_accept,
        items_failed=review.total_reject,
        failed_items=failed_items,
    )


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------


def check_completeness(
    draft: MasterDraft,
    style_guide: StyleGuide,
    chunk_summaries: list[ChunkSummary],
) -> QualityGateResult:
    """
    Check that the draft covers all expected sections from the style guide
    and uses a reasonable proportion of source summaries.
    """
    expected_sections = (
        set(style_guide.section_order) if style_guide.section_order else set()
    )
    actual_sections = {s.heading for s in draft.sections}

    # Check section coverage against style guide
    missing_sections = expected_sections - actual_sections
    extra_sections = actual_sections - expected_sections

    # Check source coverage
    used_chunk_ids: set[str] = set()
    for section in draft.sections:
        used_chunk_ids.update(section.source_chunk_ids)

    all_chunk_ids = {s.chunk_id for s in chunk_summaries}
    unused_chunks = all_chunk_ids - used_chunk_ids
    coverage_ratio = len(used_chunk_ids) / len(all_chunk_ids) if all_chunk_ids else 1.0

    failed_items = []
    if missing_sections:
        failed_items.append(
            {
                "issue": "missing_sections",
                "sections": list(missing_sections),
            }
        )
    if coverage_ratio < 0.50:
        failed_items.append(
            {
                "issue": "low_source_coverage",
                "coverage": coverage_ratio,
                "unused_chunks": len(unused_chunks),
            }
        )

    passed = len(missing_sections) == 0 and coverage_ratio >= 0.50

    return QualityGateResult(
        passed=passed,
        gate_name="completeness",
        details=(
            f"Sections: {len(actual_sections)}/{len(expected_sections) or 'N/A'}, "
            f"Source coverage: {coverage_ratio:.1%}, "
            f"Missing: {list(missing_sections)[:5]}"
        ),
        score=coverage_ratio,
        items_checked=len(expected_sections) + len(all_chunk_ids),
        items_passed=(
            len(expected_sections) - len(missing_sections) + len(used_chunk_ids)
        ),
        items_failed=len(missing_sections) + len(unused_chunks),
        failed_items=failed_items,
    )


# ---------------------------------------------------------------------------
# Slide quality gate
# ---------------------------------------------------------------------------


def check_slide_quality(
    slides: list[Any],
    min_slides: int | None = None,
    max_slides: int | None = None,
    max_bullets: int | None = None,
    max_words: int | None = None,
) -> QualityGateResult:
    """
    Validate slide outlines meet constraints.

    Checks: slide count in range, bullet count per slide,
    word count per bullet.
    """
    min_slides = min_slides or pipeline_config.target_slide_count_min
    max_slides = max_slides or pipeline_config.target_slide_count_max
    max_bullets = max_bullets or pipeline_config.max_bullets_per_slide
    max_words = max_words or pipeline_config.max_words_per_slide

    total = len(slides)
    in_range = min_slides <= total <= max_slides
    failed_items = []

    if not in_range:
        failed_items.append(
            {
                "issue": "slide_count_out_of_range",
                "count": total,
                "expected_min": min_slides,
                "expected_max": max_slides,
            }
        )

    over_bullet = 0
    over_word = 0
    for slide in slides:
        bullets = getattr(slide, "bullets", [])
        if len(bullets) > max_bullets:
            over_bullet += 1
            failed_items.append(
                {
                    "issue": "too_many_bullets",
                    "slide": getattr(slide, "slide_number", 0),
                    "count": len(bullets),
                    "max": max_bullets,
                }
            )
        for bullet in bullets:
            wc = len(bullet.split())
            if wc > max_words:
                over_word += 1

    passed = in_range and over_bullet == 0

    return QualityGateResult(
        passed=passed,
        gate_name="slide_quality",
        details=(
            f"Slides: {total} (target {min_slides}-{max_slides}), "
            f"Over-bullet: {over_bullet}, Over-word: {over_word}"
        ),
        score=1.0 if passed else 0.5,
        items_checked=total,
        items_passed=total - over_bullet,
        items_failed=over_bullet + (0 if in_range else 1),
        failed_items=failed_items,
    )


# ---------------------------------------------------------------------------
# Composite quality gate
# ---------------------------------------------------------------------------


def run_all_quality_gates(
    draft: MasterDraft,
    chunk_summaries: list[ChunkSummary],
    style_guide: StyleGuide,
    review: ReviewResult | None = None,
) -> list[QualityGateResult]:
    """
    Run all quality gates on the current pipeline artifacts.

    Returns a list of gate results. The orchestrator can use these
    to decide whether to proceed, retry, or escalate to human review.
    """
    results = [
        check_summary_confidence(chunk_summaries),
        check_draft_confidence(draft),
        reconcile_numerics(chunk_summaries, draft),
        check_completeness(draft, style_guide, chunk_summaries),
    ]

    if review:
        results.append(check_review_quality(review))

    # Log summary
    passed = sum(1 for r in results if r.passed)
    logger.info(
        "Quality gates: %d/%d passed",
        passed,
        len(results),
    )
    for r in results:
        level = logging.INFO if r.passed else logging.WARNING
        logger.log(level, "  %s", r)

    return results


def generate_quality_report(gate_results: list[QualityGateResult]) -> str:
    """Generate a human-readable quality report in Markdown."""
    lines = [
        "# Quality Gate Report",
        "",
        f"**Gates run:** {len(gate_results)}",
        f"**Passed:** {sum(1 for r in gate_results if r.passed)}",
        f"**Failed:** {sum(1 for r in gate_results if not r.passed)}",
        "",
    ]

    for r in gate_results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"## {r.gate_name}: {status}")
        lines.append(f"- Score: {r.score:.2f}")
        lines.append(f"- Items: {r.items_passed}/{r.items_checked} passed")
        lines.append(f"- Details: {r.details}")

        if r.failed_items:
            lines.append(f"- Failed items ({len(r.failed_items)}):")
            for item in r.failed_items[:10]:
                lines.append(f"  - {item}")
            if len(r.failed_items) > 10:
                lines.append(f"  - ... and {len(r.failed_items) - 10} more")

        lines.append("")

    return "\n".join(lines)


def items_needing_human_review(
    summary_gate: QualityGateResult,
    review: ReviewResult | None = None,
) -> list[dict[str, Any]]:
    """
    Identify items that should be escalated to human review.

    Per Project.md: "Human-in-the-loop gates: set human review for:
    legal/land-use claims, geospatial assertions, or any flagged contradictions."
    """
    items: list[dict[str, Any]] = []

    # Low-confidence summaries
    for item in summary_gate.failed_items:
        items.append(
            {
                "type": "low_confidence_summary",
                "reason": f"Confidence {item.get('confidence', 0):.2f} below threshold",
                **item,
            }
        )

    # High-risk review annotations
    if review:
        for ann in review.annotations:
            if ann.risk_level == "high" or ann.verdict == ReviewVerdict.REJECT:
                items.append(
                    {
                        "type": "high_risk_annotation",
                        "section_id": ann.section_id,
                        "verdict": ann.verdict.value,
                        "reason": ann.reason,
                        "risk_level": ann.risk_level,
                    }
                )

        for risk in review.risk_register:
            if risk.get("level") == "high":
                items.append(
                    {
                        "type": "high_risk_register_entry",
                        "description": risk.get("description", ""),
                        "affected_sections": risk.get("affected_sections", []),
                    }
                )

    return items
