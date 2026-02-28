"""
Live integration tests against an OpenCode server.

Run with: uv run python tests/test_live.py

Tests:
1. Structured output parsing (simple schema)
2. ChunkSummarizerAgent on a real chunk
3. Image/file part format with a small test image
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS, pipeline_config
from src.opencode_client import OpenCodeClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("test_live")


# ------------------------------------------------------------------
# Test 1: Structured output with a simple schema
# ------------------------------------------------------------------


async def test_structured_output(client: OpenCodeClient) -> bool:
    """Test that structured output works and _parse_response extracts it."""
    logger.info("=" * 60)
    logger.info("TEST 1: Structured output parsing")
    logger.info("=" * 60)

    schema = {
        "type": "object",
        "properties": {
            "capital": {"type": "string", "description": "The capital city"},
            "population_millions": {
                "type": "number",
                "description": "Approximate population in millions",
            },
            "language": {"type": "string", "description": "Official language"},
        },
        "required": ["capital", "population_millions", "language"],
    }

    session_id = await client.create_fresh_session(title="test-structured")
    model = MODELS["chunk_summarization"]  # Sonnet — cheaper

    result = await client.send_prompt(
        session_id=session_id,
        model=model,
        user_prompt="What is the capital of Colombia, its approximate population in millions, and official language?",
        json_schema=schema,
    )

    logger.info("Raw result keys: %s", list(result.keys()))
    logger.info("Text: %r", result.get("text", "")[:200])
    logger.info(
        "Structured output: %s",
        json.dumps(result.get("structured_output"), indent=2, ensure_ascii=False),
    )
    logger.info("Usage: %s", json.dumps(result.get("usage"), indent=2))
    logger.info("Error: %s", result.get("error"))

    # Validate
    so = result.get("structured_output")
    if so is None:
        logger.error("FAIL: No structured_output in response")
        return False

    if not isinstance(so, dict):
        logger.error("FAIL: structured_output is not a dict: %s", type(so))
        return False

    for key in ["capital", "population_millions", "language"]:
        if key not in so:
            logger.error("FAIL: Missing key '%s' in structured_output", key)
            return False

    logger.info("PASS: Structured output has all expected keys")
    logger.info(
        "  capital=%s, population=%.1f M, language=%s",
        so["capital"],
        so["population_millions"],
        so["language"],
    )

    # Validate usage
    usage = result.get("usage", {})
    if usage.get("input_tokens", 0) > 0:
        logger.info(
            "PASS: Token usage tracked (input=%d, output=%d)",
            usage["input_tokens"],
            usage["output_tokens"],
        )
    else:
        logger.warning("WARN: No token usage data")

    return True


# ------------------------------------------------------------------
# Test 2: ChunkSummarizerAgent on a real chunk
# ------------------------------------------------------------------


async def test_chunk_summarizer(client: OpenCodeClient) -> bool:
    """Test the ChunkSummarizerAgent end-to-end with a real chunk."""
    logger.info("=" * 60)
    logger.info("TEST 2: ChunkSummarizerAgent end-to-end")
    logger.info("=" * 60)

    from src.agents.chunk_summarizer import ChunkSummarizerAgent
    from src.models import Chunk

    # Create a realistic test chunk from POT content
    test_chunk = Chunk(
        chunk_id="test-chunk-001",
        document_id="test-doc-001",
        source_file="test_document.docx",
        content="""
## Componente General - Visión Territorial

El municipio de Uribia se localiza en el extremo norte de la península de La Guajira,
con una extensión territorial de 8.200 km², siendo el municipio más grande de Colombia
por superficie. Limita al norte con el Mar Caribe, al oriente con Venezuela, al sur con
el municipio de Maicao y al occidente con el municipio de Manaure.

La población total del municipio según el censo DANE 2018 es de 117.674 habitantes,
de los cuales el 93.2% pertenece al pueblo indígena Wayúu. La densidad poblacional
es de 14.3 habitantes por km².

### Distribución Poblacional
- Área urbana: 18.432 habitantes (15.7%)
- Área rural: 99.242 habitantes (84.3%)
- 32 corregimientos y más de 3.000 rancherías dispersas

### Indicadores Socioeconómicos
El índice de Necesidades Básicas Insatisfechas (NBI) alcanza el 96.1% en el área rural
y el 68.4% en el área urbana. La tasa de analfabetismo es del 72.8% en la población
mayor de 15 años. El PIB per cápita del municipio es de $3.2 millones de pesos anuales,
significativamente por debajo del promedio nacional de $22.8 millones.

El Plan de Ordenamiento Territorial establece como objetivo central la articulación
entre el modelo de desarrollo occidental y el sistema normativo Wayúu, fundamentado
en los principios de autonomía territorial reconocidos en la Constitución Política de 1991
y el Decreto 1953 de 2014.
""",
        token_count=350,
        heading_path=["Componente General", "Visión Territorial"],
        section_title="Componente General - Visión Territorial",
        sequence_index=0,
        contains_tables=False,
        contains_images=False,
    )

    agent = ChunkSummarizerAgent(client)
    logger.info("Calling agent.summarize_chunk()...")

    try:
        summary = await agent.summarize_chunk(test_chunk)
    except Exception as exc:
        logger.error("FAIL: Agent raised exception: %s", exc, exc_info=True)
        return False

    # Validate the ChunkSummary object
    logger.info("Summary received!")
    logger.info("  chunk_id: %s", summary.chunk_id)
    logger.info("  confidence: %.2f", summary.confidence)
    logger.info("  summary length: %d chars", len(summary.summary))
    logger.info("  key_facts: %d", len(summary.key_facts))
    logger.info("  numeric_table: %d entries", len(summary.numeric_table))
    logger.info("  uncertainties: %d", len(summary.uncertainties))
    logger.info(
        "  provenance agent: %s",
        summary.provenance.agent if summary.provenance else "none",
    )

    # Print details
    logger.info("\n--- Summary text (first 500 chars) ---")
    logger.info(summary.summary[:500])

    if summary.key_facts:
        logger.info("\n--- Key facts ---")
        for kf in summary.key_facts[:5]:
            logger.info("  [%s] %s", kf.category, kf.fact[:100])

    if summary.numeric_table:
        logger.info("\n--- Numeric table ---")
        for ne in summary.numeric_table[:5]:
            logger.info(
                "  %s = %s %s (%s)", ne.label, ne.value, ne.unit, ne.context[:50]
            )

    # Validation checks
    passed = True

    if not summary.summary:
        logger.error("FAIL: Empty summary")
        passed = False

    if summary.confidence <= 0:
        logger.error("FAIL: Confidence is 0 or negative")
        passed = False

    if len(summary.key_facts) == 0:
        logger.warning("WARN: No key facts extracted (expected some)")

    if len(summary.numeric_table) == 0:
        logger.warning(
            "WARN: No numeric entries (expected several: population, area, NBI, etc.)"
        )

    if summary.provenance is None:
        logger.error("FAIL: No provenance record")
        passed = False
    elif summary.provenance.agent != "chunk_summarization":
        logger.error("FAIL: Provenance agent mismatch: %s", summary.provenance.agent)
        passed = False

    # Token usage
    logger.info(
        "\nAgent token usage: input=%d, output=%d, total=%d",
        agent._total_input_tokens,
        agent._total_output_tokens,
        agent.total_tokens,
    )

    if passed:
        logger.info("PASS: ChunkSummarizerAgent produced valid output")
    else:
        logger.error("FAIL: ChunkSummarizerAgent output had issues")

    return passed


# ------------------------------------------------------------------
# Test 3: Image/file part format
# ------------------------------------------------------------------


async def test_image_input(client: OpenCodeClient) -> bool:
    """Test sending an image as a file part to a multimodal model."""
    logger.info("=" * 60)
    logger.info("TEST 3: Image/file part format")
    logger.info("=" * 60)

    import base64

    # Create a tiny 1x1 red PNG image (smallest valid PNG)
    # PNG header + IHDR + IDAT + IEND for a 1x1 red pixel
    import struct
    import zlib

    def make_tiny_png() -> bytes:
        """Create a minimal 1x1 red pixel PNG."""
        # IHDR
        width = 1
        height = 1
        bit_depth = 8
        color_type = 2  # RGB
        ihdr_data = struct.pack(
            ">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0
        )
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr_chunk = (
            struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        )

        # IDAT: one row, filter byte 0, then R=255, G=0, B=0
        raw = b"\x00\xff\x00\x00"  # filter=None, R, G, B
        compressed = zlib.compress(raw)
        idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
        idat_chunk = (
            struct.pack(">I", len(compressed))
            + b"IDAT"
            + compressed
            + struct.pack(">I", idat_crc)
        )

        # IEND
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

        return b"\x89PNG\r\n\x1a\n" + ihdr_chunk + idat_chunk + iend_chunk

    png_bytes = make_tiny_png()
    image_b64 = base64.b64encode(png_bytes).decode("ascii")

    # Use Gemini (supports images) or Sonnet (also supports images)
    model = MODELS["chunk_summarization"]  # Sonnet supports images
    session_id = await client.create_fresh_session(title="test-image")

    result = await client.send_prompt(
        session_id=session_id,
        model=model,
        user_prompt="Describe the image. What color is it? Reply in one sentence.",
        image_base64=image_b64,
        image_media_type="image/png",
    )

    logger.info("Result keys: %s", list(result.keys()))
    logger.info("Text: %r", result.get("text", "")[:300])
    logger.info("Usage: %s", json.dumps(result.get("usage"), indent=2))
    logger.info("Error: %s", result.get("error"))

    text = result.get("text", "")
    if result.get("error"):
        logger.error("FAIL: Got error: %s", result["error"])
        return False

    if not text:
        logger.error("FAIL: Empty response text")
        return False

    logger.info("PASS: Image input accepted and response received")
    # Check if "red" is mentioned (it's a red pixel)
    if "red" in text.lower() or "roj" in text.lower():
        logger.info("BONUS: Model correctly identified the red color!")

    return True


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


async def main() -> None:
    client = OpenCodeClient()
    await client.start()

    results: dict[str, bool] = {}

    try:
        # Test 1: Structured output
        results["structured_output"] = await test_structured_output(client)

        # Test 2: Chunk summarizer agent
        results["chunk_summarizer"] = await test_chunk_summarizer(client)

        # Test 3: Image input
        results["image_input"] = await test_image_input(client)

    except Exception as exc:
        logger.error("Unhandled error: %s", exc, exc_info=True)
    finally:
        await client.stop()

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %s: %s", name, status)

    total_pass = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info("\n%d / %d tests passed", total_pass, total)

    if total_pass < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
