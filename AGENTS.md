# AGENTS.md — Agent Reference

Quick reference for the multi-agent pipeline. Use this to understand each agent's role, model, I/O, and location in the codebase.

---

## Pipeline Overview

```
Parse → Image Preprocess → Chunk → Summarize → Style Learn → Synthesize → Review → Slides
 (1)        (1b)            (2)       (3)          (4)          (5)         (6)      (7)
```

Stages 1 and 2 are deterministic (no LLM). Stages 1b and 3-7 use LLM agents via OpenCode server.

---

## Agent Table

| # | Agent Class | File | Role (prompt key) | Primary Model | Fallback Model | Output Type |
|---|-------------|------|--------------------|--------------|---------------|-------------|
| 1b | `PreprocessorAgent` | `src/agents/preprocessor.py` | `preprocessing` | `google/gemini-3.1-pro-preview` | `google/gemini-3-pro-preview` | Structured JSON |
| 2 | `Chunker` | `src/agents/chunker.py` | *(deterministic)* | None | None | `Chunk` objects |
| 3 | `ChunkSummarizerAgent` | `src/agents/chunk_summarizer.py` | `chunk_summarization` | `azure-anthropic/claude-sonnet-4-6` | `azure-anthropic/claude-sonnet-4-5` | Structured JSON |
| 4 | `StyleLearnerAgent` | `src/agents/style_learner.py` | `style_learning` | `github-copilot/claude-opus-4.6` | *(none)* | Structured JSON |
| 5 | `CentralSummarizerAgent` | `src/agents/central_summarizer.py` | `central_summarization` | `github-copilot/claude-opus-4.6` | *(none)* | Free-form Markdown |
| 6 | `ReviewerAgent` | `src/agents/reviewer.py` | `reviewer` | `azure-gpt/gpt-5.2` | `github-copilot/gpt-5.2` | Structured JSON |
| 7 | `SlideGeneratorAgent` | `src/agents/slide_generator.py` | `slide_generation` | `azure-gpt/gpt-5.2` | `github-copilot/gpt-5.2` | Structured JSON |

---

## Agent Details

### 1b. PreprocessorAgent

- **Purpose**: Describes images, charts, maps, and table-images extracted from documents.
- **Input**: Base64-encoded image + artifact metadata.
- **Output**: `ImageDescription` — structured text description with content type, extracted data, geographic info, and confidence.
- **Prompt**: `prompts/preprocessing.md`
- **Key behavior**: Produces descriptions in Spanish. Preserves all numeric data from visual content. Flags low-confidence or illegible elements.
- **Resilience**: Fresh OpenCode session per image (prevents context-bloat timeouts). Per-image status stamped in artifact metadata (`preprocess_status`, `preprocess_model`, `preprocess_attempts`). Timeout retries with fresh session. Rate-limit streak detection triggers automatic fallback model switch (`PREPROCESSING_CONFIRM_FALLBACK=0` to suppress prompt). `set_active_model()` allows runtime model override via `--preprocess-model` CLI flag. Raises `ModelExhaustionError` when all models are exhausted, after saving all progress.

### 2. Chunker (deterministic)

- **Purpose**: Splits parsed document artifacts into semantic chunks for summarization.
- **Input**: `DocumentParseResult` with `NormalizedArtifact` list.
- **Output**: `Chunk` objects stored in `output/chunks/` and ChromaDB.
- **No prompt** — purely algorithmic.
- **Key parameters**: `chunk_token_budget` (default 10,000), `chunk_overlap_tokens` (default 500).
- **Strategy**: Groups by heading L1-L2, accumulates until token budget, splits at paragraph boundaries, adds overlap.

### 3. ChunkSummarizerAgent

- **Purpose**: Produces faithful summaries of individual chunks, preserving all facts and numbers.
- **Input**: Chunk text + section context (heading path).
- **Output**: `ChunkSummary` — summary text, key facts, numeric table, uncertainties, confidence.
- **Prompt**: `prompts/chunk_summarization.md`
- **Key behavior**: All output in Spanish. 30-50% compression. Never drops table rows. Flags uncertainty. Confidence < 0.85 triggers retry.
- **Resilience**: `call_llm_structured_resilient` with fresh session per chunk. Per-chunk save via `on_chunk_done` callback. Raises `ModelExhaustionError` after saving all completed summaries. Resumable via `--retry-failed-chunks`.

### 4. StyleLearnerAgent

- **Purpose**: Analyzes example executive summaries to infer a machine-readable style guide.
- **Input**: 2 PDF parse results + manual communication guidelines from `prompts/manual_style_guide.md`.
- **Output**: `StyleGuide` — tone, section order, rules by category, reviewer checklist, formatting conventions.
- **Prompt**: `prompts/style_learning.md`
- **Key behavior**: Evidence-based rules with do/don't examples. Manual guidelines stored in `communication_guidelines` field and given highest priority downstream.

### 5. CentralSummarizerAgent

- **Purpose**: Synthesizes all chunk summaries into a coherent master draft following the style guide.
- **Input**: `ChunkSummary` list + `StyleGuide` (including manual communication guidelines).
- **Output**: `MasterDraft` with `DraftSection` list — Markdown prose in Spanish.
- **Prompt**: `prompts/central_summarization.md`
- **Key behavior**: Organizes by theme, not source. Inline provenance markers `[Chunk: <id>]`. Follows manual style guide with highest priority. Includes `[Sugerencia de diseño: ...]` visual suggestions per manual guidelines. Dense enough for 80-100 slides.
- **Resilience**: Each `DraftSection` saved atomically to `output/drafts/sections/` immediately via `on_section_done` callback. On resume, already-completed sections are loaded and skipped (matched by heading). Full `MasterDraft` saved at end; incremental files cleared.

### 6. ReviewerAgent

- **Purpose**: Systematically verifies the master draft against source data, style guide, and factual consistency.
- **Input**: `MasterDraft` + `StyleGuide` + source facts/numerics from chunk summaries.
- **Output**: `ReviewResult` — per-paragraph annotations (accept/edit/reject), risk register, overall confidence.
- **Prompt**: `prompts/reviewer.md`
- **Key behavior**: Reviews in Spanish. Checks fact consistency, numeric reconciliation, tone compliance, completeness, internal consistency. Flags legal/geographic/ethnic claims for human review. Reject triggers re-draft loop (max 3 iterations).

### 7. SlideGeneratorAgent

- **Purpose**: Converts finalized master draft sections into structured PowerPoint slide outlines.
- **Input**: Draft section text + `StyleGuide`.
- **Output**: `SlideOutlineSet` — array of slide outlines with titles, bullets, visual suggestions, speaker notes.
- **Prompt**: `prompts/slide_generation.md`
- **Key behavior**: All content in Spanish. One idea per slide. 3-5 bullets max, 8-12 words each. Suggests visual type per slide. Speaker notes with fuller context. Target: 80-100 slides total.
- **Resilience**: Each section's slides saved atomically to `output/slides/sections/` immediately via `on_section_done` callback. On resume, completed sections injected in draft order with contiguous slide re-numbering, then skipped in generation loop. Full `SlideOutlineSet` saved at end; incremental files cleared.

---

## Base Agent Class

All LLM agents inherit from `BaseAgent` (`src/agents/base.py`), which provides:

- `call_llm(user_prompt, image_parts, fresh_session)` — sends a text prompt (with optional images) to the OpenCode server. `fresh_session=True` creates an isolated session to prevent context accumulation across many sequential calls.
- `call_llm_structured(user_prompt, schema, image_parts)` — sends a prompt with a JSON schema for structured output.
- `call_llm_resilient(...)` — wraps `call_llm` with automatic retry and fallback model chain. Defaults to `fresh_session=True`.
- `call_llm_structured_resilient(...)` — same, for structured JSON output.
- `load_prompt()` — loads system prompt from `prompts/{self.role}.md`.
- `create_provenance(chunk_ids)` — creates a `ProvenanceRecord` with model info and timestamp.
- Token tracking via `_total_input_tokens` and `_total_output_tokens`.

The OpenCode client (`src/opencode_client.py`) handles HTTP communication with the `opencode serve` server, including session management, message sending, structured output parsing, and image file parts.

---

## Prompt Files

| File | Agent | Notes |
|------|-------|-------|
| `prompts/preprocessing.md` | PreprocessorAgent | Image/visual description |
| `prompts/chunk_summarization.md` | ChunkSummarizerAgent | Chunk-level summarization |
| `prompts/style_learning.md` | StyleLearnerAgent | Style guide inference |
| `prompts/central_summarization.md` | CentralSummarizerAgent | Master draft synthesis |
| `prompts/reviewer.md` | ReviewerAgent | Quality review |
| `prompts/slide_generation.md` | SlideGeneratorAgent | Slide outline generation |
| `prompts/manual_style_guide.md` | *(loaded by orchestrator)* | User-written communication guidelines (Spanish) |

---

## Data Models

All inter-agent communication uses Pydantic models defined in `src/models.py`:

- `NormalizedArtifact` — single parsed element (text, table, image)
- `DocumentParseResult` — complete parse output for one file
- `Chunk` — semantic chunk with token count and provenance
- `ChunkSummary` — summary with key facts, numerics, confidence
- `StyleGuide` — inferred + manual style rules
- `MasterDraft` / `DraftSection` — synthesized executive summary
- `ReviewResult` / `ReviewAnnotation` — review verdicts
- `SlideOutlineSet` / `SlideOutline` — slide outlines

## Storage

- File-based: `DocumentStore` (`src/storage/document_store.py`) — JSON files in `output/` subdirectories.
- Vector: `VectorStore` (`src/storage/vector_store.py`) — ChromaDB in `data/chroma/` for chunk embeddings.
- Archiver: `src/storage/archiver.py` — `export_run()` / `import_run()` for portable tar.gz snapshots of completed or partial runs. Invoked via `--export-run` and `--import-run` CLI flags.

## Quality & Provenance

- `src/utils/quality.py` — quality gates (confidence checks, numeric reconciliation, review quality).
- `src/utils/provenance.py` — provenance tracking and validation.
- `src/utils/token_budget.py` — token counting and cost tracking per agent.
