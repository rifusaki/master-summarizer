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

| # | Agent Class | File | Role (prompt key) | Model | Output Type |
|---|-------------|------|--------------------|-------|-------------|
| 1b | `PreprocessorAgent` | `src/agents/preprocessor.py` | `preprocessing` | `google/gemini-3.1-pro-preview` | Structured JSON |
| 2 | `Chunker` | `src/agents/chunker.py` | *(deterministic)* | None | `Chunk` objects |
| 3 | `ChunkSummarizerAgent` | `src/agents/chunk_summarizer.py` | `chunk_summarization` | `github-copilot/claude-sonnet-4.6` | Structured JSON |
| 4 | `StyleLearnerAgent` | `src/agents/style_learner.py` | `style_learning` | `github-copilot/claude-opus-4.6` | Structured JSON |
| 5 | `CentralSummarizerAgent` | `src/agents/central_summarizer.py` | `central_summarization` | `github-copilot/claude-opus-4.6` | Free-form Markdown |
| 6 | `ReviewerAgent` | `src/agents/reviewer.py` | `reviewer` | `azure-gpt/gpt-5.3-codex` | Structured JSON |
| 7 | `SlideGeneratorAgent` | `src/agents/slide_generator.py` | `slide_generation` | `azure-gpt/gpt-5.3-codex` | Structured JSON |

---

## Agent Details

### 1b. PreprocessorAgent

- **Purpose**: Describes images, charts, maps, and table-images extracted from documents.
- **Input**: Base64-encoded image + artifact metadata.
- **Output**: `ImageDescription` — structured text description with content type, extracted data, geographic info, and confidence.
- **Prompt**: `prompts/preprocessing.md`
- **Key behavior**: Produces descriptions in Spanish. Preserves all numeric data from visual content. Flags low-confidence or illegible elements.

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

---

## Base Agent Class

All LLM agents inherit from `BaseAgent` (`src/agents/base.py`), which provides:

- `call_llm(user_prompt, image_parts)` — sends a text prompt (with optional images) to the OpenCode server.
- `call_llm_structured(user_prompt, schema, image_parts)` — sends a prompt with a JSON schema for structured output.
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

## Quality & Provenance

- `src/utils/quality.py` — quality gates (confidence checks, numeric reconciliation, review quality).
- `src/utils/provenance.py` — provenance tracking and validation.
- `src/utils/token_budget.py` — token counting and cost tracking per agent.
