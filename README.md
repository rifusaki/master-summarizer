# Master Summarizer

Multi-agent document summarization pipeline for POT Uribia, La Guajira. Processes ~300MB of Colombian municipal planning documents (DOCX) into a styled executive summary and 80-100 PowerPoint slide outlines, using multiple LLM models coordinated through an OpenCode server.

## Pipeline Stages

```
1. Parse & Preprocess  -  DOCX/PDF parsing + Gemini image descriptions
2. Chunk               -  Deterministic semantic chunking (heading-aware, token-budgeted)
3. Summarize Chunks    -  Per-chunk faithful summaries (Claude Sonnet)
4. Learn Style         -  Infer style guide from example PDFs + manual guidelines (Claude Opus)
5. Central Synthesis   -  Master draft from all summaries following style guide (Claude Opus)
6. Review              -  Systematic verification + refinement loop (GPT-5.2)
7. Slide Generation    -  Structured slide outlines from final draft (GPT-5.2)
```

Stages 1-2 are deterministic. Stages 1b and 3-7 use LLM agents. The pipeline pauses between stages for manual confirmation and saves state for resumability.

All LLM stages are resilient: each item (image, chunk, draft section, slide section) is saved atomically to disk immediately after completion. On restart, already-completed items are skipped automatically — no tokens are wasted.

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- [OpenCode](https://opencode.ai/) CLI installed and configured with access to:

| Stage | Primary | Fallback |
|-------|---------|---------|
| Image preprocessing | `google/gemini-3.1-pro-preview` | `google/gemini-3-pro-preview` |
| Chunk summarization | `azure-anthropic/claude-sonnet-4-6` | `azure-anthropic/claude-sonnet-4-5` |
| Style learning | `github-copilot/claude-opus-4.6` | — |
| Central synthesis | `github-copilot/claude-opus-4.6` | — |
| Review | `azure-gpt/gpt-5.2` | `github-copilot/gpt-5.2` |
| Slide generation | `azure-gpt/gpt-5.2` | `github-copilot/gpt-5.2` |

## Setup

```bash
# Install dependencies
uv sync

# Copy and edit environment config
cp .env.example .env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCODE_SERVER_HOST` | `127.0.0.1` | OpenCode server host |
| `OPENCODE_SERVER_PORT` | `4096` | OpenCode server port |
| `CONFIDENCE_THRESHOLD` | `0.85` | Min confidence for automated acceptance |
| `MAX_RETRIES` | `3` | Max retries per model before fallback |
| `CHUNK_TOKEN_BUDGET` | `10000` | Max tokens per chunk |
| `PREPROCESSING_CONFIRM_FALLBACK` | `1` | Set to `0` to auto-switch Gemini fallback without prompt (recommended for unattended runs) |
| `PREPROCESSING_RATE_LIMIT_STREAK_THRESHOLD` | `3` | Consecutive 429s before triggering fallback model switch |

### Input Files

Place input files before running:

```
input/
  raw_data/           # DOCX source documents (POT Uribia)
  style_examples/     # PDF style reference documents
```

## Usage

```bash
# Run the pipeline (auto-starts OpenCode server)
uv run summarizer
```

The pipeline runs stage-by-stage, pausing at checkpoints for confirmation. Type `c` to continue or `q` to quit and save state. On restart, it resumes automatically — already-completed items are skipped.

### CLI Flags

```bash
# Override the preprocessing model for this run only (e.g. after quota exhaustion)
uv run summarizer --preprocess-model google/gemini-3-pro-preview

# Retry only failed image descriptions
uv run summarizer --retry-failed-images

# Retry failed images with a specific model (most common recovery workflow)
uv run summarizer --retry-failed-images --preprocess-model google/gemini-3-pro-preview
uv run summarizer --retry-failed-images --preprocess-model openrouter/google/gemini-3-pro-preview

# Retry only failed/low-confidence chunk summaries
uv run summarizer --retry-failed-chunks
```

**Flag compatibility notes:**

| Flag combination | Works? | Notes |
|-----------------|--------|-------|
| `--retry-failed-images --preprocess-model X` | ✅ | Correct recovery workflow |
| `--retry-failed-images --all-runs` | ✅ | `--all-runs` is accepted but currently a no-op — retry already covers all runs by default |
| `--retry-failed-images --all-runs --preprocess-model X` | ✅ | Same as above; `--preprocess-model` applies correctly |
| `--retry-failed-chunks --preprocess-model X` | ⚠️ | `--preprocess-model` is silently ignored — it only controls Gemini preprocessing, not chunk summarization |
| `--retry-failed-chunks --all-runs` | ⚠️ | `--all-runs` is silently ignored — it only applies to `--retry-failed-images` |

## Output

```
output/
  preprocessed/       # Parsed document JSONs (images described in-place)
  chunks/             # Semantic chunk JSONs
  chunk_summaries/    # Per-chunk summary JSONs
  style_guide/        # Inferred style guide JSON
  drafts/
    sections/         # Incremental draft section saves (cleared after full draft written)
    draft_v*.json     # Full master draft versions
  reviews/            # Review result JSONs
  slides/
    sections/         # Incremental slide section saves (cleared after full set written)
    *.json            # Full slide outline sets
  pipeline_state.json # Resumable checkpoint state
  audit_log.jsonl     # Append-only audit trail

review/               # Human review files (flagged items, quality/provenance reports)

data/
  chroma/             # ChromaDB vector store (chunk embeddings)
```

## Project Structure

```
src/
  main.py                    # Pipeline orchestrator
  config.py                  # Configuration, paths, model mappings
  models.py                  # Pydantic data models
  opencode_client.py         # HTTP client for OpenCode server
  agents/
    base.py                  # Base agent class (LLM communication)
    preprocessor.py          # Image/visual description (Gemini)
    chunker.py               # Deterministic semantic chunker
    chunk_summarizer.py      # Chunk-level summarization (Sonnet)
    style_learner.py         # Style guide inference (Opus)
    central_summarizer.py    # Master draft synthesis (Opus)
    reviewer.py              # Quality review (GPT-5.2)
    slide_generator.py       # Slide outline generation (GPT-5.2)
  parsers/
    docx_parser.py           # DOCX document parser
    pdf_parser.py            # PDF document parser (PyMuPDF)
  storage/
    document_store.py        # JSON file-based storage
    vector_store.py          # ChromaDB vector store
  utils/
    provenance.py            # Provenance tracking and validation
    quality.py               # Quality gates (confidence, numerics, review)
    token_budget.py          # Token counting and cost tracking

prompts/                     # Agent system prompts (Markdown)
  preprocessing.md
  chunk_summarization.md
  style_learning.md
  central_summarization.md
  reviewer.md
  slide_generation.md
  manual_style_guide.md      # User-written communication guidelines (highest priority)
```

See `AGENTS.md` for detailed agent specifications including resilience design per stage.
