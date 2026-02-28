# Master Summarizer

Multi-agent document summarization pipeline for POT Uribia, La Guajira. Processes ~300MB of Colombian municipal planning documents (DOCX) into a styled executive summary and 80-100 PowerPoint slide outlines, using multiple LLM models coordinated through an OpenCode server.

## Pipeline Stages

```
1. Parse & Preprocess  -  DOCX/PDF parsing + Gemini image descriptions
2. Chunk               -  Deterministic semantic chunking (heading-aware, token-budgeted)
3. Summarize Chunks    -  Per-chunk faithful summaries (Claude Sonnet)
4. Learn Style         -  Infer style guide from example PDFs + manual guidelines (Claude Opus)
5. Central Synthesis   -  Master draft from all summaries following style guide (Claude Opus)
6. Review              -  Systematic verification + refinement loop (GPT-5.3 Codex)
7. Slide Generation    -  Structured slide outlines from final draft (GPT-5.3 Codex)
```

Stages 1-2 are deterministic. Stages 1b and 3-7 use LLM agents. The pipeline pauses between stages for manual confirmation and saves state for resumability.

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- [OpenCode](https://opencode.ai/) CLI installed and configured with access to:
  - `google/gemini-3.1-pro-preview`
  - `github-copilot/claude-sonnet-4.6`
  - `github-copilot/claude-opus-4.6`
  - `azure-gpt/gpt-5.3-codex`

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
| `MAX_RETRIES` | `3` | Max retries for low-confidence outputs |
| `CHUNK_TOKEN_BUDGET` | `10000` | Max tokens per chunk |

### Input Files

Place input files before running:

```
input/
  raw_data/           # DOCX source documents (POT Uribia)
  style_examples/     # PDF style reference documents
```

## Usage

```bash
# Start the OpenCode server (in a separate terminal, or the pipeline auto-starts it)
opencode serve --port 4096

# Run the pipeline
uv run summarizer
```

The pipeline runs stage-by-stage, pausing at checkpoints for confirmation. Type `c` to continue or `q` to quit and save state. On restart, it resumes from the last completed stage.

## Output

```
output/
  preprocessed/       # Parsed document JSONs
  chunks/             # Semantic chunk JSONs
  chunk_summaries/    # Per-chunk summary JSONs
  style_guide/        # Inferred style guide JSON
  drafts/             # Master draft versions (Markdown)
  reviews/            # Review result JSONs
  slides/             # Slide outline JSONs
  pipeline_state.json # Resumable state

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
    reviewer.py              # Quality review (GPT-5.3)
    slide_generator.py       # Slide outline generation (GPT-5.3)
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

See `AGENTS.md` for detailed agent specifications.
