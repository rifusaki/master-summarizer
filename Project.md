I have 300MB worth of word documents filled with information (POT Uribia, Guajira) and I need to summarize them following specific communication guidelines (target reader, etc) in the style of a couple of executive summary examples (20MB PDFs) I have. this data includes text, tables, graphs and maps. This means the amount of information is huge, there are style constraints and summarization decisions must be done based on rules inferred by the example documents. The end result should be like the example execute summaries, roughly enough for a 80 to 100 slide PowerPoint to be done at a later stage.

## Stages, agents and models
### Summary
I can afford Opus for the heavy lifting, but we have to plan it carefully to avoid iterating too much. We have the following stages/agents then:
- Orchestration: Overseeing and coordinating the whole process. Model must be able to efficiently coordinate subagents and ensure information integrity in their communication.
- Document pre-processing and normalization: multimodal extraction of text, tables, captions and images/graphs/maps conversion into detailed descriptions from DOCX and PDF files. No summarization is done at this point. This should have a series of outputs for the next stage. Model should be able to process complex data types.
- Chunk summarization/compaction: distilling prose and making high-fidelity summary of thematic, lenght-limited chunks. Information should be preserved as much as possible. Chunks should be clearly organized and outlined for central summarization. Model should have generous context window and stick to source.
- Style learning: Ingestion of processed example documents to infer rules alongside specific, personalizaed style instructions to create a set of rules.
- Central summarization: Ingestion of source document chunk summaries alongside style instructions. This will produce the working master draft. As source is still larger than its context window, the previously done chunk outline is crucial.
- Secondary reviewer: Iterative refination of tone, cross-checking against sources, consistency enforcement, and output stres-testing against established rules. This agent should transmit its feedback to Central Summarization. Model TBD.
- Slide outline generation: After the working draft is done, convert it into slide outlines.

### Models

| **Role**                                           | **Top model choice**                  | **Why**                                                                                                   | **Fallback / secondary**                                               |
| -------------------------------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Orchestration (controller)**                     | **GPT‑5.3**                           | Fast, excellent structured output, good at deterministic coordination, low-latency for many control loops | Claude Sonnet 4.6 (if you prefer Anthropic safety/consistency)         |
| **Preprocessing / multimodal extraction**          | **Gemini 3.1 Pro**                    | Strong multimodal parsing (tables, charts, maps → rich textual descriptions)                              | Vision-specialized pipeline + OCR (Tesseract/Adobe SDK) for edge cases |
| **Chunk summarization / compaction**               | **Claude Sonnet 4.6**                 | Cost-effective, reliable at faithful chunk-level compression, good at sticking to source                  | Sonnet for most chunks; reserve Opus for the hardest chunks            |
| **Style learning**                                 | **Claude Opus 4.6**                   | Best at inferring and codifying style rules from examples; long-context pattern extraction                | —                                                                      |
| **Central summarization (master draft)**           | **Claude Opus 4.6**                   | Highest fidelity style transfer and long-form synthesis                                                   | —                                                                      |
| **Secondary reviewer (consistency, cross-checks)** | **GPT‑5.3-Codex or regular GPT-5.2?** | Fast, deterministic checks, rule enforcement, provenance tracing, automated cross-referencing             | Opus 4.6 for final high-cost QA pass on critical sections              |
| **Slide outline generation**                       | **GPT‑5.3**                           | Produces structured, slide-ready outlines and bullet hierarchies efficiently                              | Gemini 3.1 Pro for multimodal slide notes (figure captions)            |

### Architecture (agents and data flow)
1. **Ingest layer (storage + indexing)**
    - Store raw files (DOCX, PDFs, images) in object storage.
    - Run a **document parser** that extracts: plain text, structural metadata (headings, tables, captions), images, and geospatial/map assets.
    - Save parsed outputs and file-level metadata to a document store and to a **vector DB** (embeddings per chunk).

2. **Preprocessing agent (Gemini 3.1 Pro + deterministic tools)**    
    - Tasks: OCR, table-to-CSV, chart-to-data (where possible), map-to-description (legend, scale, annotations).
    - Output: **normalized artifacts** — text blocks, table CSVs, chart-data JSON, image captions, and a confidence score per artifact.
    - No summarization here; only canonicalized, machine-friendly representations.
        
3. **Chunking + embedding**
    - Chunk rules: prefer semantic boundaries (headings, paragraphs, table blocks), limit chunk size to a token budget (e.g., 8–12k tokens for Sonnet-level summarization).
    - Create embeddings for each chunk and store in vector DB for retrieval.
    - Tag chunks with provenance metadata (source file, page, coordinates, confidence).
        
4. **Chunk summarization agent (Claude Sonnet 4.6)**
    - Input: one chunk + explicit extraction metadata + strict instruction template (preserve facts, list extracted entities, preserve numeric values).
    - Output: **chunk summary** (length-limited), **key facts table**, **uncertainties/assumptions**, and **links to source chunk IDs**.
        
5. **Style learning agent (Opus 4.6)**
    - Input: the 20MB example PDFs (preprocessed into the same canonical format) + explicit communication guidelines in an MD file (yet to be created).
    - Tasks: infer **rule set** (tone, section order, preferred headings, bullet density, allowed abbreviations, numeric formatting, citation style).
    - Output: a **machine-readable style guide** (JSON) and a short human-readable rubric with examples and “do / don’t” rules.
        
6. **Central summarization agent (Opus 4.6)**
    - Input: style guide + prioritized list of chunk summaries (retrieved by relevance) + global outline template.
    - Tasks: synthesize a **master draft** organized to the target reader and slide-count goal. Produce hierarchical sections and a slide-outline-ready narrative.
    - Output: master draft + mapping from each paragraph/claim → source chunk IDs + confidence.
        
7. **Secondary reviewer agent (GPT‑5.3)**
    - Input: master draft + style guide + chunk summaries + provenance links.
    - Tasks: automated cross-checks (fact consistency, numeric reconciliation, missing citations), tone adjustments, flagging contradictions, and producing a prioritized list of edits.
    - Output: annotated draft with suggested edits and a short “risk & uncertainty” register.
        
8. **Refinement loop**
    - Orchestration agent applies reviewer edits back to Central Summarization (either automatically for low-risk edits or via human-in-the-loop for high-risk ones).
    - Final pass by Opus for polishing.
        
9. **Slide outline generator (GPT‑5.3)**
    - Input: final master draft + style guide + slide constraints (80–100 slides, per-slide word limits, visuals per slide).
    - Output: slide-by-slide outline with **title, 3–5 bullets, suggested visual (table/graph/map), and source references**.

## Orchestration design and agent responsibilities
- **Orchestrator (GPT‑5.3)** responsibilities:
    - Maintain **task queue**, dispatch jobs to agents, enforce token budgets and cost caps, collect outputs, and ensure provenance metadata is attached.
    - Implement **quality gates**: e.g., require chunk summaries to include numeric reconciliation and a confidence score before central ingestion.
    - Retry logic: deterministic retries for low-confidence outputs; escalate to human review for repeated failures.
    - Logging: immutable audit trail linking every claim in the master draft to source chunk IDs and agent outputs.
        
- **Communication format between agents**: JSON with fields: `chunk_id`, `source_file`, `text`, `summary`, `key_facts[]`, `numeric_table[]`, `confidence`, `style_tags[]`. This keeps messages machine-parseable and reduces iteration.

## Prompting, templates, and token budgeting (practical rules)
- **Preprocessing prompts** (Gemini): ask for structured outputs only (CSV/JSON). Include explicit extraction rules (e.g., “extract table headers, units, footnotes, and numeric precision”).
- **Chunk summarizer prompt template** (Sonnet):
    - **System prompt example**: “You are a faithful summarizer. Do not hallucinate. Preserve numeric values and units. Return: 1) Aummary, 2) key facts list, 3) numeric table, 4) provenance IDs.”
    - **Budget**: Require a `confidence` field.
- **Style learning prompt** (Opus): feed multiple examples and ask for a **rule set** in JSON with examples for each rule. Ask Opus to produce a short checklist for reviewers.
- **Central summarization prompt** (Opus): include the style JSON, a prioritized list of chunk summaries (top N per section), and an explicit slide-target constraint. Ask for a hierarchical draft with inline source references.
- **Reviewer prompt** (GPT‑5.3): ask for a **diff-style** annotated output: “Accept / Edit / Reject” per paragraph, with reason and suggested replacement.
- **Token budgeting**:
    - Reserve Opus calls for style learning and final synthesis only.
    - Use Sonnet and GPT‑5.3 for high-volume chunk work and orchestration.
    - Example budget: preprocess + chunking (free deterministic tools), 300–500 Sonnet calls for chunk summaries, 3–6 Opus calls (style learning + central synth passes), multiple GPT‑5.3 passes for orchestration and review.

### Quality control, provenance, and iteration minimization
- **Provenance-first**: every claim in the master draft must link to at least one chunk ID and show the original text excerpt (or numeric table) in the audit log.
- **Confidence thresholds**: only accept automated merges when `confidence >= 0.85`. Lower-confidence items go to human review.
- **Automated reconciliation**: reviewer agent runs numeric checks (sums, units, ranges) and flags mismatches before Opus finalization.
- **Human-in-the-loop gates**: set human review for: legal/land-use claims, geospatial assertions, or any flagged contradictions.
- **Batching strategy**: group similar chunks (same topic/section) and summarize them together to reduce Opus passes. Use Sonnet to pre-compact groups before Opus ingestion. 
- **Change control**: store versions; orchestrator enforces single-source-of-truth for the master draft.