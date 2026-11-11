# IIT Gandhinagar RAG — Pipeline Design

Ingestion, retrieval, and generation pipelines for a RAG system over IIT Gandhinagar data (scraped HTML pages, PDFs such as circulars, fee structures, academic calendars and handbooks, and other files).

---

## 0. Suggested stack

| Layer | Default pick | Alternative |
|---|---|---|
| Parsing | `trafilatura` (HTML), `PyMuPDF` + `pdfplumber` (PDFs) | `unstructured`, `docling` |
| Embeddings | `BAAI/bge-m3` or `bge-base-en-v1.5` | Gemini `text-embedding-004` |
| Vector DB | Qdrant (local Docker; hybrid search + metadata filters) | Chroma, FAISS |
| Keyword search | BM25 (`rank_bm25`, or Qdrant sparse vectors) | Elasticsearch |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) | Cohere Rerank |
| LLM | Gemini 2.x Flash | Llama 3.x / Qwen via Ollama |
| Orchestration | Plain Python + FastAPI | LangChain / LlamaIndex |

Start with plain Python so every step can be explained; add a framework later if useful.

---

## 1. Data ingestion pipeline (offline, re-runnable)

**Step 1: Build an inventory.**
Walk the data folder and create a manifest (CSV or SQLite) with one row per file: `doc_id`, `path`, `source_url`, `file_type`, `sha256`, `last_modified`, `crawl_date`. The hash makes re-ingestion incremental.

**Step 2: Parse by file type.**

- **HTML:** `trafilatura` to extract main content and drop navbars, footers and sidebars (repeated menus pollute retrieval).
- **Text PDFs:** PyMuPDF for text; `pdfplumber` for tables, converted to Markdown tables (fee structures and calendars are mostly tables).
- **Scanned PDFs:** if a page yields almost no text, run OCR (Tesseract or `docling`).
- **DOCX/XLSX:** `python-docx` and `pandas`; store spreadsheets as Markdown tables or row-wise sentences.

Output one normalized JSON per document: `{doc_id, title, text_markdown, source_url, file_type}`.

**Step 3: Clean and normalize.**
Fix Unicode (`ftfy`), collapse whitespace, strip repeated PDF headers/footers (lines appearing on more than 50% of pages), and drop mostly-empty pages.

**Step 4: Deduplicate.**
Exact duplicates by hash; near-duplicates with MinHash (`datasketch`, Jaccard about 0.9). The same notice often appears on several department pages.

**Step 5: Enrich with metadata** (enables query-time filtering).

- `category`: academics / admissions / fees / hostel / faculty / research / events / placements (from URL path plus keyword rules or a cheap LLM call)
- `department`: CSE, EE, ME, ... (from URL or text)
- `doc_date`: from PDF metadata, URL, or regex (e.g. "dated 12/08/2025")
- `audience`: UG / PG / PhD / all

**Step 6: Chunk.**

- Split on structure first (Markdown headings, PDF sections), then by size: about 400–600 tokens with 15% overlap.
- Never split a table: keep it as one chunk, or split by rows and repeat the header row in each piece.
- Add **contextual headers**: prepend `[Title] > [Section] | Source: [URL]` to each chunk before embedding.
- Store `chunk_id`, `doc_id`, `chunk_index`, `text`, plus all document metadata.

**Step 7: Embed.**
Batch-embed chunks (batch size 32–64 on GPU), L2-normalize. For BGE models, add the query instruction prefix to queries only, per the model card.

**Step 8: Index.**

- Dense vectors into a Qdrant collection with metadata as payload; create payload indexes on `category`, `department`, `doc_date`.
- Sparse/BM25 index (or Qdrant sparse vectors for single-call hybrid search).
- Raw chunks in SQLite or Parquet as the source of truth.

---

## 2. Retrieval pipeline (online, per query)

**Step 1: Query understanding.**

- Rewrite follow-ups into standalone questions using chat history ("What about for PhD?" becomes "What is the hostel fee for PhD students at IITGN?").
- Extract filters with a small LLM call or rules ("CSE M.Tech fee 2025" gives `department=CSE, audience=PG, category=fees`).
- Optional: decompose multi-part questions into sub-queries; HyDE for vague queries.

**Step 2: Hybrid search.**

- Dense: top 30 by cosine similarity, with metadata filters.
- Sparse (BM25): top 30. Catches exact terms such as course codes ("CS 612"), professor names and acronyms.
- Merge with **Reciprocal Rank Fusion**: `score = sum(1 / (60 + rank))`.

**Step 3: Rerank.**
Run the cross-encoder over the ~40 fused candidates and keep the top 5–8. Usually the largest single quality gain.

**Step 4: Post-process.**

- Recency boost for time-sensitive categories (fees, deadlines, admissions) when chunks conflict.
- Diversity: at most 2–3 chunks per document.
- Optional parent expansion: replace small chunks with neighbouring chunks or the full section.

**Step 5: Confidence gate.**
If the top reranker score is below a threshold (tuned on the eval set), mark the query low-confidence so the generator says it doesn't know instead of guessing.

```python
def retrieve(query, history):
    q = rewrite(query, history)
    filters = extract_filters(q)
    dense  = qdrant.search(embed_query(q), filter=filters, limit=30)
    sparse = bm25.search(q, filter=filters, limit=30)
    fused  = rrf([dense, sparse])[:40]
    ranked = reranker.rank(q, fused)[:8]
    ranked = apply_recency_and_diversity(ranked)
    return q, ranked, ranked[0].score if ranked else 0
```

---

## 3. Generation pipeline

**Step 1: Context assembly.**
Number each chunk with title, URL and date:

```text
[1] (Fee Structure 2025-26 | https://iitgn.ac.in/... | 2025-07-01)
<chunk text>
```

Put the strongest chunks first and last (models attend less to the middle of long contexts). Keep total context around 3–6k tokens.

**Step 2: System prompt rules.**

- Answer **only** from the provided context; if the answer is missing, say so and point to the most relevant official page.
- Cite sources inline as [1], [2].
- If sources conflict, prefer the most recent and mention the conflict.
- Copy numbers, dates and fees exactly as in the source.
- Be concise; use tables for structured data.

**Step 3: Generate.**
Low temperature (0–0.3); stream responses via FastAPI (SSE).

**Step 4: Output verification.**

- Check every [n] citation refers to a retrieved chunk.
- Optional groundedness check: a second cheap LLM call verifies each claim against its cited chunk; drop or flag unsupported claims.
- Append a "Sources" list with clickable URLs.

**Step 5: Conversation memory.**
Keep the last N turns for query rewriting; summarize older turns. Never reuse old retrieved chunks; retrieve fresh every turn.

**Step 6: Logging.**
Log query, rewritten query, retrieved chunk IDs and scores, answer, per-stage latency, and user feedback. These logs become evaluation and debugging data.

---

## 4. Evaluation loop (build early)

1. **Golden set:** 50–100 real questions with expected answers and source documents across all categories, including tables, dates, acronyms, follow-ups, and questions with no answer in the data.
2. **Retrieval metrics:** Recall@k, MRR, nDCG (is the correct chunk in the top 8?).
3. **Generation metrics:** faithfulness, answer relevance, context precision (RAGAS or an LLM judge).
4. **Ablations:** dense-only vs hybrid vs hybrid + rerank; chunk sizes 300 / 500 / 800 tokens.

---

## 5. Build order

1. End-to-end slice: HTML + text-PDF ingestion, dense retrieval, generation.
2. Golden evaluation set.
3. Hybrid search and reranking, measuring the gain at each step.
4. Table handling, metadata filters, recency.
5. Query rewriting and conversation memory.
6. FastAPI backend, simple UI (Streamlit or React), Docker, incremental re-ingestion cron job.
