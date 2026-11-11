# IITGN RAG Project — Chat Summary and Handoff

A summary of the whole chat in which this project was built, written so a new chat can pick up where this one stopped.
Last updated: 2026-11-11 (checked line by line against the current code).

**How to use it in a new chat:** attach or paste this file and say something like
*"This is the summary of my IITGN RAG project so far. Read it, then read the code before changing anything."*
The design document `IITGN_RAG_Pipeline_Design.md` is the plan this project follows; section numbers below refer to it.

---

## 1. Goal and working rules

**Goal:** a Retrieval-Augmented Generation (RAG) system that answers questions about IIT Gandhinagar (IITGN) from scraped IITGN web pages and PDFs, built step by step following `IITGN_RAG_Pipeline_Design.md`.

**Rules the user set (keep following them):**
- Simple Python with clear, short comments. No frameworks (no LangChain or LlamaIndex orchestration).
- Each module exposes a `run_...()` function called from `main.py`. Modules have no `if __name__ == "__main__"` blocks.
- Ask before deleting anything. Deleted files went to the Recycle Bin.
- The Gemini API key lives only in `.env`, which git ignores. Never print or share it.
- **Current status: changes are paused.** The user is reading and learning the code. Don't change code until asked.
- **Deferred, don't implement until asked:** FastAPI, UI, Docker, scheduled re-ingestion, `git init`, and reducing startup time (see section 9).

---

## 2. Environment

| Item | Value |
|---|---|
| OS / hardware | Windows 11; Ryzen 5 5500U, 7.3 GB RAM, CPU only (no GPU) |
| Project folder | `C:\Users\Jaya Prakash\Desktop\IITGN_RAG_Project` |
| Python | 3.10 in `.venv` (`.venv\Scripts\python.exe`) |
| Main libraries | torch 2.14.1+cpu, transformers 5.20.2, sentence-transformers 6.2, qdrant-client 1.20.0, rank_bm25 0.2.2, google-genai 2.33.0, requests. `requirements.txt` pins no versions and does not list `torch` (it comes in through sentence-transformers) |
| Git | The project has **no repository of its own**; it shows up as an untracked folder inside a repository at `C:\Users\Jaya Prakash`. Nothing is committed. The project `.gitignore` excludes `.env`, `.venv/`, `__pycache__/`, `output/` and `qdrant_db/`. |
| Windows output encoding | Characters such as ₹ can break printing when output is redirected. Use `PYTHONIOENCODING=utf-8`; `run_chat()` also switches stdout to UTF-8. |

---

## 3. Files and commands

```
IITGN_RAG_Project/
├── IITGN_RAG_Pipeline_Design.md   the plan (sections 0-5)
├── CHAT_SUMMARY.md                this file
├── .env                           settings + GEMINI_API_KEY (git-ignored)
├── config.py                      reads .env (load_dotenv(override=True))
├── main.py                        entry point, picks the command
├── data_ingest.py                 design steps 1-6: inventory, parse, clean, dedupe, enrich, chunk
├── embed.py                       step 7: chunk vectors
├── vector_store.py                step 8: Qdrant collection
├── retrieve.py                    section 2: hybrid search + rerank + confidence
├── generate.py                    section 3: context, prompt, Gemini answer, citation check, log
├── evaluate.py                    section 4: retrieval metrics (Hit@k, MRR)
├── evaluate_answers.py            section 4.3: answer grading with an LLM judge
├── evaluation/questions.jsonl     50 hand-written test questions (meant to be tracked by git)
├── requirements.txt
├── data/                          scraped .txt files (pages and PDF text), one folder per domain
├── output/                        generated: manifest.csv, parsed/, chunks.jsonl, embeddings/, results, logs
└── qdrant_db/                     local Qdrant database folder
```

| Command | What it does |
|---|---|
| `python main.py` | Build the index: ingestion, then embeddings, then the vector store |
| `python main.py search` | Interactive retrieval only: shows the ranked chunks |
| `python main.py chat` | Interactive Q&A with Gemini answers, citations and a Sources list |
| `python main.py evaluate` | Retrieval evaluation on the 50 questions |
| `python main.py evaluate-answers` | Answer evaluation. Resumable: re-running continues where it stopped. Delete `output/answer_evaluation.json` to start over |

Only one program can open the local `qdrant_db` folder at a time, so don't run two commands together.

`main.py` is the only way to run anything. The docstrings of `data_ingest.py` and `embed.py` still mention `python data_ingest.py` and `python embed.py --limit 200`, but those modules have no `__main__` block, so these commands do nothing. `run_embeddings()` still reads a `--limit` argument, but `python main.py --limit 200` is not a quick test: it falls through to the full build and re-runs all of ingestion first.

---

## 4. What was built, stage by stage

### 4.1 Ingestion: `data_ingest.py` (design steps 1-6), done in earlier sessions
- **Step 1, inventory:** `output/manifest.csv`, one row per file (doc_id, path, URL, type, sha256, status, `duplicate_of`, ...). About 1,600 files.
- **Step 2, parse:** each scraped `.txt` file has a header (URL, title, ...) followed by content. One JSON per document goes to `output/parsed/`.
- **Step 3, clean:** fixes broken text encoding (ftfy), removes invisible characters, strips page headers that repeat at least 4 times, drops near-empty documents.
- **Step 4, deduplicate:** exact matches by hash; near-duplicates with MinHash (`datasketch`, 5-word groups, 90% similarity, only documents of 50+ words). A near-duplicate is only removed if all its numbers and dates also appear in the kept document, so a notice with a new date or amount is kept. Longest documents are checked first, so the most complete copy is the one kept. A duplicate's text is stored under the document it duplicates (`duplicate_of`).
- **Step 5, enrich:** rules assign `category`, `department`, `doc_date` (years 2008 to next year) and `audience` (UG / PG / PhD / all).
- **Step 6, chunk:** splits at headings first, then by size: **450 tokens with 45 tokens (10%) overlap**, using the embedding model's tokenizer. Tables are split by rows with the header row repeated. Each chunk's `embed_text` starts with a context line, `Title > Section | Source: URL`, and stays within the model's 512-token limit.
- **Output:** `output/chunks.jsonl`, **18,371 chunks**. Fields: chunk_id, doc_id, chunk_index, title, section, source_url, path, domain, file_type, is_ocr, category, department, doc_date, audience, text, embed_text.

### 4.2 Embeddings: `embed.py` (step 7)
- Provider is set in `.env` with `EMBEDDING_PROVIDER`, currently **local `BAAI/bge-base-en-v1.5`** (768 dimensions, normalized). Gemini `gemini-embedding-001` is the alternative.
- Embedding all chunks on this CPU took about 3 hours. It saves progress every 512 chunks and resumes if interrupted.
- **Output:** `output/embeddings/local__BAAI_bge-base-en-v1.5/` (`embeddings.npy` + `chunk_ids.json`).

### 4.3 Vector store: `vector_store.py` (step 8)
- Qdrant runs in **local mode** in the `qdrant_db/` folder; setting `QDRANT_URL` in `.env` switches to a server.
- **Collection:** `iitgn__local__BAAI_bge-base-en-v1.5`, cosine distance. Every chunk field except `embed_text` is stored as payload, plus `doc_year` (a number) for dated chunks.
- **Payload indexes** (category, department, audience, file_type, domain, doc_year) are **only created on a Qdrant server** (`QDRANT_URL` set). In the current local folder mode they are skipped; filtered searches still work, just without the speed-up.
- The upload is skipped when `vector_store.json` shows the collection already holds these exact embeddings; otherwise the collection is rebuilt from scratch and checked (point count, and a chunk's own vector finds that chunk).
- Exports `CHUNKS_FILE`, `COLLECTION` and `connect()`, which `retrieve.py` uses.

### 4.4 Retrieval: `retrieve.py` (design section 2), built this session
`Retriever.search(query, filters=None)` runs these steps:
1. **Query understanding, done with rules:** `extract_preferences()` finds department (CSE, EE, ...), audience (PhD / PG / UG), category (fees first, then hostel, admissions, ...) and year. These are **soft preferences**: each one adds an extra filtered dense list (limit 10) to the merge. Strict filtering happens only through an explicit `filters` dict.
   - *Why soft:* 92% of chunks have no department, 91% have audience "all", and many have no date, so hard filters would drop good chunks.
   - *Known misfires:* department short forms (CSE, EE, ME, CE, ...) are matched in capitals on the raw question, so an all-caps question such as "TELL ME ..." picks ME; "gate" in "main gate" picks the admissions category. As these are soft preferences, the effect is small.
2. **Hybrid search:**
   - Dense: Qdrant `query_points`, top 30. Queries get the BGE prefix `"Represent this sentence for searching relevant passages: "`; chunks don't.
   - Keyword: BM25 (`rank_bm25`), top 30, built in memory from `chunks.jsonl` at startup (about 3 s). The tokenizer splits letters from digits, so "CS612" matches "CS 612".
   - Merge: **Reciprocal Rank Fusion** (RRF, k = 60).
3. **Rerank:** cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` with Sigmoid for 0-1 scores, over 40 candidates, about 3 s per question.
   - *Why this model:* `BAAI/bge-reranker-base` took about 40 s per question on this CPU.
4. **Post-process:**
   - **RRF again, merging the hybrid order with the reranker order.** On its own, the reranker pushed down correct chunks that hybrid search had ranked first; the merge fixed that.
   - A small recency bonus for fees and admissions (+0.05 this year, decaying 0.01 per year).
   - At most 2 chunks per document; 8 results returned.
5. **Confidence:** the best rerank score among the results, compared with `MIN_CONFIDENCE = 0.05`.

### 4.5 Retrieval evaluation: `evaluate.py` + `evaluation/questions.jsonl`
- **50 questions**, each with id, category, type (fact / date / number / table / acronym / no_answer), question, expected answer, and sources (parts of document paths).
  - q01-q44 are answerable, across admissions, academics, fees, hostel, campus, events, library, faculty, research, administration and placements.
  - q45-q50 have no answer in the data: Wi-Fi password, IIT Bombay hostel fee, parking fine, regular BTech tuition, IPL 2025, capital of France.
- **Metrics:** document-level Hit@1, 3, 5 and 8, plus MRR. Sources are matched through `manifest.csv`, following `duplicate_of`.
- **Comparison:** dense, keyword, hybrid, and hybrid + rerank. The report also suggests a confidence threshold from the no-answer questions.
- **Answer-key fixes:** q12, q33 and q34 were widened after checking that other documents also held valid answers.

### 4.6 Generation: `generate.py` (design section 3), built this session
`Assistant.answer(question, show=callback)` runs these steps:
1. **Rewrite follow-ups:** when there is chat history (last 3 turns, first 500 characters of each old answer), the LLM rewrites "What about for PhD?" into a complete question. The rewritten question is shown as "(Searched for: ...)". Retrieval runs fresh every turn.
2. **Context:**
   - Sources are numbered `[n] (Title > Section | URL | date)`.
   - The best 3 results also get the chunk before and after them, and neighbouring chunks of a document are joined into one source with the shared overlap text removed. This fixed the dining table: its heading and price table had landed in separate chunks.
   - Sources are added in rank order up to about **5,000 tokens** (estimated at 4 characters per token). The best source is always kept; a source that doesn't fit is skipped, and smaller ones after it can still be added.
   - The strongest sources go first and last in the prompt (ranks 1, 3, 5 ... 4, 2), because models read the middle of a long context less carefully.
   - Today's date is included in the prompt.
3. **System prompt rules:**
   - Answer only from the sources.
   - Otherwise reply exactly "I could not find this in the IIT Gandhinagar data."; the same reply is used for questions about other institutes.
   - Cite as [1] or [2][3], citing the one or two most direct sources, not every source that repeats a fact.
   - Copy numbers exactly, prefer the newest source when sources disagree, and use Markdown tables for structured data.
4. **Generate:** the answer is streamed through the `show()` callback. Providers:
   - **Gemini (in use):** main model `gemini-3.8-flash`, fallback `gemini-3.6-flash`, thinking level `low` (3.1 s instead of 8.6 s in a test).
   - **Busy errors (429, 500, 503, 504):** retry after 2, 5 and 10 s, then switch to the fallback. A model that keeps failing is skipped for 5 minutes, except the last model in the list, which is always tried. Errors are only retried before any answer text has been shown.
   - **Daily-quota 429s** (error text contains "PerDay"): switch at once without retrying, and skip that model for 1 hour.
   - **Ollama** is a local alternative (`LLM_PROVIDER=ollama`, `llama3.2:3b`), never tested because Ollama isn't installed.
5. **Verify:** `check_citations()` removes citations to sources that don't exist (`[2, 99]` becomes `[2]`). The answer is followed by a Sources list of the cited sources with links. There's a warning when an answer cites nothing, and a low-confidence note when the score is below 0.05.
   - **Confidence gate design change:** a low score **does not skip the LLM**. It adds a caution to the prompt and a warning to the output.
   - *Why:* a reworded dining question ("mess lunch timings and prices in the dining hall") scored 0.034, below the cut-off, but was answerable, and the LLM answered it correctly.
6. **Log:** each turn is written to `output/chat_log.jsonl` with time, model used, question, rewritten question, answer, removed citations, confidence, rerank scores of retrieved chunks, the chunk IDs behind each source number, and timings.
   - **No `output/chat_log.jsonl` exists at the moment** (only `answer_evaluation_log.jsonl`), so the manual chat checks in section 6 are not in a log. The file is created by the next `python main.py chat` question.

`LAST_MODEL["name"]` records which Gemini model wrote the last answer.

### 4.7 Answer evaluation: `evaluate_answers.py`, built this session
- Every question is asked on its own (history cleared). The **judge model `gemini-3.5-flash-lite`**, a different model from the one answering, grades each answerable question:
  - **correct:** yes / partly / no (scored 1 / 0.5 / 0);
  - **faithful:** yes / no;
  - **citations:** yes / no.
- With `LLM_PROVIDER=ollama` the judge is the same local model. The judge has no fallback model.
- If the judge's JSON reply can't be read, the question is saved with a `judge_error` and is answered and graded again on the next run.
- No-answer questions pass if the answer contains the standard "not found" sentence.
- **Resumable:** progress is saved after each question to `output/answer_evaluation.json`; a re-run skips finished questions and stops after 3 failures in a row. Its log is kept separately in `output/answer_evaluation_log.jsonl`.
- **The report covers:**
  - grade counts, faithfulness rate, citation correctness and correctness score by category;
  - how often it wrongly answered "not found", and how many made-up citations were removed;
  - average answer time and the models used;
  - whether each no-answer question was declined;
  - a list of the problem answers, with the judge's reason.

---

## 5. Settings in `.env` / `config.py`

| Setting | Current value | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | local | or gemini |
| `EMBEDDING_BATCH_SIZE` | 16 | |
| `LOCAL_EMBEDDING_MODEL` | BAAI/bge-base-en-v1.5 | changing it requires re-running ingestion, because chunk limits depend on its tokenizer |
| `GEMINI_EMBEDDING_MODEL` / `_DIMENSION` | gemini-embedding-001 / 768 | only used if the provider is gemini |
| `QDRANT_URL` / `QDRANT_PATH` | empty / qdrant_db | |
| `LOCAL_QUERY_PREFIX` | BGE search prefix | |
| `RERANKER_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | |
| `LLM_PROVIDER` | gemini | or ollama |
| `GEMINI_API_KEY` | set by the user | one line in `.env` (the earlier empty duplicate line is gone) |
| `GEMINI_CHAT_MODEL` | gemini-3.8-flash | |
| `GEMINI_FALLBACK_MODEL` | gemini-3.6-flash | |
| `GEMINI_JUDGE_MODEL` | gemini-3.5-flash-lite | |
| `GEMINI_THINKING_LEVEL` | low | minimal / low / medium / high |
| `LLM_TEMPERATURE` | 0.25 (set by the user) | Google recommends the default for Gemini 3 models; empty means the default |
| `OLLAMA_URL` / `OLLAMA_MODEL` | localhost:11434 / llama3.2:3b | untested |

Constants that are in code, not `.env`:
- **`retrieve.py`:** DENSE_LIMIT 30, KEYWORD_LIMIT 30, PREFERRED_LIMIT 10, RRF_K 60, RERANK_LIMIT 40, FINAL_LIMIT 8, MAX_CHUNKS_PER_DOC 2, MIN_CONFIDENCE 0.05.
- **`retrieve.py`, recency:** RECENCY_BONUS 0.05, RECENCY_DECAY 0.01, for the categories fees and admissions.
- **`generate.py`:** MAX_CONTEXT_TOKENS 5000, NEIGHBOUR_RESULTS 3, HISTORY_TURNS 3, HISTORY_ANSWER_CHARS 500, MIN_OVERLAP_CHARS 20, RETRY_WAITS [2, 5, 10], BUSY_CODES {429, 500, 503, 504}, BUSY_COOLDOWN 300 s, DAILY_LIMIT_COOLDOWN 3600 s.
- **`evaluate_answers.py`:** MAX_FAILURES_IN_A_ROW 3.

Gemini model status was checked on Google's deprecation page on 2026-11-10:
- `gemini-2.0-flash` has been shut down.
- 3.8-flash, 3.6-flash and 2.5-flash have no shutdown date.
- `gemini-3.8-flash` often returned 503 "high demand" that day.

---

## 6. Results

### Retrieval: 44 answerable questions

| Setup | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR |
|---|---|---|---|---|---|
| dense | 57% | 89% | 93% | 95% | 0.73 |
| keyword (BM25) | 64% | 75% | 84% | 89% | 0.71 |
| hybrid | 70% | 91% | 95% | 98% | 0.80 |
| reranker order alone (old) | 73% | 89% | — | 91% | — |
| **hybrid + rerank (RRF merge, current)** | **82%** | 89% | 93% | **98%** | **0.87** |

- **Only miss:** q35, "How do I access e-journals from home?". A library guide that only says remote access exists came up instead.
- **Every category is fully covered except library** (3/4).
- **Confidence scores:**
  - Unanswerable questions: Wi-Fi 0.036, parking 0.000, France 0.000; but IIT Bombay 0.951, BTech tuition 0.932 and IPL 0.705.
  - Lowest answerable questions: 0.103 (psychiatrist), 0.162 (mess lunch).
  - The 0.05 cut-off catches 3 of the 6 unanswerable questions and wrongly flags none of the 44 answerable ones. Near-miss questions can't be caught by the score; the answer prompt has to handle them.

### Answers: 7 of 50 graded so far (stopped by the free-tier daily limit)
- **q01-q07 (all admissions):** 7/7 correct, faithful and correctly cited, about 10 s per answer, written by `gemini-3.6-flash`. These 7 were also graded by `gemini-3.6-flash`, before the judge moved to flash-lite.
- **Manual end-to-end checks that worked:**
  - scholarship amount;
  - a follow-up ("Who is eligible for it?", rewritten correctly);
  - the dining lunch price table, even though its score was low;
  - IIT Bombay correctly declined.
- **Free-tier limit:** 20 requests per model per day. Each question needs 2 calls (answer + judge), so about 20 questions can be graded per day. 43 remain (`output/answer_evaluation.json` holds q01-q07): run `python main.py evaluate-answers` daily, or turn on billing in Google AI Studio to finish in one run of about 15 minutes.

---

## 7. Key decisions and why

| Decision | Reason |
|---|---|
| Local `bge-base-en-v1.5` embeddings | No API limits; works on CPU; 512-token limit sets the chunk size |
| Soft preferences instead of hard metadata filters | Most chunks lack department, audience and date |
| BM25 built in memory, not Qdrant sparse vectors | Simple; about 3 s to build; lets "CS 612" match "CS612" |
| MiniLM-L-6 reranker | bge-reranker-base was about 13 times slower on this CPU |
| RRF merge of hybrid and reranker orders | Raised Hit@1 from 73% to 82% and Hit@8 from 91% to 98%. k = 60 is the standard value, so it isn't tuned to these 44 questions |
| MIN_CONFIDENCE 0.05 (was 0.3) | 0.3 flagged correct answers; off-topic questions scored ≤ 0.04 and answerable ones ≥ 0.10 |
| Low confidence warns but doesn't block | A reworded answerable question scored below the cut-off |
| Neighbour-chunk expansion | Headings and their tables sometimes land in separate chunks |
| Thinking level `low` | About 3 times faster; answering from given sources needs little reasoning |
| Separate judge model | Avoids a model grading itself, and spreads use across daily quotas |

---

## 8. Known issues and inaccuracies

- **Comment error in `generate.py`:** the `join_texts()` docstring (line 167) calls the chunk overlap "about 15%"; it is **10%** (45 tokens).
- **Outdated run commands** in the `data_ingest.py` and `embed.py` docstrings (see section 3).
- **Payload indexes** are not created in local mode, although the earlier summary said they were (see section 4.3).
- **`requirements.txt`** lists no versions and no `torch`, so a fresh install may not get the CPU-only torch 2.14.1 used here.
- **Hostel fee retrieval:** MTech and PhD hostel fee questions mostly bring back annual reports instead of fee documents.
- **Garbled OCR chunks** still appear in results (e.g. RTI Guidelines, the bakery price list). Filtering them requires a re-embed of about 3 hours.
- **Threshold reliability:** the 0.05 cut-off rests on only 6 unanswerable questions, with small margins.
- **Possible question bias:** the evaluation questions were written while reading the sources, so their wording may make retrieval look better than real student questions would.
- **Startup time:** about 30 s (loading chunks, BM25, models). Part of it is that `main.py` imports `data_ingest` for every command, including `chat` and `search`, and that import loads the BGE tokenizer and langchain. Each question takes about 5 s for retrieval plus 3-15 s for Gemini.
- **Repeated citations:** annual reports repeat the same facts every year, so answers used to cite many of them. The prompt now asks for one or two sources; this hasn't been re-measured yet.
- **Ollama:** the alternative path has never been run.

---

## 9. Pending tasks

### Open (pick up when the user asks)
1. Finish the answer evaluation (43 questions left), then read the problem list it prints.
2. **Claim check** (design 3.4, optional): a second LLM call checks each sentence against its cited source.
3. **Summarise older turns** (design 3.5): turns before the last 3 are currently dropped.
4. **User feedback** recorded in the chat log (design 3.6).
5. Fix the hostel fee retrieval (MTech and PhD).
6. Fix q35 (library e-journals from home).
7. Junk / garbled chunk filter (needs a re-embed).
8. More unanswerable questions, to set `MIN_CONFIDENCE` more reliably.
9. Optional LLM-based detail extraction (design 2.1) instead of rules.
10. Review the 50 questions; add real student questions and follow-up questions (target about 100).
11. Chunk size comparison at 300 / 500 / 800 tokens (design 4.4); each size needs a re-chunk and re-embed.
12. Add nDCG to the retrieval report (design 4.2).
13. Small cleanups found on 2026-11-11: fix the "15%" docstring, update the old run commands in the docstrings, add `torch` and versions to `requirements.txt`, and tighten the query rules that misfire (capital short forms, "gate").

### Deferred: the user said not to implement these for now
14. FastAPI backend with streaming answers.
15. Simple UI (Streamlit or React).
16. Docker.
17. Scheduled re-ingestion that only processes changed files.
18. `git init` in the project folder. The existing `.gitignore` already excludes `.env`, `output/` and `qdrant_db/`; `evaluation/` should be tracked.
19. Reducing the 30 s startup time (including the `data_ingest` import in `main.py`).

---

## 10. Suggested reading order for the code

1. `main.py`: the commands.
2. `config.py`: every setting.
3. `retrieve.py`: start at `Retriever.search()` and follow each step.
4. `generate.py`: start at `Assistant.answer()`, then `build_sources()` and `check_citations()`, then `create_llm()` for the retry and fallback logic.
5. `evaluate.py` and `evaluate_answers.py`: how quality is measured.
6. `data_ingest.py`, `embed.py`, `vector_store.py`: how the index was built (only re-run these if the data or embedding model changes; the full rebuild takes hours).
