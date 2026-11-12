# IITGN RAG Assistant

A question-answering assistant for **IIT Gandhinagar (IITGN)**. You ask a question in plain English, for example *"What is the campus shuttle schedule?"*, and it answers using only official IITGN web pages and PDFs. It also shows which pages the answer came from.

It uses **RAG (Retrieval-Augmented Generation)**:

1. **Retrieve:** find the parts of the IITGN documents that best match your question.
2. **Generate:** give those parts to an LLM (Google Gemini or a local Ollama model), which writes the answer and cites its sources as [1], [2] and so on.

You can use it in the browser through a **Streamlit app**, or in the terminal.

---

## Table of contents

1. [What is in this project](#1-what-is-in-this-project)
2. [How it works](#2-how-it-works)
3. [Project structure](#3-project-structure)
4. [Requirements](#4-requirements)
5. [Setup: run it on your own computer](#5-setup-run-it-on-your-own-computer)
6. [Using the Streamlit app (version 1)](#6-using-the-streamlit-app-version-1)
7. [All commands](#7-all-commands)
8. [Settings in the .env file](#8-settings-in-the-env-file)
9. [Evaluation results](#9-evaluation-results)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What is in this project

- **Data:** about 1,600 text files scraped from IITGN websites (`iitgn.ac.in`, `campus`, `library`, `cse`, `ccl`, `sports`, `safety` and others). They cover admissions, fees, hostels, dining, academics, the library, campus services and more. The files are in the `data/` folder.
- **Pipeline:** Python scripts that clean the data, split it into about 18,000 small pieces called *chunks*, turn each chunk into a vector, and store the vectors in a search database (Qdrant).
- **Search:** hybrid search that combines meaning-based search with keyword search (BM25), then reorders the results with a reranker model.
- **Answers:** an LLM writes the answer using only the retrieved chunks and cites them.
- **Frontend:** a Streamlit web app (`app_v1.py`) with a **Chat** page and a **Search** page.
- **Evaluation:** 50 test questions with expected answers (`evaluation/questions.jsonl`) to measure how well the search and the answers work.

---

## 2. How it works

```text
             BUILD THE INDEX (once, offline)                      ANSWER A QUESTION (every time)
 ┌──────────────────────────────────────────────┐     ┌─────────────────────────────────────────────┐
 │ data/  (IITGN pages and PDFs as text files)  │     │ Your question                               │
 │   │                                          │     │   │                                         │
 │   ▼  data_ingest.py                          │     │   ▼  rewrite follow-ups ("What about PhD?") │
 │ clean → remove duplicates → add metadata     │     │   ▼  retrieve.py                            │
 │   → split into chunks (output/chunks.jsonl)  │     │ meaning search (Qdrant) + keyword (BM25)    │
 │   │                                          │     │   → merge → rerank → best 8 chunks          │
 │   ▼  embed.py                                │     │   │                                         │
 │ chunk → vector (BAAI/bge-base-en-v1.5)       │     │   ▼  generate.py                            │
 │   │                                          │     │ LLM (Gemini or Ollama) writes the answer    │
 │   ▼  vector_store.py                         │     │ with citations [1], [2] ...                 │
 │ vectors → Qdrant database (qdrant_db/)       │     │   │                                         │
 └──────────────────────────────────────────────┘     │   ▼  Streamlit app or terminal              │
                                                      └─────────────────────────────────────────────┘
```

**Building the index** happens once. It reads every file in `data/`, cleans it, removes duplicates, and splits it into chunks. A local embedding model turns each chunk into a vector, and the vectors are saved in a Qdrant database in the `qdrant_db/` folder.

**Answering a question** happens every time you ask something:

1. If the question is a follow-up (for example *"Who is eligible for it?"*), it is rewritten into a full question using the earlier chat.
2. Two searches run: a meaning-based search in Qdrant and a keyword search (BM25). Their results are merged.
3. A reranker model scores the best 40 candidates against the question, and the best 8 chunks are kept.
4. The LLM gets those chunks, numbered, and is told to answer **only** from them and to cite them.
5. Citations that point to sources that do not exist are removed. If even the best chunk scored low, a "Low confidence" warning is shown.
6. Every question and answer is logged to `output/chat_log.jsonl`.

The full design is in [IITGN_RAG_Pipeline_Design.md](IITGN_RAG_Pipeline_Design.md).

---

## 3. Project structure

```text
IITGN_RAG/
├── main.py                       Starting point: runs the app or any command below
├── app_v1.py                     Streamlit frontend, version 1 (Chat and Search pages)
├── config.py                     Reads the settings from the .env file
├── data_ingest.py                Cleans the data and splits it into chunks
├── embed.py                      Turns the chunks into vectors
├── vector_store.py               Saves the vectors in the Qdrant database
├── retrieve.py                   Hybrid search + reranking
├── generate.py                   Writes answers with the LLM (Gemini or Ollama)
├── evaluate.py                   Measures search quality on the test questions
├── evaluate_answers.py           Grades the LLM's answers on the test questions
├── requirements.txt              Python packages to install
├── data/                         The IITGN source files (text), one folder per website
├── evaluation/
│   └── questions.jsonl           50 test questions with expected answers
│
│   Created on your computer (not in Git):
├── .env                          Your settings and API key (you create this, see step 4 below)
├── .venv/                        Python virtual environment
├── output/                       Chunks, vectors, logs and evaluation results
└── qdrant_db/                    The vector database
```

---

## 4. Requirements

| What | Details |
|---|---|
| **Python** | Version **3.10** (the project was built and tested with 3.10.11). 3.11 should also work. |
| **Git** | To download the project. |
| **Disk space** | About 3 GB free (Python packages, AI models and the database). |
| **RAM** | 8 GB or more. |
| **Internet** | Needed to install packages, download the AI models the first time, and call Gemini. |
| **Gemini API key** | Free. Needed for answers in Chat (unless you use Ollama instead). See step 4 below. |
| **GPU** | Not needed. Everything runs on a normal CPU. |

> **Time warning:** building the index (step 6 below) took **about 3 hours** on a laptop CPU, because it turns about 18,000 chunks into vectors. You only do this **once**. It saves its progress, so if it stops, run the same command again and it continues where it stopped.

---

## 5. Setup: run it on your own computer

Follow the steps in order. Commands are shown for **Windows** and for **macOS / Linux** where they differ.

### Step 1: Download the project

```bash
git clone https://github.com/jayaprakash-kolla/IITGN_RAG.git
cd IITGN_RAG
```

All the commands below are run from inside this folder.

### Step 2: Create a virtual environment

A virtual environment keeps this project's packages separate from the rest of your computer.

**Windows (PowerShell or Command Prompt):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When it is active, you will see `(.venv)` at the start of your terminal line.
**Every time you open a new terminal to use this project, run the activate command again.**

> On Windows, if PowerShell says *"running scripts is disabled on this system"*, run this once and then try again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Step 3: Install the packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes. It installs PyTorch, sentence-transformers, Qdrant, Streamlit, the Gemini client and the other packages.

### Step 4: Create the `.env` settings file

The `.env` file holds your settings and your API key. It is **not** in Git (so keys are never shared), so you must create it.

**4a. Get a free Gemini API key**

1. Go to **https://aistudio.google.com/apikey** and sign in with a Google account.
2. Click **Create API key** and copy the key.

**4b. Create the file**

In the project folder (next to `main.py`), create a file named exactly **`.env`** and paste this in. Replace `paste-your-key-here` with your key.

```ini
# ---------- Embeddings (turning text into vectors) ----------
EMBEDDING_PROVIDER=local
EMBEDDING_BATCH_SIZE=16
LOCAL_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_EMBEDDING_DIMENSION=768

# ---------- Vector database ----------
# Leave QDRANT_URL empty to store the database in the qdrant_db folder
QDRANT_URL=
QDRANT_PATH=qdrant_db

# ---------- Search ----------
LOCAL_QUERY_PREFIX="Represent this sentence for searching relevant passages: "
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ---------- Answers (LLM) ----------
# gemini (needs GEMINI_API_KEY) or ollama (runs on your computer)
LLM_PROVIDER=gemini
GEMINI_API_KEY=paste-your-key-here
GEMINI_CHAT_MODEL=gemini-3.5-flash-lite
GEMINI_FALLBACK_MODEL=gemini-3.8-flash
GEMINI_JUDGE_MODEL=gemini-3.6-flash
GEMINI_THINKING_LEVEL=low
LLM_TEMPERATURE=0.25

# Only used when LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

> **Windows tip:** Notepad may save the file as `.env.txt`. In the *Save as* window, choose **Save as type: All files** and type the name as `.env`.

**Never commit `.env` or share your API key.** It is already listed in `.gitignore`.

### Step 5: Check that everything is installed

```bash
python -c "import streamlit, sentence_transformers, qdrant_client; print('All good')"
```

If it prints `All good`, continue. If not, run step 3 again and read any error it shows.

### Step 6: Build the index (one time only)

```bash
python main.py index
```

This runs three stages one after another:

| Stage | What it does | Output | Time |
|---|---|---|---|
| 1. Ingest | Cleans the files in `data/`, removes duplicates, splits them into chunks | `output/chunks.jsonl`, `output/manifest.csv` | A few minutes |
| 2. Embed | Turns each chunk into a vector | `output/embeddings/` | **About 3 hours on a CPU** |
| 3. Store | Saves the vectors in the Qdrant database | `qdrant_db/` | A few minutes |

Good to know:

- The first run downloads the embedding model (about 440 MB) from Hugging Face.
- The embed stage prints progress and the estimated time left.
- **If it stops** (you close the laptop, press Ctrl+C, and so on), run `python main.py index` again. Finished parts are skipped and it continues from where it stopped.
- When it is done, you will see `embeddings are completed` and the vector store stage finishing.

> **Shortcut:** if someone who already built the index gives you their `output/` and `qdrant_db/` folders, copy both folders into the project folder and skip this step. Both of you must use the same `LOCAL_EMBEDDING_MODEL` in `.env`.

### Step 7: Start the Streamlit app

```bash
python main.py
```

Your browser opens **http://localhost:8501** with the app. If it does not open by itself, open that address yourself.

- The **first start takes about a minute** while the models load. After that, it is fast.
- To stop the app, go to the terminal and press **Ctrl+C**.

---

## 6. Using the Streamlit app (version 1)

`app_v1.py` is a simple web page on top of the same pipeline used in the terminal. Pick a mode in the left sidebar:

### Chat mode

The full assistant: it searches the IITGN data and the LLM writes an answer.

- Type a question in the box at the bottom, or click one of the **Try asking** example questions in the sidebar.
- The answer appears as it is written, with citations such as [1] and [2].
- Open **Sources** under the answer to see the pages it used, with links.
- You can ask **follow-up questions** (for example *"Who is eligible for it?"*). If the question was rewritten, the app shows what it actually searched for.
- **New chat** clears the conversation.
- A yellow **Low confidence** box means even the best match scored low, so check the sources before trusting the answer.

### Search mode

Search only, with **no LLM** (no API key needed and no API usage). It shows the chunks the search found, ranked, with their scores, source link, category and date. Use it to see what the assistant would read before answering.

### Show details (sidebar switch)

Turn it on to see extra information under each answer or search:

- the top match score and the confidence threshold;
- the time taken by each stage;
- citations that were removed because they pointed to sources that do not exist;
- the exact text (context) the LLM was given.

The sidebar also shows which LLM provider is in use (`gemini` or `ollama`).

---

## 7. All commands

Run these from the project folder with the virtual environment active.

| Command | What it does | Needs the index? | Uses the LLM? |
|---|---|---|---|
| `python main.py` | Starts the Streamlit app (Chat and Search pages) | Yes | Chat only |
| `python main.py index` | Builds the index: ingest, embed, store | No (it builds it) | No |
| `python main.py chat` | Chat with the assistant in the terminal | Yes | Yes |
| `python main.py search` | Search the chunks in the terminal, no LLM | Yes | No |
| `python main.py evaluate` | Measures search quality on the 50 test questions | Yes | No |
| `python main.py evaluate-answers` | Grades the LLM's answers on the test questions | Yes | Yes (2 calls per question) |
| `python embed.py --limit 200` | Quick speed test: embeds only 200 chunks | No | No |

> **Run only one of these at a time.** The local Qdrant database can be opened by only one program at a time. Close the Streamlit app (Ctrl+C) before running `chat`, `search`, `evaluate` or `index`.

---

## 8. Settings in the `.env` file

You normally only need to set `GEMINI_API_KEY`. The other settings are optional.

| Setting | Default | What it means |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Which LLM writes the answers: `gemini` (online) or `ollama` (on your computer). |
| `GEMINI_API_KEY` | (empty) | Your Gemini API key. Needed when `LLM_PROVIDER=gemini`. |
| `GEMINI_CHAT_MODEL` | `gemini-3.5-flash-lite` | Gemini model that writes the answers. |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.8-flash` | Used when the main model is busy or out of quota. Leave empty to turn off. |
| `GEMINI_JUDGE_MODEL` | `gemini-3.6-flash` | Model that grades answers in `evaluate-answers`. |
| `GEMINI_THINKING_LEVEL` | `low` | How much Gemini thinks before answering: `minimal`, `low`, `medium` or `high`. `low` is faster. |
| `LLM_TEMPERATURE` | `0.25` | Lower values keep answers closer to the sources. Empty uses the model's default. |
| `EMBEDDING_PROVIDER` | `local` | `local` (free, on your computer) or `gemini`. |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model. **If you change it, rebuild the index.** |
| `EMBEDDING_BATCH_SIZE` | `16` | Chunks embedded together. Lower it if your computer runs out of memory. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model (about 3 seconds per question on a CPU). |
| `QDRANT_PATH` | `qdrant_db` | Folder for the local vector database. |
| `QDRANT_URL` | (empty) | Set it (for example `http://localhost:6333`) only to use a Qdrant server instead of the folder. |

### Using Ollama instead of Gemini (no API key, fully offline answers)

1. Install Ollama from **https://ollama.com**.
2. Download a model: `ollama pull llama3.2:3b`
3. In `.env`, set `LLM_PROVIDER=ollama`.
4. Make sure Ollama is running, then start the app with `python main.py`.

Small local models are slower and less accurate than Gemini, but they are free and work offline.

---

## 9. Evaluation results

Measured with `python main.py evaluate` on the 44 test questions that have an answer in the data. *Hit@k* means the correct document is among the first *k* results.

| Search setup | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR |
|---|---|---|---|---|---|
| Meaning search only | 57% | 89% | 93% | 95% | 0.73 |
| Keyword search only (BM25) | 64% | 75% | 84% | 89% | 0.71 |
| Hybrid (both merged) | 70% | 91% | 95% | 98% | 0.80 |
| **Hybrid + reranker (used by the app)** | **82%** | 89% | 93% | **98%** | **0.87** |

The first 7 graded answers (admissions questions) were all correct, faithful to the sources, and correctly cited.

> **Free-tier limit:** the free Gemini tier allows about 20 requests per model per day. `evaluate-answers` needs 2 requests per question, so it saves its progress and skips finished questions when you run it again the next day.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `(.venv)` is not shown, or `No module named ...` | Activate the virtual environment (step 2), then run `pip install -r requirements.txt`. |
| App says **"The assistant could not start"** | Build the index first (`python main.py index`), check `GEMINI_API_KEY` in `.env`, and close any other program using the database. |
| **"Storage folder qdrant_db is already accessed by another instance"** | Another command or app is still using the database. Stop it with Ctrl+C (or close that terminal) and try again. |
| `No embeddings in output/embeddings/...` | The embed stage did not finish. Run `python main.py index` again; it continues where it stopped. |
| Chat says **"Could not answer: ... 429"** or **"quota"** | The Gemini daily free limit is used up. Wait until tomorrow, use another API key, or switch to Ollama. |
| Chat says **"Could not answer: ... 503"** | Gemini is busy. The app already retries and tries the fallback model; wait a minute and ask again. |
| `.env` changes seem to be ignored | Check the file is named `.env` and not `.env.txt`, and restart the app after changing it. |
| Embedding is very slow or the computer freezes | Lower `EMBEDDING_BATCH_SIZE` to `8` in `.env` and close other heavy programs. Progress is saved, so you can stop and restart. |
| Port 8501 is already in use | Another Streamlit app is running. Close it, or run `python -m streamlit run app_v1.py --server.port 8502`. |
| PowerShell: *"running scripts is disabled"* | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |

---

**Data source:** public pages and documents from IIT Gandhinagar websites (`*.iitgn.ac.in`). Answers may be out of date if the websites have changed, so check important details (fees, dates, deadlines) on the official site.
