"""
IITGN RAG - Generation Pipeline (section 3 of the design document)

Answers a question from the retrieved chunks, in six steps:
1. Rewrite  : turn a follow-up ("What about PhD?") into a full question, using the last few turns
2. Context  : number the chunks with title, link and date, add the chunks next to the best ones,
              and put the strongest sources first and last
3. Prompt   : rules - answer only from the sources, cite them as [1], copy numbers exactly
4. Generate : the LLM writes the answer, shown as it arrives
5. Verify   : drop citations to sources that do not exist, and list the cited sources with links
6. Log      : save question, chunks, scores, answer and timings to output/chat_log.jsonl

Retrieval runs fresh for every question; old chunks are never reused.

Input : the retriever from retrieve.py and an LLM (Gemini or Ollama, chosen in .env)
Run   : python main.py chat
"""

import io
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests

from config import (
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    GEMINI_FALLBACK_MODEL,
    GEMINI_THINKING_LEVEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_MODEL,
    OLLAMA_URL,
)
from retrieve import MIN_CONFIDENCE, Retriever

# ---------- Settings ----------

MAX_CONTEXT_TOKENS = 5000   # Design: keep the context around 3-6k tokens
NEIGHBOUR_RESULTS = 3       # The best 3 results also get the chunk before and after them
HISTORY_TURNS = 3           # Earlier turns used to rewrite follow-up questions
HISTORY_ANSWER_CHARS = 500  # Only the start of each old answer is needed for rewriting
MIN_OVERLAP_CHARS = 20      # Shorter matches between chunks are treated as chance
RETRY_WAITS = [2, 5, 10]    # Seconds to wait before each retry when Gemini is busy
BUSY_CODES = {429, 500, 503, 504}   # Too many requests, or a temporary server problem
BUSY_COOLDOWN = 300         # Seconds to skip a model after all its retries failed
DAILY_LIMIT_COOLDOWN = 3600 # Seconds to skip a model whose daily request limit is used up
LAST_MODEL = {"name": ""}   # The model that wrote the last answer (it may be the fallback)
LOG_FILE = Path("output/chat_log.jsonl")

NOT_FOUND = "I could not find this in the IIT Gandhinagar data."

SYSTEM_PROMPT = f"""You are the IIT Gandhinagar (IITGN) assistant. You answer questions using only the numbered sources given with each question.

Rules:
- Use only facts from the sources. Do not add outside knowledge.
- If the sources do not contain the answer, reply "{NOT_FOUND}" and name the source that looks most relevant, if any.
- The sources only cover IIT Gandhinagar. If the question is about another institute, reply "{NOT_FOUND}" and say the data only covers IIT Gandhinagar.
- Put the source number after each fact, like [1] or [2][3]. Only use numbers that appear in the sources.
- Cite the one or two most direct sources for a fact, not every source that repeats it.
- Copy numbers, dates, fees and names exactly as they are written in the source.
- If sources disagree, prefer the most recent one and mention the disagreement.
- Be concise. Use a Markdown table for fees, timetables and other structured data."""

REWRITE_PROMPT = """Rewrite the user's last question so it can be understood without the conversation.
Keep names, programmes, years and course codes. If it is already complete, return it unchanged.
Reply with the question only."""


# ---------- The LLM ----------

def create_llm(gemini_models=None):
    """Return a function llm(system, user) that yields the answer text piece by piece.

    gemini_models: Gemini models to try in order (default: the chat model, then the fallback).
    """
    temperature = float(LLM_TEMPERATURE) if LLM_TEMPERATURE else None

    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import errors, types

        if not GEMINI_API_KEY:
            raise SystemExit("Set GEMINI_API_KEY in .env, or set LLM_PROVIDER=ollama")
        client = genai.Client(api_key=GEMINI_API_KEY)
        models = [model for model in (gemini_models or [GEMINI_CHAT_MODEL, GEMINI_FALLBACK_MODEL]) if model]
        busy_until = {}   # model -> time until which it is skipped
        thinking = types.ThinkingConfig(thinking_level=GEMINI_THINKING_LEVEL) if GEMINI_THINKING_LEVEL else None

        def gemini(system, user):
            settings = types.GenerateContentConfig(
                system_instruction=system, temperature=temperature, thinking_config=thinking
            )
            last_error = None
            for model in models:
                # Skip a model that stayed busy recently, unless it is the last one left
                if time.time() < busy_until.get(model, 0) and model != models[-1]:
                    continue
                daily_limit = False
                for wait in [0] + RETRY_WAITS:
                    time.sleep(wait)
                    started = False
                    try:
                        for piece in client.models.generate_content_stream(
                            model=model, contents=user, config=settings
                        ):
                            if piece.text:   # Some pieces carry no text
                                started = True
                                LAST_MODEL["name"] = model
                                yield piece.text
                        return
                    except errors.APIError as error:
                        # Retry only "busy" errors, and only before any text was shown
                        if started or error.code not in BUSY_CODES:
                            raise
                        last_error = error
                        # The free tier allows a fixed number of requests per model per day;
                        # once that is used up, retrying today cannot help
                        daily_limit = "PerDay" in str(error)
                        if daily_limit:
                            print(f"[{model} daily limit reached]", end=" ", flush=True)
                            break
                        print(f"[{model} busy ({error.code}), retrying]", end=" ", flush=True)
                # Every retry failed: use the next model for a while
                busy_until[model] = time.time() + (DAILY_LIMIT_COOLDOWN if daily_limit else BUSY_COOLDOWN)
            raise last_error or RuntimeError("No Gemini model is set in .env")

        if not gemini_models:
            LAST_MODEL["name"] = GEMINI_CHAT_MODEL
        return gemini

    if LLM_PROVIDER == "ollama":
        LAST_MODEL["name"] = OLLAMA_MODEL

        def ollama(system, user):
            request = {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": True,
            }
            if temperature is not None:
                request["options"] = {"temperature": temperature}
            with requests.post(f"{OLLAMA_URL}/api/chat", json=request, stream=True, timeout=300) as response:
                response.raise_for_status()
                # Ollama sends one JSON object per line
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line).get("message", {}).get("content", "")

        return ollama

    raise SystemExit(f"LLM_PROVIDER must be gemini or ollama, not '{LLM_PROVIDER}'")


# ---------- Helpers ----------

def estimate_tokens(text):
    # About 4 characters per token for English text; close enough for a budget
    return len(text) // 4


def join_texts(texts):
    """Join neighbouring chunks of one document, dropping the text they share (about 15%)."""
    joined = texts[0]
    for text in texts[1:]:
        overlap = 0
        for size in range(min(len(joined), len(text)), MIN_OVERLAP_CHARS - 1, -1):
            if joined.endswith(text[:size]):
                overlap = size
                break
        joined += "\n" + text[overlap:]
    return joined


def split_into_runs(indexes):
    """[3, 4, 5, 9] -> [[3, 4, 5], [9]]"""
    runs = []
    for index in sorted(indexes):
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def strongest_at_the_edges(sources):
    """Order ranks 1,2,3,4,5 as 1,3,5,4,2: models read the start and end of a context best."""
    return sources[0::2] + sources[1::2][::-1]


def check_citations(answer, source_count):
    """Remove citations like [9] that point to no source. Return the clean answer and the bad numbers."""
    bad = set()

    def keep_valid(match):
        numbers = [int(number) for number in re.findall(r"\d+", match.group(0))]
        valid = [number for number in numbers if 1 <= number <= source_count]
        bad.update(number for number in numbers if number not in valid)
        return "".join(f"[{number}]" for number in valid)

    # Matches [3] and also [2, 3]
    clean = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", keep_valid, answer)
    cited = sorted({int(number) for number in re.findall(r"\[(\d+)\]", clean)})
    return clean, cited, sorted(bad)


# ---------- The assistant ----------

class Assistant:
    """Retrieves, builds the context, asks the LLM and checks the answer."""

    def __init__(self, log_file=LOG_FILE):
        self.log_file = log_file
        self.retriever = Retriever()
        self.llm = create_llm()
        # (doc_id, chunk_index) -> chunk_id, to find the chunks next to a result
        self.chunk_at = {(chunk["doc_id"], chunk["chunk_index"]): chunk_id
                         for chunk_id, chunk in self.retriever.chunks.items()}
        self.history = []   # Earlier (question, answer) pairs

    def close(self):
        self.retriever.close()

    def ask_llm(self, system, user, show=None):
        """Collect the LLM's answer, passing each piece to show() as it arrives."""
        pieces = []
        for piece in self.llm(system, user):
            pieces.append(piece)
            if show:
                show(piece)
        return "".join(pieces).strip()

    # ----- Step 1: Rewrite follow-up questions -----

    def rewrite(self, question):
        if not self.history:
            return question
        turns = "\n\n".join(f"User: {old_question}\nAssistant: {old_answer[:HISTORY_ANSWER_CHARS]}"
                            for old_question, old_answer in self.history[-HISTORY_TURNS:])
        user = f"Conversation:\n{turns}\n\nLast question: {question}"
        return self.ask_llm(REWRITE_PROMPT, user) or question

    # ----- Step 2: Build the context -----

    def build_sources(self, results):
        """Turn the ranked results into numbered sources that fit the token budget."""
        # Chunk indexes wanted from each document, with the best rank that asked for them
        wanted = {}
        for rank, result in enumerate(results):
            indexes = wanted.setdefault(result["doc_id"], {})
            around = [-1, 0, 1] if rank < NEIGHBOUR_RESULTS else [0]
            for step in around:
                index = result["chunk_index"] + step
                if (result["doc_id"], index) in self.chunk_at:
                    indexes[index] = min(rank, indexes.get(index, rank))

        # Chunks next to each other in a document become one source
        sources = []
        for doc_id, indexes in wanted.items():
            for run in split_into_runs(indexes):
                chunk_ids = [self.chunk_at[(doc_id, index)] for index in run]
                chunks = [self.retriever.chunks[chunk_id] for chunk_id in chunk_ids]
                best = min(run, key=lambda index: indexes[index])
                first = self.retriever.chunks[self.chunk_at[(doc_id, best)]]
                sources.append({
                    "rank": indexes[best],
                    "chunk_ids": chunk_ids,
                    "title": first["title"],
                    "section": first["section"],
                    "url": first["source_url"] or first["path"],
                    "date": first["doc_date"],
                    "text": join_texts([chunk["text"] for chunk in chunks]),
                })
        sources.sort(key=lambda source: source["rank"])

        # Keep the best sources until the budget is used (always keep the first)
        kept, used = [], 0
        for source in sources:
            tokens = estimate_tokens(source["text"])
            if kept and used + tokens > MAX_CONTEXT_TOKENS:
                continue
            kept.append(source)
            used += tokens

        kept = strongest_at_the_edges(kept)
        for number, source in enumerate(kept, start=1):
            source["number"] = number
        return kept

    def format_context(self, sources):
        blocks = []
        for source in sources:
            name = f"{source['title']} > {source['section']}" if source["section"] else source["title"]
            blocks.append(f"[{source['number']}] ({name} | {source['url']} | {source['date'] or 'no date'})\n"
                          f"{source['text']}")
        return "\n\n".join(blocks)

    # ----- All steps together -----

    def answer(self, question, show=None):
        """Answer one question. show() receives the answer text as it is written."""
        timings = {}

        start = time.time()
        standalone = self.rewrite(question)
        timings["rewrite"] = time.time() - start

        retrieval = self.retriever.search(standalone)
        timings.update(retrieval["timings"])
        results = retrieval["results"]

        if not results:
            text = NOT_FOUND
            if show:
                show(text)
            sources, cited, bad = [], [], []
        else:
            sources = self.build_sources(results)
            # Confidence gate: a low score only warns the LLM. It does not skip it, because some
            # answerable questions score below MIN_CONFIDENCE too (e.g. a reworded dining question).
            caution = (f"Note: the search found only weak matches, so the answer may not be in the "
                       f"sources. If no source clearly states it, reply \"{NOT_FOUND}\"\n\n"
                       if retrieval["low_confidence"] else "")
            user = (f"Today's date: {date.today().isoformat()}\n\n"
                    f"Sources:\n{self.format_context(sources)}\n\n"
                    f"{caution}Question: {standalone}")
            start = time.time()
            raw = self.ask_llm(SYSTEM_PROMPT, user, show)
            timings["generate"] = time.time() - start
            # Step 5: Verify the citations
            text, cited, bad = check_citations(raw, len(sources))

        self.history.append((question, text))
        reply = {
            "question": question,
            "standalone_question": standalone,
            "answer": text,
            "sources": [source for source in sources if source["number"] in cited],
            "context": self.format_context(sources),   # Exactly what the LLM saw
            "removed_citations": bad,
            "low_confidence": retrieval["low_confidence"],
            "top_score": retrieval["top_score"],
            "timings": {stage: round(seconds, 2) for stage, seconds in timings.items()},
        }
        self.log(reply, results, sources)
        return reply

    # ----- Step 6: Log -----

    def log(self, reply, results, sources):
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "llm": f"{LLM_PROVIDER}:{LAST_MODEL['name']}",
            **{key: value for key, value in reply.items() if key not in ("sources", "context")},
            "retrieved": [{"chunk_id": result["chunk_id"], "rerank_score": result["rerank_score"]}
                          for result in results],
            "context_chunks": {source["number"]: source["chunk_ids"] for source in sources},
        }
        self.log_file.parent.mkdir(exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- Run the pipeline ----------

def print_reply_footer(reply):
    """Show what the answer was based on, after the streamed answer."""
    if reply["standalone_question"] != reply["question"]:
        print(f"\n\n(Searched for: {reply['standalone_question']})", end="")
    if reply["low_confidence"]:
        print(f"\n\n(Low confidence: best match scored {reply['top_score']:.2f}, "
              f"below {MIN_CONFIDENCE}. Check the sources.)", end="")
    if reply["removed_citations"]:
        print(f"\n\nNote: removed citations to sources that do not exist: {reply['removed_citations']}", end="")
    if reply["sources"]:
        print("\n\nSources:")
        for source in reply["sources"]:
            print(f"  [{source['number']}] {source['title']} | {source['url']} | {source['date'] or 'no date'}")
    elif NOT_FOUND not in reply["answer"]:
        print("\n\nWarning: the answer cites no source, so check it before trusting it.")
    else:
        print()
    print(f"Time taken (seconds): {reply['timings']}")


def run_chat():
    print("starting the chat")
    # Some characters in the data (e.g. the rupee sign) break the default Windows encoding
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    assistant = Assistant()
    print(f"Answers come from {LLM_PROVIDER}; low-confidence cut-off is {MIN_CONFIDENCE}")
    try:
        print("\nAsk a question (press Enter on an empty line to stop)")
        while True:
            question = input("\nQuestion: ").strip()
            if not question:
                break
            print("\nAnswer: ", end="")
            try:
                reply = assistant.answer(question, show=lambda piece: print(piece, end="", flush=True))
            except Exception as error:   # e.g. no internet or an API limit; keep the chat open
                print(f"\nCould not answer: {error}")
                continue
            print_reply_footer(reply)
    finally:
        assistant.close()
    print(f"chat is completed (log saved to {LOG_FILE})")
