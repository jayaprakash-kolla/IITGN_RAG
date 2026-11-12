"""
IITGN RAG - Embedding Pipeline (Step 7 of the design document)

Turns every chunk into a vector, using the provider set in the .env file.
Input : output/chunks.jsonl
Output: output/embeddings/<provider>__<model>/embeddings.npy  (one vector per chunk)
        output/embeddings/<provider>__<model>/chunk_ids.json  (same order as the vectors)

Run: python embed.py              (all chunks, resumes if interrupted)
     python embed.py --limit 200  (quick speed test)
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_DIMENSION,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
)

CHUNKS_FILE = Path("output/chunks.jsonl")
EMBEDDINGS_DIR = Path("output/embeddings")

SAVE_EVERY = 512          # Progress is saved after this many chunks
MAX_RETRIES = 5           # Attempts for a failed Gemini request
RETRY_WAIT_SECONDS = 60   # Pause after hitting a Gemini rate limit


# ---------- Embedding providers ----------

def create_embedder():
    """Return the model name and a function that turns texts into vectors."""
    if EMBEDDING_PROVIDER == "local":
        return LOCAL_EMBEDDING_MODEL, create_local_embedder()
    if EMBEDDING_PROVIDER == "gemini":
        return GEMINI_EMBEDDING_MODEL, create_gemini_embedder()
    raise SystemExit(f"EMBEDDING_PROVIDER must be local or gemini, not '{EMBEDDING_PROVIDER}'")


def create_local_embedder():
    """Load the model that runs on this computer."""
    # Imported here so that only the chosen provider is loaded
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

    def embed(texts):
        return model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return embed


def create_gemini_embedder():
    """Connect to the Gemini API."""
    if not GEMINI_API_KEY:
        raise SystemExit("EMBEDDING_PROVIDER is gemini, but GEMINI_API_KEY is empty in .env")

    # Imported here so that only the chosen provider is loaded
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=GEMINI_API_KEY)
    settings = types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=GEMINI_EMBEDDING_DIMENSION,
    )

    def request_vectors(batch):
        """Send one batch and return one vector per text."""
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL, contents=batch, config=settings
        )
        vectors = [embedding.values for embedding in response.embeddings or []]

        # Stop rather than save vectors that do not match the chunks
        if len(vectors) != len(batch) or not all(vectors):
            raise RuntimeError(f"Gemini returned {len(vectors)} vectors for {len(batch)} texts")
        return vectors

    def embed_batch(batch):
        """Send one batch, waiting and retrying on temporary errors."""
        for _ in range(MAX_RETRIES - 1):
            try:
                return request_vectors(batch)
            except errors.APIError as error:
                is_temporary = error.code == 429 or error.code >= 500
                if not is_temporary:
                    raise
                print(f"Gemini error {error.code}, retrying in {RETRY_WAIT_SECONDS} seconds")
                time.sleep(RETRY_WAIT_SECONDS)

        # Final attempt: an error here stops the run
        return request_vectors(batch)

    def embed(texts):
        vectors = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            vectors.extend(embed_batch(texts[start:start + EMBEDDING_BATCH_SIZE]))
        # Gemini only normalizes full-size vectors, so normalize them here
        vectors = np.array(vectors, dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return embed


# ---------- Reading chunks ----------

def load_chunks(limit):
    """Read the chunks, or an evenly spread sample when a limit is given."""
    with open(CHUNKS_FILE, encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file]

    if limit and limit < len(chunks):
        step = len(chunks) // limit
        chunks = chunks[::step][:limit]
    return chunks


# ---------- Saving progress ----------

def prepare_run_folder(model_name, chunks, is_test):
    """Create the output folder, and clear it if the chunks have changed."""
    run_name = f"{EMBEDDING_PROVIDER}__{model_name}".replace("/", "_")
    run_dir = EMBEDDINGS_DIR / (f"{run_name}__test" if is_test else run_name)

    # A fingerprint detects whether saved progress still matches the chunks
    all_text = "".join(chunk["embed_text"] for chunk in chunks)
    fingerprint = hashlib.sha256(all_text.encode()).hexdigest()
    info_file = run_dir / "run_info.json"

    if info_file.exists() and json.loads(info_file.read_text())["fingerprint"] != fingerprint:
        print("Chunks have changed, starting again")
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    info_file.write_text(json.dumps({"model": model_name, "fingerprint": fingerprint}))
    return run_dir


def save_part(part_file, vectors):
    """Save vectors so that an interruption cannot leave a broken file."""
    temporary_file = part_file.with_suffix(".tmp")
    with open(temporary_file, "wb") as file:
        np.save(file, vectors)
    temporary_file.replace(part_file)


def merge_parts(run_dir, chunks):
    """Combine the saved parts into the final files."""
    part_files = sorted(run_dir.glob("part_*.npy"))
    vectors = np.concatenate([np.load(part) for part in part_files])

    if len(vectors) != len(chunks):
        raise SystemExit(f"Expected {len(chunks)} vectors but found {len(vectors)}")

    np.save(run_dir / "embeddings.npy", vectors)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    (run_dir / "chunk_ids.json").write_text(json.dumps(chunk_ids))

    # The parts are no longer needed
    for part in part_files:
        part.unlink()
    return vectors


# ---------- Run the pipeline ----------

def run_embeddings():
    print("starting the embeddings")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="embed only this many chunks, for testing")
    # parse_known_args skips the "index" command that main.py receives
    args, _ = parser.parse_known_args()

    chunks = load_chunks(args.limit)
    print(f"Provider: {EMBEDDING_PROVIDER}, chunks: {len(chunks)}")

    model_name, embed = create_embedder()
    run_dir = prepare_run_folder(model_name, chunks, is_test=bool(args.limit))

    # Nothing to do if this exact job has already finished
    if (run_dir / "embeddings.npy").exists():
        print(f"Already finished, see {run_dir}")
        return

    start_time = time.time()
    embedded = 0   # Chunks embedded in this run, used to estimate speed

    for part_number, start in enumerate(range(0, len(chunks), SAVE_EVERY)):
        part_file = run_dir / f"part_{part_number:05d}.npy"
        batch = chunks[start:start + SAVE_EVERY]

        # Skip work already saved by an earlier run
        if part_file.exists():
            print(f"Part {part_number} already done")
            continue

        vectors = embed([chunk["embed_text"] for chunk in batch])
        save_part(part_file, np.asarray(vectors, dtype=np.float32))

        embedded += len(batch)
        speed = embedded / (time.time() - start_time)
        remaining = len(chunks) - start - len(batch)
        print(f"Done {start + len(batch)}/{len(chunks)} chunks, "
              f"{speed:.1f} chunks/s, about {remaining / speed / 60:.0f} min left")

    vectors = merge_parts(run_dir, chunks)
    print(f"\nSaved {vectors.shape[0]} vectors of size {vectors.shape[1]} to {run_dir}")
    print("embeddings are completed")

