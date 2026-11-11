"""
IITGN RAG - Vector Store (Step 8 of the design document)

Loads the chunk embeddings into a Qdrant collection so they can be searched.
Input : output/chunks.jsonl
        output/embeddings/<provider>__<model>/embeddings.npy and chunk_ids.json
Output: Qdrant collection named iitgn__<provider>__<model>, saved in the
        qdrant_db folder (or on a Qdrant server if QDRANT_URL is set in .env)

Run: python main.py
"""

import json
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from config import EMBEDDING_RUN_NAME, QDRANT_PATH, QDRANT_URL

CHUNKS_FILE = Path("output/chunks.jsonl")
RUN_DIR = Path("output/embeddings") / EMBEDDING_RUN_NAME
STORE_INFO_FILE = RUN_DIR / "vector_store.json"   # Records what was last stored

# Each provider and model gets its own collection, since their vectors cannot be mixed
COLLECTION = f"iitgn__{EMBEDDING_RUN_NAME}"
UPLOAD_BATCH = 256   # Points sent to Qdrant at a time

# Metadata fields that searches can filter on
KEYWORD_FIELDS = ["category", "department", "audience", "file_type", "domain"]


# ---------- Loading ----------

def load_embeddings():
    """Read the chunks and their vectors, and check that they match."""
    if not (RUN_DIR / "embeddings.npy").exists():
        raise SystemExit(f"No embeddings in {RUN_DIR}, run the embedding step first")

    with open(CHUNKS_FILE, encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file]
    vectors = np.load(RUN_DIR / "embeddings.npy")
    chunk_ids = json.loads((RUN_DIR / "chunk_ids.json").read_text())

    # Vectors are stored in chunk order, so the order must be identical
    if chunk_ids != [chunk["chunk_id"] for chunk in chunks]:
        raise SystemExit("Embeddings do not match the current chunks, run the embedding step again")

    fingerprint = json.loads((RUN_DIR / "run_info.json").read_text())["fingerprint"]
    return chunks, vectors, fingerprint


# ---------- Qdrant ----------

def connect():
    """Connect to a Qdrant server, or open the database folder."""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=QDRANT_PATH)


def database_location():
    return QDRANT_URL or QDRANT_PATH


def is_up_to_date(client, fingerprint, point_count):
    """Check whether the collection already holds exactly these embeddings."""
    if not STORE_INFO_FILE.exists() or not client.collection_exists(COLLECTION):
        return False

    expected = {"collection": COLLECTION, "location": database_location(), "fingerprint": fingerprint}
    stored = json.loads(STORE_INFO_FILE.read_text())
    return stored == expected and client.count(COLLECTION).count == point_count


def create_collection(client, vector_size):
    """Create an empty collection, replacing any old one."""
    # Rebuilding from scratch means removed chunks never linger in the database
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )

    # Indexes speed up filtered searches; the local folder mode does not use them
    if QDRANT_URL:
        for field in KEYWORD_FIELDS:
            client.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "doc_year", models.PayloadSchemaType.INTEGER)


def build_payload(chunk):
    """Choose what is stored with each vector."""
    # The text with its title line is only needed for embedding
    payload = {key: value for key, value in chunk.items() if key != "embed_text"}

    # A number allows searches like "documents from 2024 onwards"
    if chunk["doc_date"]:
        payload["doc_year"] = int(chunk["doc_date"][:4])
    return payload


def point_id(chunk_id):
    """Qdrant needs a UUID, so derive a stable one from the chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def upload_points(client, chunks, vectors):
    """Send the vectors and their metadata to Qdrant in batches."""
    for start in range(0, len(chunks), UPLOAD_BATCH):
        batch = range(start, min(start + UPLOAD_BATCH, len(chunks)))
        points = [
            models.PointStruct(
                id=point_id(chunks[i]["chunk_id"]),
                vector=vectors[i].tolist(),
                payload=build_payload(chunks[i]),
            )
            for i in batch
        ]
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        print(f"Stored {batch.stop}/{len(chunks)} chunks")


def verify(client, chunks, vectors):
    """Check the point count, and that a chunk's own vector finds that chunk."""
    stored = client.count(COLLECTION).count
    if stored != len(chunks):
        raise SystemExit(f"Expected {len(chunks)} points but the collection has {stored}")

    result = client.query_points(COLLECTION, query=vectors[0].tolist(), limit=1)
    if result.points[0].payload["chunk_id"] != chunks[0]["chunk_id"]:
        raise SystemExit("Search check failed: a chunk did not find itself")
    print(f"Verified: {stored} points, and search finds the right chunk")


# ---------- Run the pipeline ----------

def run_vector_store():
    print("starting the vector store")
    chunks, vectors, fingerprint = load_embeddings()
    client = connect()

    try:
        # Nothing to do if this exact data is already stored
        if is_up_to_date(client, fingerprint, len(chunks)):
            print(f"Collection {COLLECTION} is already up to date")
            return

        create_collection(client, vector_size=vectors.shape[1])
        upload_points(client, chunks, vectors)
        verify(client, chunks, vectors)

        info = {"collection": COLLECTION, "location": database_location(), "fingerprint": fingerprint}
        STORE_INFO_FILE.write_text(json.dumps(info))
        print(f"Saved collection {COLLECTION} to {database_location()}")
    finally:
        # The local folder can only be opened by one program at a time
        client.close()

    print("vector store is completed")
