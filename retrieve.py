"""
IITGN RAG - Retrieval Pipeline (section 2 of the design document)

Finds the chunks that best answer a question, in five steps:
1. Understand the query : find the department, audience, topic and year it mentions
2. Hybrid search        : meaning search (Qdrant) + keyword search (BM25), merged with RRF
3. Rerank               : a cross-encoder scores every candidate against the question
4. Post-process         : merge the reranker order with the hybrid order (RRF), prefer recent
                          fee and admission documents, limit chunks per document
5. Confidence gate      : flag the result as unsure when even the best chunk scores low

Input : output/chunks.jsonl and the Qdrant collection built by vector_store.py
Run   : python main.py search
"""

import json
import re
import time
from datetime import date

import numpy as np
import torch
from qdrant_client import models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_DIMENSION,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_QUERY_PREFIX,
    RERANKER_MODEL,
)
from vector_store import CHUNKS_FILE, COLLECTION, connect

# ---------- Settings ----------

DENSE_LIMIT = 30        # Chunks taken from the meaning search
KEYWORD_LIMIT = 30      # Chunks taken from the keyword search
PREFERRED_LIMIT = 10    # Chunks taken from each search limited to a detail in the question
RRF_K = 60              # Standard Reciprocal Rank Fusion constant
RERANK_LIMIT = 40       # Merged candidates sent to the reranker
FINAL_LIMIT = 8         # Chunks returned
MAX_CHUNKS_PER_DOC = 2  # Keeps one long document from filling every slot

# Recent documents win for topics where old information is wrong
TIME_SENSITIVE = {"fees", "admissions"}
RECENCY_BONUS = 0.05    # Added to the score of this year's documents
RECENCY_DECAY = 0.01    # Bonus lost for each year of age
THIS_YEAR = date.today().year

# Below this reranker score the answer is probably not in the data.
# From the evaluation: off-topic questions scored 0.04 or less, answerable ones 0.10 or more.
# Re-check it with python main.py evaluate if the reranker model changes.
MIN_CONFIDENCE = 0.05

# Common words that carry no meaning in a keyword search
STOPWORDS = set(
    "a an the and or of to in on at for by with from is are was were be been what which who whom "
    "when where why how do does did can could should would will shall i me my we our you your it "
    "its this that these those there here about as into than then so if not no any all".split()
)


# ---------- Step 1: Understand the query ----------

# (department, short form matched only in capitals, full name)
QUERY_DEPARTMENT_RULES = [
    ("CSE", r"CSE", r"computer science|computer engineering"),
    ("EE", r"EE", r"electrical"),
    ("ME", r"ME", r"mechanical"),
    ("CE", r"CE", r"civil"),
    ("CL", r"", r"chemical engineering"),
    ("BE", r"", r"biological|biomedical|bioengineering"),
    ("MSE", r"MSE", r"materials science|materials engineering"),
    ("CH", r"", r"chemistry"),
    ("PH", r"", r"physics"),
    ("MA", r"", r"mathematics|maths"),
    ("ES", r"", r"earth sciences?"),
    ("CGS", r"CGS", r"cognitive science"),
    ("HSS", r"HSS", r"humanities|social sciences"),
    ("AR", r"", r"archaeology"),
]

# The first matching rule wins, as in data_ingest.py
QUERY_AUDIENCE_RULES = [
    ("PhD", r"phd|ph d|doctoral|doctorate"),
    ("PG", r"mtech|m tech|msc|m sc|ma|masters?|postgraduate|pg|emasters"),
    ("UG", r"btech|b tech|undergraduate|ug|bachelors?"),
]

QUERY_CATEGORY_RULES = [
    ("fees", r"fees?|tuition|scholarships?|stipends?|fellowships?"),
    ("hostel", r"hostels?|mess|dining|food|menu|guest house"),
    ("admissions", r"admissions?|apply|eligibility|cutoffs?|jam|gate|jee|coap"),
    ("placements", r"placements?|internships?|recruiters?|cds"),
    ("library", r"library|books?|journals?"),
    ("research", r"research|publications?|patents?"),
    ("events", r"events?|workshops?|festivals?|conclave"),
    ("campus", r"campus|shuttle|transport|safety"),
    ("administration", r"rti|tenders?|recruitment|annual reports?|nirf"),
    ("faculty", r"professors?|faculty"),
]


def has_word(pattern, text):
    """Check whether any whole word in the pattern appears in the text."""
    return bool(pattern) and re.search(rf"\b(?:{pattern})\b", text) is not None


def extract_preferences(query):
    """Find details in the question that the answer should probably match."""
    # "M.Tech" becomes "m tech", matching the rules above
    words = re.sub(r"[^a-z0-9]+", " ", query.lower())
    preferences = {}

    for department, short_form, full_name in QUERY_DEPARTMENT_RULES:
        if has_word(short_form, query) or has_word(full_name, words):
            preferences["department"] = department
            break

    for field, rules in [("audience", QUERY_AUDIENCE_RULES), ("category", QUERY_CATEGORY_RULES)]:
        for value, pattern in rules:
            if has_word(pattern, words):
                preferences[field] = value
                break

    # A year between the institute's founding and next year
    for year in re.findall(r"\b(20\d\d)\b", query):
        if 2008 <= int(year) <= THIS_YEAR + 1:
            preferences["doc_year"] = int(year)
            break
    return preferences


# ---------- Helpers ----------

def tokenize(text):
    """Split text into lowercase words for the keyword search."""
    # Letters and digits are split apart, so "CS612" and "CS 612" both match
    return [word for word in re.findall(r"[a-z]+|\d+", text.lower()) if word not in STOPWORDS]


def build_filter(conditions):
    """Turn {"field": value} into a Qdrant filter where every condition must match."""
    if not conditions:
        return None
    return models.Filter(must=[
        models.FieldCondition(key=field, match=models.MatchValue(value=value))
        for field, value in conditions.items()
    ])


def reciprocal_rank_fusion(ranked_lists):
    """Merge ranked lists: a chunk scores 1 / (60 + rank) in every list it appears in."""
    scores = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
    return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)


def recency_bonus(chunk):
    """A small score bonus for recent fee and admission documents."""
    if chunk["category"] not in TIME_SENSITIVE or chunk["doc_year"] is None:
        return 0.0
    return max(0.0, RECENCY_BONUS - RECENCY_DECAY * (THIS_YEAR - chunk["doc_year"]))


def create_query_embedder():
    """Return a function that turns a question into a vector, matching the stored vectors."""
    if EMBEDDING_PROVIDER == "local":
        model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        # Questions get the model's search prefix; stored chunks did not
        return lambda query: model.encode(LOCAL_QUERY_PREFIX + query, normalize_embeddings=True)

    if EMBEDDING_PROVIDER == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        # Gemini uses a task type instead of a prefix
        settings = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY", output_dimensionality=GEMINI_EMBEDDING_DIMENSION
        )

        def embed(query):
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL, contents=[query], config=settings
            )
            embeddings = response.embeddings or []
            if not embeddings or not embeddings[0].values:
                raise RuntimeError("Gemini returned no vector for the question")
            vector = np.array(embeddings[0].values, dtype=np.float32)
            return vector / np.linalg.norm(vector)

        return embed

    raise SystemExit(f"EMBEDDING_PROVIDER must be local or gemini, not '{EMBEDDING_PROVIDER}'")


# ---------- The retriever ----------

class Retriever:
    """Loads the models and indexes once, then answers many questions."""

    def __init__(self):
        print("Loading chunks and building the keyword index")
        with open(CHUNKS_FILE, encoding="utf-8") as file:
            chunk_list = [json.loads(line) for line in file]
        for chunk in chunk_list:
            # Same number as in Qdrant, so filters behave the same in both searches
            chunk["doc_year"] = int(chunk["doc_date"][:4]) if chunk["doc_date"] else None

        self.chunk_ids = [chunk["chunk_id"] for chunk in chunk_list]
        self.chunks = {chunk["chunk_id"]: chunk for chunk in chunk_list}
        # The title line helps here too, e.g. a department name in the title
        self.bm25 = BM25Okapi([tokenize(chunk["embed_text"]) for chunk in chunk_list])

        print("Opening the vector database")
        self.client = connect()
        if not self.client.collection_exists(COLLECTION):
            raise SystemExit(f"Collection {COLLECTION} not found, run python main.py index first")

        print("Loading the embedding model and the reranker")
        self.embed_query = create_query_embedder()
        # Sigmoid turns every model's raw output into a score from 0 to 1
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512, activation_fn=torch.nn.Sigmoid())

    def close(self):
        # The local database folder can only be opened by one program at a time
        self.client.close()

    # ----- Step 2: Hybrid search -----

    def dense_search(self, vector, conditions, limit):
        """Return chunk_ids ranked by meaning, closest first."""
        response = self.client.query_points(
            COLLECTION,
            query=vector.tolist(),
            query_filter=build_filter(conditions),
            limit=limit,
            with_payload=["chunk_id"],
        )
        return [point.payload["chunk_id"] for point in response.points if point.payload]

    def keyword_search(self, query, filters, limit):
        """Return chunk_ids ranked by BM25 keyword score, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = []
        for index in np.argsort(-scores):
            if scores[index] <= 0 or len(ranked) == limit:
                break   # Remaining chunks share no words with the question
            chunk = self.chunks[self.chunk_ids[index]]
            if all(chunk.get(field) == value for field, value in filters.items()):
                ranked.append(chunk["chunk_id"])
        return ranked

    def hybrid_search(self, query, vector, filters, preferences):
        """Merge the meaning search, keyword search and preferred searches with RRF."""
        ranked_lists = [
            self.dense_search(vector, filters, DENSE_LIMIT),
            self.keyword_search(query, filters, KEYWORD_LIMIT),
        ]
        # One extra list per detail in the question, e.g. only fee chunks
        for field, value in preferences.items():
            if field not in filters:
                ranked_lists.append(self.dense_search(vector, {**filters, field: value}, PREFERRED_LIMIT))
        return reciprocal_rank_fusion(ranked_lists)

    # ----- Step 3: Rerank -----

    def rerank(self, query, candidates):
        """Score each candidate against the question, from 0 (unrelated) to 1."""
        pairs = [(query, self.chunks[chunk_id]["embed_text"]) for chunk_id in candidates]
        scores = self.reranker.predict(pairs, batch_size=8)
        return [float(score) for score in scores]

    # ----- Step 4: Post-process -----

    def select_final(self, candidates, rerank_scores):
        """Merge the hybrid order with the reranker order, and keep at most 2 chunks per document."""
        rerank_by_id = dict(zip(candidates, rerank_scores))

        # Reranker order, with a small bonus for recent fee and admission documents
        reranked = sorted(candidates, reverse=True,
                          key=lambda chunk_id: rerank_by_id[chunk_id] + recency_bonus(self.chunks[chunk_id]))

        # RRF again, so one low reranker score cannot drop a chunk the hybrid search ranked first.
        # The evaluation showed this beats either order on its own.
        order = reciprocal_rank_fusion([candidates, reranked])

        results, per_doc = [], {}
        for chunk_id in order:
            chunk = self.chunks[chunk_id]
            if per_doc.get(chunk["doc_id"], 0) == MAX_CHUNKS_PER_DOC:
                continue
            per_doc[chunk["doc_id"]] = per_doc.get(chunk["doc_id"], 0) + 1
            results.append({"rerank_score": round(rerank_by_id[chunk_id], 4),
                            **{key: value for key, value in chunk.items() if key != "embed_text"}})
            if len(results) == FINAL_LIMIT:
                break
        return results

    # ----- All steps together -----

    def search(self, query, filters=None):
        """Find the best chunks for a question.

        filters: optional strict conditions, e.g. {"category": "fees"}.
        Details found in the question itself only boost matching chunks,
        because most chunks have no department, audience or date.
        """
        filters = filters or {}
        timings = {}

        start = time.time()
        preferences = extract_preferences(query)
        vector = self.embed_query(query)
        timings["embed_query"] = time.time() - start

        start = time.time()
        candidates = self.hybrid_search(query, vector, filters, preferences)[:RERANK_LIMIT]
        timings["search"] = time.time() - start

        start = time.time()
        rerank_scores = self.rerank(query, candidates) if candidates else []
        timings["rerank"] = time.time() - start

        results = self.select_final(candidates, rerank_scores)

        # Step 5: Confidence gate, based on the best reranker score among the results
        top_score = max((result["rerank_score"] for result in results), default=0.0)
        return {
            "query": query,
            "preferences": preferences,
            "results": results,
            "top_score": top_score,
            "low_confidence": top_score < MIN_CONFIDENCE,
            "timings": {stage: round(seconds, 2) for stage, seconds in timings.items()},
        }


# ---------- Run the pipeline ----------

def print_results(answer):
    """Show the results in the terminal."""
    print(f"\nDetails found in the question: {answer['preferences'] or 'none'}")
    print(f"Time taken (seconds): {answer['timings']}")
    if answer["low_confidence"]:
        print(f"Low confidence: best score {answer['top_score']:.2f} is below {MIN_CONFIDENCE}")

    for rank, result in enumerate(answer["results"], start=1):
        source = result["source_url"] or result["path"]
        preview = " ".join(result["text"].split())[:200]
        print(f"\n[{rank}] {result['rerank_score']:.3f} | {result['title']} > {result['section']}")
        print(f"    {source} | {result['category']} | {result['doc_date'] or 'no date'}")
        print(f"    {preview}")


def run_retrieval():
    print("starting retrieval")
    retriever = Retriever()
    try:
        print("\nAsk a question (press Enter on an empty line to stop)")
        while True:
            query = input("\nQuestion: ").strip()
            if not query:
                break
            print_results(retriever.search(query))
    finally:
        retriever.close()
    print("retrieval is completed")
