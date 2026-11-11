"""
IITGN RAG - Retrieval Evaluation (section 4 of the design document)

Checks how often retrieval finds the right document for the questions in
evaluation/questions.jsonl, and compares four setups:
  dense           : meaning search only
  keyword         : BM25 keyword search only
  hybrid          : both, merged with RRF
  hybrid + rerank : the full retrieval pipeline
It also suggests a confidence threshold, using the questions that have no answer in the data.

Input : evaluation/questions.jsonl (sources are parts of document paths)
Output: a printed report and output/evaluation_results.json
Run   : python main.py evaluate
"""

import csv
import json
import statistics
from pathlib import Path

from retrieve import FINAL_LIMIT, MIN_CONFIDENCE, Retriever, extract_preferences

QUESTIONS_FILE = Path("evaluation/questions.jsonl")
MANIFEST_FILE = Path("output/manifest.csv")
RESULTS_FILE = Path("output/evaluation_results.json")

K_VALUES = [1, 3, 5, 8]   # Hit@k: is a correct document among the first k chunks?
MODES = ["dense", "keyword", "hybrid", "hybrid + rerank"]


# ---------- Loading ----------

def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def resolve_sources(questions):
    """Turn each expected source (part of a file path) into document ids."""
    with open(MANIFEST_FILE, encoding="utf-8") as file:
        documents = list(csv.DictReader(file))

    for question in questions:
        question["doc_ids"] = set()
        for part in question["sources"]:
            matches = [doc for doc in documents if part in doc["path"]]
            if not matches:
                raise SystemExit(f"{question['id']}: no document path contains '{part}'")
            for doc in matches:
                # A duplicate's text is stored under the document it duplicates
                question["doc_ids"].add(doc["duplicate_of"] or doc["doc_id"])


# ---------- Scoring ----------

def first_hit_rank(chunk_ids, retriever, doc_ids):
    """Position (from 1) of the first chunk from a correct document, or 0 if none."""
    for rank, chunk_id in enumerate(chunk_ids[:FINAL_LIMIT], start=1):
        if retriever.chunks[chunk_id]["doc_id"] in doc_ids:
            return rank
    return 0


def rank_with_every_mode(retriever, query):
    """Return the chunk_ids each setup finds, plus the full pipeline's answer."""
    vector = retriever.embed_query(query)
    answer = retriever.search(query)
    ranked = {
        "dense": retriever.dense_search(vector, {}, FINAL_LIMIT),
        "keyword": retriever.keyword_search(query, {}, FINAL_LIMIT),
        "hybrid": retriever.hybrid_search(query, vector, {}, extract_preferences(query))[:FINAL_LIMIT],
        "hybrid + rerank": [result["chunk_id"] for result in answer["results"]],
    }
    return ranked, answer


def summarize(ranks):
    """Hit@k for each k, and Mean Reciprocal Rank (1 = always first)."""
    summary = {f"hit@{k}": sum(0 < rank <= k for rank in ranks) / len(ranks) for k in K_VALUES}
    summary["mrr"] = sum(1 / rank for rank in ranks if rank) / len(ranks)
    return summary


def suggest_threshold(answerable_scores, no_answer_scores):
    """Pick the cut-off that best separates answerable from unanswerable questions."""
    best_correct, best_threshold = -1, MIN_CONFIDENCE
    for step in range(1, 20):
        threshold = step / 20
        correct = (sum(score >= threshold for score in answerable_scores)
                   + sum(score < threshold for score in no_answer_scores))
        if correct > best_correct:
            best_correct, best_threshold = correct, threshold
    return best_threshold, best_correct


# ---------- Report ----------

def print_report(questions, records):
    answerable = [record for record in records if record["sources"]]
    no_answer = [record for record in records if not record["sources"]]

    print(f"\n===== Retrieval: {len(answerable)} answerable questions =====")
    print(f"{'Setup':<17}" + "".join(f"{'Hit@' + str(k):>8}" for k in K_VALUES) + f"{'MRR':>8}")
    for mode in MODES:
        summary = summarize([record["ranks"][mode] for record in answerable])
        print(f"{mode:<17}" + "".join(f"{summary[f'hit@{k}']:>8.0%}" for k in K_VALUES)
              + f"{summary['mrr']:>8.2f}")

    print("\nFull pipeline, Hit@8 by category:")
    for category in sorted({record["category"] for record in answerable}):
        ranks = [record["ranks"]["hybrid + rerank"] for record in answerable if record["category"] == category]
        print(f"  {category:<15} {sum(rank > 0 for rank in ranks)}/{len(ranks)}")

    print("\nMissed by the full pipeline (no correct document in the top 8):")
    for record in answerable:
        if not record["ranks"]["hybrid + rerank"]:
            top = record["top_results"][0] if record["top_results"] else "nothing"
            print(f"  {record['id']} {record['question']}\n      found instead: {top}")

    # Confidence gate: unanswerable questions should score low
    answerable_scores = [record["top_score"] for record in answerable]
    no_answer_scores = [record["top_score"] for record in no_answer]
    print(f"\n===== Confidence: {len(no_answer)} questions with no answer =====")
    for record in no_answer:
        print(f"  {record['top_score']:.3f}  {record['question']}")
    print(f"Answerable questions: lowest {min(answerable_scores):.3f}, "
          f"median {statistics.median(answerable_scores):.3f}")

    flagged = sum(score < MIN_CONFIDENCE for score in answerable_scores)
    caught = sum(score < MIN_CONFIDENCE for score in no_answer_scores)
    print(f"Current MIN_CONFIDENCE {MIN_CONFIDENCE}: catches {caught}/{len(no_answer)} unanswerable, "
          f"wrongly flags {flagged}/{len(answerable)} answerable")

    threshold, correct = suggest_threshold(answerable_scores, no_answer_scores)
    print(f"Suggested MIN_CONFIDENCE {threshold:.2f}: {correct}/{len(records)} questions handled correctly")


# ---------- Run the evaluation ----------

def run_evaluation():
    print("starting the evaluation")
    questions = load_questions()
    resolve_sources(questions)
    retriever = Retriever()

    records = []
    try:
        for number, question in enumerate(questions, start=1):
            print(f"Question {number}/{len(questions)}: {question['question']}")
            ranked, answer = rank_with_every_mode(retriever, question["question"])
            records.append({
                "id": question["id"],
                "category": question["category"],
                "question": question["question"],
                "sources": question["sources"],
                "ranks": {mode: first_hit_rank(ranked[mode], retriever, question["doc_ids"])
                          for mode in MODES},
                "top_score": answer["top_score"],
                "top_results": [f"{result['title']} | {result['path']}" for result in answer["results"][:3]],
            })
    finally:
        retriever.close()

    print_report(questions, records)
    RESULTS_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nDetails saved to {RESULTS_FILE}")
    print("evaluation is completed")
