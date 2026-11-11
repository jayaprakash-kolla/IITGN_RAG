import sys

from data_ingest import run_data_ingestion_pipeline
from embed import run_embeddings
from evaluate import run_evaluation
from evaluate_answers import run_answer_evaluation
from generate import run_chat
from retrieve import run_retrieval
from vector_store import run_vector_store

def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""

    # python main.py chat : ask questions and get answers with sources
    if command == "chat":
        run_chat()
        return

    # python main.py search : ask questions against the built index
    if command == "search":
        run_retrieval()
        return

    # python main.py evaluate : score retrieval on evaluation/questions.jsonl
    if command == "evaluate":
        run_evaluation()
        return

    # python main.py evaluate-answers : grade the LLM's answers to the same questions
    if command == "evaluate-answers":
        run_answer_evaluation()
        return

    # python main.py : build the index
    run_data_ingestion_pipeline()
    run_embeddings()
    run_vector_store()

if __name__ == "__main__":
    main()
