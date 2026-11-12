import subprocess
import sys
from pathlib import Path

# Each command imports only what it needs, so the frontend starts without loading the
# ingestion and embedding code.

FRONTEND = Path(__file__).parent / "app_v1.py"


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""

    # python main.py chat : ask questions in the terminal and get answers with sources
    if command == "chat":
        from generate import run_chat
        run_chat()
        return

    # python main.py search : ask questions against the built index in the terminal
    if command == "search":
        from retrieve import run_retrieval
        run_retrieval()
        return

    # python main.py evaluate : score retrieval on evaluation/questions.jsonl
    if command == "evaluate":
        from evaluate import run_evaluation
        run_evaluation()
        return

    # python main.py evaluate-answers : grade the LLM's answers to the same questions
    if command == "evaluate-answers":
        from evaluate_answers import run_answer_evaluation
        run_answer_evaluation()
        return

    # python main.py index : build the index
    if command == "index":
        from data_ingest import run_data_ingestion_pipeline
        from embed import run_embeddings
        from vector_store import run_vector_store
        run_data_ingestion_pipeline()
        run_embeddings()
        run_vector_store()
        return

    if command:
        raise SystemExit(f"Unknown command '{command}'. "
                         "Use: index, chat, search, evaluate, evaluate-answers, or nothing for the frontend")

    # python main.py : start the Streamlit frontend (Chat and Search pages)
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(FRONTEND)], check=False)
    except KeyboardInterrupt:
        pass   # Ctrl+C stops the frontend


if __name__ == "__main__":
    main()
