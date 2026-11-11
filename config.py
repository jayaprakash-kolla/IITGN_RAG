"""Loads the project settings from the .env file."""

import os

from dotenv import load_dotenv

# Values in the .env file take priority
load_dotenv(override=True)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_EMBEDDING_DIMENSION = int(os.getenv("GEMINI_EMBEDDING_DIMENSION", "768"))

# Model used by the chosen provider
EMBEDDING_MODEL = LOCAL_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "local" else GEMINI_EMBEDDING_MODEL
# Name for this provider and model, e.g. local__BAAI_bge-base-en-v1.5
EMBEDDING_RUN_NAME = f"{EMBEDDING_PROVIDER}__{EMBEDDING_MODEL}".replace("/", "_")

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_db")

# Added to questions only, never to chunks (BGE models expect it)
LOCAL_QUERY_PREFIX = os.getenv(
    "LOCAL_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM that writes the answers: gemini or ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
# Used when the main model stays busy; empty turns this off
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash").strip()
# Grades answers in python main.py evaluate-answers. A different model from the one answering
# avoids a model grading its own work, and uses a separate free-tier daily limit.
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite").strip()
# How much the model thinks before answering: minimal, low, medium or high (empty = model default)
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
# Empty means the model's own default
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE", "").strip()
