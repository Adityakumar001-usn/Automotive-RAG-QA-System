"""
Configuration for the Automotive RAG Question Answering System.
Holds constants for both Phase 1 (Ingestion) and Phase 2 (QA System).
"""

# Text Chunking Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval Settings
TOP_K = 5
ENABLE_RETRIEVAL_LOGGING = False

# Embedding Model (Phase 1/2)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Inference Settings (Phase 2)
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
SUPPORTED_LLMS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2b-it",
    "Qwen/Qwen2.5-3B-Instruct"
]
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1
USE_4BIT = True
MAX_CONTEXT_CHARS = 3000
