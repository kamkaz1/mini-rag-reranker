"""
Configuration settings for the Mini RAG system
"""
import os

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300  # words
CHUNK_OVERLAP = 50  # words

# Search settings
DEFAULT_K = 10
CANDIDATE_K = 30  # Number of candidates for reranking
SIMILARITY_THRESHOLD = 0.7  # Threshold for answer confidence

# Reranker settings
ALPHA = 0.6  # Weight for vector score in hybrid reranker (1-ALPHA for keyword score)

# Database settings
DATABASE_PATH = "chunks.db"
FAISS_INDEX_PATH = "faiss_index.bin"

# Reproducibility
RANDOM_SEED = 42

# File paths
PDF_DIR = "pdfs"
SOURCES_FILE = "sources.json"
