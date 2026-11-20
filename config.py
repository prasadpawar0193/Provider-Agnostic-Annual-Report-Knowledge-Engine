
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"

ALLOWED_EXTENSIONS = {"pdf"}

MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5

# Minimum similarity score for accepting retrieved content
MIN_SIMILARITY_SCORE = 0.05

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

MISSING_ANSWER = "Information not found in the provided document"
