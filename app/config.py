import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
PDF_DIR = os.getenv("PDF_DIR", "./data/pdfs")
TOP_K = int(os.getenv("TOP_K", "3"))
COLLECTION = "policies"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Default documents: pristine copies (committed to git). Any that are missing from PDF_DIR are restored at startup.
SAMPLES_DIR = os.getenv("SAMPLES_DIR", "./data/samples")
RESTORE_SAMPLES_ON_START = os.getenv("RESTORE_SAMPLES_ON_START", "true").lower() not in ("0", "false", "no")

# Retrieval-level evidence check: if the BEST chunk's cosine distance is above this, abstain without calling
# the LLM. This is a tuned cutoff, not a calibrated probability. Empty string disables it.
_md = os.getenv("EVIDENCE_MAX_DISTANCE", "0.55")
MAX_DISTANCE = float(_md) if _md.strip() else None
CHUNK_CHARS = 800
