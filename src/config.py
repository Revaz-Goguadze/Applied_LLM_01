"""config for plagiarism detection system"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("missing GEMINI_API_KEY - set it in .env file")

# models
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

# chunking limits
MIN_FUNCTION_LINES = 3
MAX_FUNCTION_LINES = 500

# retrieval settings
DEFAULT_TOP_K = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.78

# hybrid rag fusion weight (0=bm25 only, 1=dense only)
DEFAULT_ALPHA = 0.5

RANDOM_SEED = 42
