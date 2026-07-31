"""Central configuration for SubSense.

Every tunable lives here and is read from the environment, so nothing that
varies between machines (keys, paths, rate limits) is hardcoded in the engine
or the UI. See `.env.example` for the full list.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get_str(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get_str(name, str(default)))
    except ValueError:
        return default


# --- Credentials -----------------------------------------------------------
GEMINI_API_KEY = _get_str("GEMINI_API_KEY", "")

# --- Gemini ----------------------------------------------------------------
GEMINI_CHAT_MODEL = _get_str("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
GEMINI_EMBED_MODEL = _get_str("GEMINI_EMBED_MODEL", "models/text-embedding-004")

# --- Reddit ----------------------------------------------------------------
REDDIT_BASE_URL = _get_str("REDDIT_BASE_URL", "https://www.reddit.com").rstrip("/")
REDDIT_USER_AGENT = _get_str("REDDIT_USER_AGENT", "SubSense/2.0 (public JSON reader)")
REQUEST_TIMEOUT = _get_float("REQUEST_TIMEOUT", 15.0)
REQUEST_DELAY = _get_float("REQUEST_DELAY", 1.0)
MAX_RETRIES = _get_int("MAX_RETRIES", 3)
PAGE_SIZE = 100  # Reddit's own hard cap per listing request.
TIME_FILTERS = ("hour", "day", "week", "month", "year", "all")

# --- Persistence -----------------------------------------------------------
VIRALITY_MODEL_PATH = _get_str("VIRALITY_MODEL_PATH", "virality_model.json")
RAG_STORE_DIR = _get_str("RAG_STORE_DIR", "rag_store")

# --- Embedding / indexing --------------------------------------------------
EMBED_BATCH_SIZE = _get_int("EMBED_BATCH_SIZE", 20)
EMBED_BATCH_DELAY = _get_float("EMBED_BATCH_DELAY", 1.0)
RAG_TOP_K = _get_int("RAG_TOP_K", 5)

# --- Analysis --------------------------------------------------------------
MAX_CLUSTERS = _get_int("MAX_CLUSTERS", 5)
POSTS_PER_CLUSTER = _get_int("POSTS_PER_CLUSTER", 5)
RANDOM_SEED = _get_int("RANDOM_SEED", 42)
CACHE_TTL = _get_int("CACHE_TTL", 3600)

# Risk weights for the mod classifier. Tuned so a post needs at least two
# independent bad signals to clear HIGH_RISK_THRESHOLD.
RISK_LOCKED_WEIGHT = _get_float("RISK_LOCKED_WEIGHT", 0.8)
RISK_LOW_RATIO_WEIGHT = _get_float("RISK_LOW_RATIO_WEIGHT", 0.5)
RISK_SENTIMENT_WEIGHT = _get_float("RISK_SENTIMENT_WEIGHT", 0.5)
LOW_RATIO_THRESHOLD = _get_float("LOW_RATIO_THRESHOLD", 0.60)
NEGATIVE_SENTIMENT_THRESHOLD = _get_float("NEGATIVE_SENTIMENT_THRESHOLD", -0.25)
HIGH_RISK_THRESHOLD = _get_float("HIGH_RISK_THRESHOLD", 1.0)
