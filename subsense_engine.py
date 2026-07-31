"""SubSense intelligence engine.

Pure-python, UI-free layers that the Streamlit app and the CLI both sit on top
of. Every layer accepts and returns a pandas DataFrame following the universal
schema in `POST_COLUMNS`, so the layers can be composed in any order and tested
without a network or an API key.
"""

import os
import time
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional accelerator. Absence is not an error: the vector search falls back
# to numpy, which is plenty fast for the dataset sizes this app handles.
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError as e:  # pragma: no cover - depends on the install
    logger.warning(f"FAISS not available ({e}). Switching to Numpy for Vector Search.")
    faiss = None


# --- Universal schema ------------------------------------------------------
# Declared once so an empty result set still produces a well-formed DataFrame
# instead of a column-less one that blows up every downstream layer.
POST_COLUMNS: Dict[str, Any] = {
    "post_id": "object",
    "subreddit": "object",
    "title": "object",
    "body": "object",
    "author": "object",
    "created_utc": "datetime64[ns]",
    "flair": "object",
    "ups": "int64",
    "score": "int64",
    "num_comments": "int64",
    "upvote_ratio": "float64",
    "is_text": "bool",
    "media_type": "object",
    "is_stickied": "bool",
    "is_locked": "bool",
    "is_nsfw": "bool",
    "url": "object",
}

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.webp')


def empty_frame() -> pd.DataFrame:
    """An empty DataFrame that still carries the full schema and dtypes."""
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in POST_COLUMNS.items()})


def _to_utc_datetime(epoch: Any) -> Optional[datetime]:
    """Reddit hands out UTC epochs; keep them UTC rather than local time."""
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).replace(tzinfo=None)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def detect_media_type(post: Dict[str, Any]) -> str:
    """Classify a raw Reddit post into text / image / video / gallery / link."""
    if post.get('is_video'):
        return 'video'
    if post.get('gallery_data') or post.get('is_gallery'):
        return 'gallery'

    url = (post.get('url') or '').lower().split('?')[0]
    if url.endswith(IMAGE_EXTENSIONS):
        return 'image'

    hint = post.get('post_hint') or ''
    if hint == 'image':
        return 'image'
    if hint in ('hosted:video', 'rich:video'):
        return 'video'

    if post.get('is_self', True):
        return 'text'
    return 'link'


class RedditFetcher:
    """Fetches posts from Reddit's public JSON endpoints (no PRAW / OAuth needed)."""

    def __init__(self, session: Optional[requests.Session] = None,
                 base_url: str = config.REDDIT_BASE_URL,
                 user_agent: str = config.REDDIT_USER_AGENT,
                 timeout: float = config.REQUEST_TIMEOUT,
                 delay: float = config.REQUEST_DELAY,
                 max_retries: int = config.MAX_RETRIES):
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip('/')
        self.headers = {'User-Agent': user_agent}
        self.timeout = timeout
        self.delay = delay
        self.max_retries = max_retries
        logger.info("RedditFetcher initialized.")

    def fetch_data(self, subreddits: Sequence[str], time_filter: str = 'month',
                   limit: int = 100) -> pd.DataFrame:
        if time_filter not in config.TIME_FILTERS:
            raise ValueError(
                f"Unsupported time_filter {time_filter!r}. Expected one of {config.TIME_FILTERS}."
            )

        names = [s.strip().lstrip('/').removeprefix('r/').strip() for s in subreddits]
        names = [n for n in names if n]
        if not names:
            logger.warning("No valid subreddit names supplied.")
            return empty_frame()

        all_data: List[Dict[str, Any]] = []
        logger.info(f"Fetching data for subreddits: {names}, Time: {time_filter}, Limit: {limit}")

        for sub in names:
            url = f"{self.base_url}/r/{sub}/top/.json?t={time_filter}"
            try:
                results = self._fetch_paginated(url, limit)
                for post in results:
                    post['source_subreddit'] = sub
                all_data.extend(results)
                logger.info(f"Successfully fetched {len(results)} posts from r/{sub}")
            except Exception as e:
                logger.error(f"Error fetching r/{sub}: {e}")

        df = self._normalize(all_data)
        logger.info(f"Total normalized records: {len(df)}")
        return df

    def _get(self, url: str) -> Optional[Dict[str, Any]]:
        """One listing request, with backoff on throttling and transient errors."""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, headers=self.headers, timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning(f"Request to {url} failed ({e}); attempt {attempt}/{self.max_retries}")
                if attempt == self.max_retries:
                    return None
                time.sleep(self.delay * attempt)
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as e:
                    logger.error(f"Non-JSON response from {url}: {e}")
                    return None

            if resp.status_code in (429, 500, 502, 503, 504):
                wait = self.delay * attempt * 2
                logger.warning(
                    f"Throttled/unavailable ({resp.status_code}) on {url}; "
                    f"retrying in {wait:.1f}s (attempt {attempt}/{self.max_retries})"
                )
                if attempt == self.max_retries:
                    return None
                time.sleep(wait)
                continue

            logger.warning(f"Failed request to {url}: HTTP {resp.status_code}")
            return None
        return None

    def _fetch_paginated(self, base_url: str, limit: int) -> List[Dict[str, Any]]:
        children: List[Dict[str, Any]] = []
        after: Optional[str] = None
        seen_cursors = set()

        while len(children) < limit:
            page_size = min(config.PAGE_SIZE, limit - len(children))
            req_url = f"{base_url}&limit={page_size}"
            if after:
                req_url = f"{req_url}&after={after}"

            data = self._get(req_url)
            if not data:
                break

            listing = data.get('data') or {}
            new_children = [c['data'] for c in listing.get('children', []) if 'data' in c]
            if not new_children:
                break
            children.extend(new_children)

            after = listing.get('after')
            # Reddit occasionally echoes the same cursor forever; without this
            # guard the loop only ends when `limit` is reached, re-fetching the
            # same page over and over.
            if not after or after in seen_cursors:
                break
            seen_cursors.add(after)

            if self.delay:
                time.sleep(self.delay)

        return children[:limit]

    def _normalize(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert raw Reddit JSON into the universal schema."""
        if not raw_data:
            return empty_frame()

        normalized = []
        for p in raw_data:
            created = _to_utc_datetime(p.get('created_utc'))
            normalized.append({
                "post_id": p.get('id'),
                "subreddit": p.get('source_subreddit') or p.get('subreddit'),
                "title": p.get('title') or '',
                "body": p.get('selftext') or '',
                "author": p.get('author') or '',
                "created_utc": created,
                "flair": p.get('link_flair_text') or '',
                "ups": p.get('ups') or 0,
                "score": p.get('score') or 0,
                "num_comments": p.get('num_comments') or 0,
                "upvote_ratio": p.get('upvote_ratio') if p.get('upvote_ratio') is not None else 1.0,
                "is_text": bool(p.get('is_self', True)),
                "media_type": detect_media_type(p),
                "is_stickied": bool(p.get('stickied', False)),
                "is_locked": bool(p.get('locked', False)),
                "is_nsfw": bool(p.get('over_18', False)),
                "url": p.get('url') or '',
            })

        df = pd.DataFrame(normalized, columns=list(POST_COLUMNS))
        # Duplicate posts show up when several subreddits are crossposted to.
        df = df.drop_duplicates(subset='post_id', keep='first').reset_index(drop=True)
        return coerce_schema(df)


def column_or_default(df: pd.DataFrame, name: str, default: Any) -> pd.Series:
    """A column as a Series, substituting a constant when it is missing or null."""
    if name in df.columns:
        return df[name].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Force the universal dtypes so later layers never guess at a column's type."""
    df = df.copy()
    for name, dtype in POST_COLUMNS.items():
        if name not in df.columns:
            df[name] = pd.Series([None] * len(df))
        if dtype == "datetime64[ns]":
            df[name] = pd.to_datetime(df[name], errors='coerce')
        elif dtype == "int64":
            df[name] = pd.to_numeric(df[name], errors='coerce').fillna(0).astype('int64')
        elif dtype == "float64":
            df[name] = pd.to_numeric(df[name], errors='coerce').fillna(0.0).astype('float64')
        elif dtype == "bool":
            df[name] = df[name].apply(lambda v: bool(v) if pd.notna(v) else False).astype('bool')
        else:
            df[name] = df[name].fillna('').astype('object')
    return df


class TrendEngine:
    """Unsupervised topic modelling: TF-IDF vectors clustered with K-Means."""

    def __init__(self, max_clusters: int = config.MAX_CLUSTERS,
                 posts_per_cluster: int = config.POSTS_PER_CLUSTER,
                 random_state: int = config.RANDOM_SEED):
        self.max_clusters = max_clusters
        self.posts_per_cluster = posts_per_cluster
        self.random_state = random_state
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_names: Dict[int, str] = {}
        logger.info("TrendEngine initialized.")

    def _cluster_count(self, n_docs: int) -> int:
        return max(1, min(self.max_clusters, n_docs // self.posts_per_cluster))

    def extract_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            logger.warning("Empty DataFrame passed to TrendEngine.")
            df['topic_cluster'] = pd.Series(dtype='int64')
            df['topic_keywords'] = pd.Series(dtype='object')
            return df

        logger.info("Extracting trends from dataset.")
        text_data = (df['title'].fillna('') + " " + df['body'].fillna('')).str.strip()
        # A fresh vectorizer per call: reusing a fitted one across datasets
        # would silently score new posts against a stale vocabulary.
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(text_data)
            if tfidf_matrix.shape[1] == 0:
                raise ValueError("Empty vocabulary after stop-word removal.")

            num_clusters = min(self._cluster_count(len(df)), tfidf_matrix.shape[0])
            # random_state + explicit n_init keep the same corpus mapping to the
            # same topics between runs, which matters for a persisted dashboard.
            self.kmeans = KMeans(n_clusters=num_clusters, random_state=self.random_state, n_init=10)
            df['topic_cluster'] = self.kmeans.fit_predict(tfidf_matrix).astype('int64')

            feature_names = self.vectorizer.get_feature_names_out()
            self.cluster_names = {}
            for i in range(num_clusters):
                center = self.kmeans.cluster_centers_[i]
                top_ind = center.argsort()[:-6:-1]
                self.cluster_names[i] = ", ".join(feature_names[ind] for ind in top_ind)

            df['topic_keywords'] = df['topic_cluster'].map(self.cluster_names).fillna("General")
            logger.info(f"Trend extraction complete ({num_clusters} clusters).")
        except Exception as e:
            logger.error(f"Trend extraction failed: {e}")
            df['topic_cluster'] = 0
            df['topic_keywords'] = "General"
        return df

    @staticmethod
    def top_topics(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Topic leaderboard ordered by volume, with engagement alongside."""
        if df.empty or 'topic_keywords' not in df.columns:
            return pd.DataFrame(columns=['topic_keywords', 'posts', 'avg_score', 'total_comments'])
        summary = (
            df.groupby('topic_keywords')
              .agg(posts=('post_id', 'count'),
                   avg_score=('score', 'mean'),
                   total_comments=('num_comments', 'sum'))
              .reset_index()
              .sort_values('posts', ascending=False)
        )
        summary['avg_score'] = summary['avg_score'].round(1)
        return summary.head(top_n).reset_index(drop=True)


class ViralityPredictor:
    """XGBoost regressor over cheap, pre-publication post features, persisted to disk."""

    FEATURES = ['title_len', 'word_count', 'has_media', 'hour', 'day_of_week', 'is_text']

    def __init__(self, model_path: str = config.VIRALITY_MODEL_PATH):
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50,
                                      random_state=config.RANDOM_SEED)
        self.model_path = model_path
        self.is_trained = False
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            return
        try:
            self.model.load_model(self.model_path)
            booster = self.model.get_booster()
            saved_features = list(booster.feature_names or [])
            if saved_features and saved_features != self.FEATURES:
                # An older model trained on a different feature set would raise
                # on every predict; treat it as untrained instead of half-broken.
                logger.warning(
                    f"Persisted model expects features {saved_features}, engine uses "
                    f"{self.FEATURES}. Ignoring stale model; retrain to refresh it."
                )
                self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50,
                                              random_state=config.RANDOM_SEED)
                return
            self.is_trained = True
            logger.info(f"Loaded Virality Model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load virality model: {e}")

    def save_model(self) -> None:
        try:
            parent = os.path.dirname(os.path.abspath(self.model_path))
            os.makedirs(parent, exist_ok=True)
            self.model.save_model(self.model_path)
            logger.info(f"Saved Virality Model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save virality model: {e}")

    @classmethod
    def build_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Feature matrix built the same way for training and inference."""
        created = pd.to_datetime(column_or_default(df, 'created_utc', pd.NaT), errors='coerce')
        titles = column_or_default(df, 'title', '').astype(str)
        media = column_or_default(df, 'media_type', 'text').astype(str)
        features = pd.DataFrame({
            'title_len': titles.str.len().astype('int64'),
            'word_count': titles.str.split().apply(len).astype('int64'),
            'has_media': (media != 'text').astype('int64'),
            'hour': created.dt.hour.fillna(12).astype('int64'),
            'day_of_week': created.dt.dayofweek.fillna(0).astype('int64'),
            'is_text': (media == 'text').astype('int64'),
        }, index=df.index)
        return features[cls.FEATURES]

    def train_and_score(self, df: pd.DataFrame, force_retrain: bool = False,
                        min_rows: int = 10) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            df['predicted_score'] = pd.Series(dtype='float64')
            return df

        X = self.build_features(df)

        if self.is_trained and not force_retrain:
            try:
                df['predicted_score'] = self.model.predict(X)
                logger.info("Used persisted model for scoring.")
                return df
            except Exception as e:
                logger.warning(f"Persisted model failed to predict ({e}). Retraining...")

        if len(df) < min_rows:
            logger.warning(
                f"Not enough data to train ViralityPredictor ({len(df)} < {min_rows} rows)."
            )
            df['predicted_score'] = np.nan
            return df

        try:
            logger.info(f"Training Virality model on {len(df)} posts.")
            y = pd.to_numeric(df['score'], errors='coerce').fillna(0)
            self.model.fit(X, y)
            self.is_trained = True
            df['predicted_score'] = self.model.predict(X)
            self.save_model()
        except Exception as e:
            logger.error(f"Virality model training failed: {e}")
            df['predicted_score'] = np.nan

        return df

    def predict_new(self, title: str, media_type: str, hour: int,
                    day_of_week: int = 0) -> float:
        """Score a hypothetical post. Returns 0.0 when no model is available."""
        if not self.is_trained:
            return 0.0
        row = pd.DataFrame([{
            'title': title or '',
            'media_type': media_type or 'text',
            'created_utc': pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(day_of_week) % 7,
                                                                     hours=int(hour) % 24),
        }])
        try:
            return float(self.model.predict(self.build_features(row))[0])
        except Exception as e:
            logger.error(f"Virality prediction failed: {e}")
            return 0.0


class ModClassifier:
    """Heuristic + sentiment signals for how likely a post is to draw mod action."""

    def __init__(self, locked_weight: float = config.RISK_LOCKED_WEIGHT,
                 low_ratio_weight: float = config.RISK_LOW_RATIO_WEIGHT,
                 sentiment_weight: float = config.RISK_SENTIMENT_WEIGHT):
        self.locked_weight = locked_weight
        self.low_ratio_weight = low_ratio_weight
        self.sentiment_weight = sentiment_weight
        logger.info("ModClassifier initialized.")

    @staticmethod
    def _polarity(text: str) -> float:
        try:
            return float(TextBlob(str(text)).sentiment.polarity)
        except Exception:
            return 0.0

    def score_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.empty:
            df['sentiment'] = pd.Series(dtype='float64')
            df['mod_risk_score'] = pd.Series(dtype='float64')
            df['risk_reasons'] = pd.Series(dtype='object')
            return df

        logger.info("Calculating Mod Risk Scores.")
        locked = column_or_default(df, 'is_locked', False).astype(bool)
        stickied = column_or_default(df, 'is_stickied', False).astype(bool)
        ratio = pd.to_numeric(column_or_default(df, 'upvote_ratio', 1.0), errors='coerce').fillna(1.0)

        try:
            df['sentiment'] = column_or_default(df, 'title', '').astype(str).apply(self._polarity)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            df['sentiment'] = 0.0

        # A stickied lock is a mod announcement, not a moderated post.
        locked_flag = locked & ~stickied
        low_ratio_flag = ratio < config.LOW_RATIO_THRESHOLD
        negative_flag = df['sentiment'] < config.NEGATIVE_SENTIMENT_THRESHOLD

        score = (
            locked_flag.astype(float) * self.locked_weight
            + low_ratio_flag.astype(float) * self.low_ratio_weight
            + negative_flag.astype(float) * self.sentiment_weight
        )
        df['mod_risk_score'] = score.round(2)

        signals = (
            (locked_flag, "locked thread"),
            (low_ratio_flag, "low upvote ratio"),
            (negative_flag, "negative sentiment"),
        )
        df['risk_reasons'] = [
            ", ".join(label for flag, label in signals if bool(flag.iloc[i])) or "none"
            for i in range(len(df))
        ]
        return df

    @staticmethod
    def high_risk(df: pd.DataFrame, threshold: float = config.HIGH_RISK_THRESHOLD) -> pd.DataFrame:
        if df.empty or 'mod_risk_score' not in df.columns:
            return df.iloc[0:0]
        return df[df['mod_risk_score'] >= threshold]


class GeminiClient:
    """Thin adapter over google-generativeai.

    Isolating the SDK behind two methods keeps `RAGEngine` testable: tests pass
    a stub with the same shape and never touch the network.
    """

    def __init__(self, api_key: str,
                 chat_model: str = config.GEMINI_CHAT_MODEL,
                 embed_model: str = config.GEMINI_EMBED_MODEL):
        if not api_key:
            raise ValueError("A Gemini API key is required. Set GEMINI_API_KEY in your .env.")
        import google.generativeai as genai  # imported lazily so the engine loads without it

        self._genai = genai
        self._genai.configure(api_key=api_key)
        self.embed_model = embed_model
        self.model = self._genai.GenerativeModel(chat_model)

    def embed(self, texts: Sequence[str], task_type: str = "retrieval_document") -> List[List[float]]:
        result = self._genai.embed_content(
            model=self.embed_model,
            content=list(texts),
            task_type=task_type,
        )
        embedding = result.get('embedding') if isinstance(result, dict) else None
        if embedding is None:
            return []
        # The SDK returns a flat vector for a single string, a list of vectors
        # for a batch. Normalise to "list of vectors" either way.
        if embedding and not isinstance(embedding[0], (list, tuple, np.ndarray)):
            return [list(embedding)]
        return [list(vec) for vec in embedding]

    def generate(self, prompt: str) -> str:
        return self.model.generate_content(prompt).text


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class RAGEngine:
    """Retrieval-augmented chat over the fetched posts, persisted to disk.

    Embeddings are L2-normalised, so both the FAISS inner-product index and the
    numpy fallback rank by cosine similarity and return identical neighbours.
    """

    DOC_FIELDS = ('post_id', 'subreddit', 'title', 'body', 'score', 'num_comments', 'url')

    def __init__(self, api_key: Optional[str] = None, persist_dir: str = config.RAG_STORE_DIR,
                 client: Optional[Any] = None):
        self.index = None
        self.docs: List[Dict[str, Any]] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.persist_dir = persist_dir
        self.last_error: Optional[str] = None
        self.fingerprint: Optional[str] = None
        self.client = client or GeminiClient(api_key or config.GEMINI_API_KEY)
        self._load_index()

    # --- persistence -------------------------------------------------------
    @property
    def _docs_file(self) -> str:
        return os.path.join(self.persist_dir, "docs.json")

    @property
    def _index_file(self) -> str:
        return os.path.join(self.persist_dir, "index.faiss")

    @property
    def _embed_file(self) -> str:
        return os.path.join(self.persist_dir, "embeddings.npy")

    def _load_index(self) -> None:
        if not os.path.isdir(self.persist_dir):
            os.makedirs(self.persist_dir, exist_ok=True)
            return
        try:
            if os.path.exists(self._docs_file):
                with open(self._docs_file, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                # Older stores were a bare list; newer ones carry a fingerprint.
                if isinstance(payload, dict):
                    self.docs = payload.get('docs', [])
                    self.fingerprint = payload.get('fingerprint')
                else:
                    self.docs = payload
                logger.info(f"Loaded {len(self.docs)} docs from disk.")

            if os.path.exists(self._embed_file):
                self.doc_embeddings = np.load(self._embed_file)
                logger.info("Loaded Numpy embeddings from disk.")

            if faiss and os.path.exists(self._index_file):
                self.index = faiss.read_index(self._index_file)
                logger.info("Loaded FAISS index from disk.")
            elif faiss and self.doc_embeddings is not None:
                self._build_faiss(self.doc_embeddings)
        except Exception as e:
            logger.error(f"Failed to load persisted RAG index: {e}")
            self.docs, self.index, self.doc_embeddings = [], None, None

    def save_index(self) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        try:
            with open(self._docs_file, 'w', encoding='utf-8') as f:
                json.dump({'fingerprint': self.fingerprint, 'docs': self.docs}, f)
            # Always persist the raw matrix: it is the portable form, and the
            # FAISS index can be rebuilt from it if the accelerator disappears.
            if self.doc_embeddings is not None:
                np.save(self._embed_file, self.doc_embeddings)
            if self.index is not None and faiss:
                faiss.write_index(self.index, self._index_file)
            logger.info("Successfully persisted RAG index.")
        except Exception as e:
            logger.error(f"Failed to save RAG index: {e}")

    # --- indexing ----------------------------------------------------------
    @classmethod
    def _to_documents(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Project posts to a small, JSON-serialisable record.

        `df.to_dict('records')` leaks Timestamps and numpy scalars, which makes
        `json.dump` raise and silently loses the whole store.
        """
        docs = []
        for _, row in df.iterrows():
            doc = {}
            for field in cls.DOC_FIELDS:
                value = row.get(field)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = '' if field not in ('score', 'num_comments') else 0
                elif isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value)
                elif isinstance(value, pd.Timestamp):
                    value = value.isoformat()
                else:
                    value = value if isinstance(value, (int, float, str)) else str(value)
                doc[field] = value
            created = row.get('created_utc')
            doc['created_utc'] = created.isoformat() if isinstance(created, pd.Timestamp) and pd.notna(created) else ''
            docs.append(doc)
        return docs

    @staticmethod
    def _fingerprint(docs: List[Dict[str, Any]]) -> str:
        """Identity of a document set, so a new corpus triggers a re-index."""
        digest = hashlib.sha256()
        for doc in docs:
            digest.update(str(doc.get('post_id', '')).encode('utf-8'))
            digest.update(b'\x00')
        return digest.hexdigest()

    @staticmethod
    def _doc_text(doc: Dict[str, Any]) -> str:
        return f"Sub: {doc.get('subreddit', '')} | Title: {doc.get('title', '')} | Body: {str(doc.get('body', ''))[:500]}"

    def _build_faiss(self, matrix: np.ndarray) -> None:
        if not faiss:
            return
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def is_indexed(self) -> bool:
        return self.doc_embeddings is not None or self.index is not None

    def index_data(self, df: pd.DataFrame, force_reindex: bool = False,
                   batch_size: int = config.EMBED_BATCH_SIZE,
                   batch_delay: float = config.EMBED_BATCH_DELAY) -> bool:
        """Embed and index the posts. Returns True when the store is usable."""
        self.last_error = None
        if df.empty:
            self.last_error = "Nothing to index: the dataset is empty."
            logger.warning(self.last_error)
            return False

        docs = self._to_documents(df)
        fingerprint = self._fingerprint(docs)

        # Skip only when the *same* corpus is already indexed. The old check
        # skipped whenever any index existed, so switching subreddits kept
        # answering from the previous run's data.
        if self.is_indexed() and not force_reindex and fingerprint == self.fingerprint:
            logger.info("RAG Index already covers this dataset. Skipping re-indexing.")
            return True

        logger.info(f"Indexing {len(docs)} documents for RAG.")
        texts = [self._doc_text(d) for d in docs]
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                vectors = self.client.embed(batch, task_type="retrieval_document")
            except Exception as e:
                self.last_error = f"Gemini Embedding batch error: {e}"
                logger.error(self.last_error)
                return False
            if len(vectors) != len(batch):
                self.last_error = (
                    f"Embedding count mismatch: asked for {len(batch)}, got {len(vectors)}."
                )
                logger.error(self.last_error)
                return False
            all_embeddings.extend(vectors)
            logger.info(f"Embedded batch {i // batch_size + 1}")
            if batch_delay and i + batch_size < len(texts):
                time.sleep(batch_delay)

        if not all_embeddings:
            self.last_error = "No embeddings generated (possibly API empty response)."
            return False

        try:
            matrix = _l2_normalize(np.asarray(all_embeddings, dtype='float32'))
            self.doc_embeddings = matrix
            self.docs = docs
            self.fingerprint = fingerprint
            self.index = None
            self._build_faiss(matrix)
            self.save_index()
            return True
        except Exception as e:
            self.last_error = f"Index build failed: {e}"
            logger.error(self.last_error)
            return False

    # --- retrieval ---------------------------------------------------------
    def retrieve(self, user_query: str, top_k: int = config.RAG_TOP_K) -> List[Dict[str, Any]]:
        """Nearest documents for a query. Raises if the store is not built."""
        if not self.is_indexed() or not self.docs:
            raise RuntimeError("Index not built.")

        vectors = self.client.embed([user_query], task_type="retrieval_query")
        if not vectors:
            raise RuntimeError("Query embedding returned no vector.")
        q_embed = _l2_normalize(np.asarray(vectors, dtype='float32'))

        top_k = max(1, min(top_k, len(self.docs)))
        if self.index is not None:
            _, indices = self.index.search(q_embed, top_k)
            order = indices[0]
        else:
            scores = self.doc_embeddings @ q_embed[0]
            order = np.argsort(scores)[::-1][:top_k]

        return [self.docs[int(i)] for i in order if 0 <= int(i) < len(self.docs)]

    @staticmethod
    def format_context(docs: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"- [r/{d.get('subreddit', '?')}] {d.get('title', '')} "
            f"(Score: {d.get('score', 0)}, Comments: {d.get('num_comments', 0)})"
            for d in docs
        )

    def query(self, user_query: str, top_k: int = config.RAG_TOP_K) -> Tuple[str, str]:
        """Answer a question from the indexed posts. Returns (answer, sources)."""
        if not self.is_indexed() or not self.docs:
            msg = "Index not built."
            if self.last_error:
                msg += f" (Last Error: {self.last_error})"
            return msg, ""

        logger.info(f"Processing RAG Query: {user_query}")
        try:
            docs = self.retrieve(user_query, top_k=top_k)
            context = self.format_context(docs)
            prompt = (
                "You are analysing Reddit community discussions.\n\n"
                f"Context from Subreddit Analysis:\n{context}\n\n"
                f"User Query: {user_query}\n\n"
                "Answer using only the context provided. Cite specific post titles when "
                "relevant. If the context does not cover the question, say so plainly."
            )
            return self.client.generate(prompt), context
        except Exception as e:
            logger.error(f"RAG Query failed: {e}")
            return f"Error: {e}", ""


def run_pipeline(df: pd.DataFrame, trend_engine: Optional[TrendEngine] = None,
                 predictor: Optional[ViralityPredictor] = None,
                 mod_classifier: Optional[ModClassifier] = None) -> pd.DataFrame:
    """Apply every offline intelligence layer in order.

    Shared by the Streamlit app and the CLI so both produce identical columns.
    """
    trend_engine = trend_engine or TrendEngine()
    predictor = predictor or ViralityPredictor()
    mod_classifier = mod_classifier or ModClassifier()

    df = trend_engine.extract_trends(df)
    df = mod_classifier.score_risk(df)
    df = predictor.train_and_score(df)
    return df


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    """Headline metrics for a processed dataset (dashboard + CLI report)."""
    if df.empty:
        return {
            "total_posts": 0, "subreddits": [], "avg_score": 0.0, "median_score": 0.0,
            "total_comments": 0, "media_posts": 0, "high_risk_posts": 0,
            "avg_sentiment": 0.0, "top_topics": [], "busiest_hour": None,
        }

    hours = pd.to_datetime(df['created_utc'], errors='coerce').dt.hour.dropna()
    topics = TrendEngine.top_topics(df) if 'topic_keywords' in df.columns else pd.DataFrame()
    return {
        "total_posts": int(len(df)),
        "subreddits": sorted(str(s) for s in df['subreddit'].dropna().unique()),
        "avg_score": round(float(df['score'].mean()), 1),
        "median_score": round(float(df['score'].median()), 1),
        "total_comments": int(df['num_comments'].sum()),
        "media_posts": int((df['media_type'] != 'text').sum()),
        "high_risk_posts": int(len(ModClassifier.high_risk(df))),
        "avg_sentiment": round(float(df['sentiment'].mean()), 3) if 'sentiment' in df.columns else 0.0,
        "top_topics": topics.to_dict('records') if not topics.empty else [],
        "busiest_hour": int(hours.mode().iloc[0]) if not hours.empty and not hours.mode().empty else None,
    }
