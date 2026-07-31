"""Shared synthetic fixtures. No network, no API key, no real Reddit data."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subsense_engine import coerce_schema  # noqa: E402

import pandas as pd  # noqa: E402


def raw_post(post_id, title, **overrides):
    """A raw Reddit JSON payload with sensible defaults, mirroring the API shape."""
    post = {
        "id": post_id,
        "subreddit": "synthetic",
        "title": title,
        "selftext": "",
        "author": "test_user",
        "created_utc": 1_700_000_000,
        "link_flair_text": None,
        "ups": 10,
        "score": 10,
        "num_comments": 2,
        "upvote_ratio": 0.95,
        "is_self": True,
        "stickied": False,
        "locked": False,
        "over_18": False,
        "url": "https://reddit.com/r/synthetic/comments/x",
    }
    post.update(overrides)
    return post


def listing(children, after=None):
    """Wrap raw posts in Reddit's listing envelope."""
    return {"data": {"after": after, "children": [{"kind": "t3", "data": c} for c in children]}}


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Serves a queued list of responses and records what was requested."""

    def __init__(self, responses, token_responses=None):
        self.responses = list(responses)
        self.token_responses = list(token_responses or [])
        self.requests = []
        self.headers_seen = []
        self.token_requests = []

    def get(self, url, headers=None, timeout=None):
        self.requests.append(url)
        self.headers_seen.append(dict(headers or {}))
        if not self.responses:
            return FakeResponse({"data": {"after": None, "children": []}})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def post(self, url, auth=None, data=None, headers=None, timeout=None):
        self.token_requests.append({"url": url, "auth": auth, "data": data})
        if not self.token_responses:
            return FakeResponse({"access_token": "tok", "expires_in": 3600})
        item = self.token_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def posts_df():
    """A 20-row dataset with two clearly separable topics and known risk signals."""
    rows = []
    topics = [
        ("hiring engineers for our startup team", "we are hiring backend engineers"),
        ("machine learning model training tips", "how to train a neural network faster"),
    ]
    for i in range(20):
        title, body = topics[i % 2]
        rows.append({
            "post_id": f"p{i:02d}",
            "subreddit": "synthetic" if i % 2 == 0 else "synthetic2",
            "title": f"{title} {i}",
            "body": body,
            "author": f"user{i}",
            "created_utc": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
            "flair": "",
            "ups": 10 * i,
            "score": 10 * i,
            "num_comments": i,
            "upvote_ratio": 0.95,
            "is_text": True,
            "media_type": "text" if i % 3 else "image",
            "is_stickied": False,
            "is_locked": False,
            "is_nsfw": False,
            "url": f"https://reddit.com/p{i}",
        })
    return coerce_schema(pd.DataFrame(rows))


class FakeGeminiClient:
    """Deterministic stand-in for GeminiClient.

    Embeds text as a bag-of-characters vector, which is enough for retrieval
    ordering to be meaningful and stable without any network call.
    """

    DIM = 26

    def __init__(self):
        self.embed_calls = []
        self.prompts = []

    def embed(self, texts, task_type="retrieval_document"):
        self.embed_calls.append((list(texts), task_type))
        vectors = []
        for text in texts:
            vec = [0.0] * self.DIM
            for ch in str(text).lower():
                if 'a' <= ch <= 'z':
                    vec[ord(ch) - 97] += 1.0
            if not any(vec):
                vec[0] = 1.0
            vectors.append(vec)
        return vectors

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "synthetic answer"


@pytest.fixture
def fake_client():
    return FakeGeminiClient()
