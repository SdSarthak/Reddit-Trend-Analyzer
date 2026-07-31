# SubSense — Reddit Trend Analyzer

Turns raw Reddit discussions into a readable intelligence report: what a
community is talking about, which posts are heading for the mod queue, what a
draft title is likely to score, and a citation-backed chat over the fetched
posts.

Runs two ways — a Streamlit dashboard, or a headless CLI that prints a report
and writes JSON.

```
python cli.py --subreddits startups,MachineLearning --time week --limit 100
streamlit run app.py
```

---

## What it does

| Layer | How | Output |
| --- | --- | --- |
| **Fetch** | Reddit application-only OAuth, cursor pagination, retry/backoff | Posts normalised to one schema (`POST_COLUMNS`) |
| **Topics** | TF-IDF + K-Means (seeded, so runs are reproducible) | `topic_cluster`, `topic_keywords` |
| **Mod risk** | Locked-thread / low-ratio / negative-sentiment heuristics | `mod_risk_score`, `risk_reasons`, `sentiment` |
| **Virality** | XGBoost regression on pre-publication features | `predicted_score`, plus a what-if sandbox |
| **Chat (RAG)** | Gemini embeddings → FAISS or numpy cosine search → Gemini answer | Answer + the posts it cited |

Every layer takes a DataFrame and returns a DataFrame, so they compose in any
order and are testable without a network connection.

---

## Setup

**1. Install**

```bash
pip install -r requirements.txt
```

Python 3.10+. `faiss-cpu` is optional — the vector search falls back to numpy
and returns identical rankings.

**2. Get Reddit credentials** (required)

Reddit returns **HTTP 403** for anonymous access to the `.json` endpoints, so a
free app registration is now mandatory. It takes about a minute:

1. Go to <https://www.reddit.com/prefs/apps> → **create another app**.
2. Choose type **script**. Any name; redirect URI `http://localhost:8080`.
3. The **client id** is the string under the app name; the **secret** is
   labelled `secret`.

No user login or password is involved — this is application-only OAuth, and it
only reads public listings.

**3. Get a Gemini key** (optional, for the chat tab)

<https://aistudio.google.com/apikey>. Everything except the chat layer works
without it.

**4. Configure**

```bash
cp .env.example .env
```

Then fill in:

```env
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
GEMINI_API_KEY=...
```

`.env` is gitignored. Every other setting in `.env.example` is optional and
documented inline (rate limits, model names, clustering size, risk weights).

---

## Usage

### CLI

```bash
# Report to the terminal
python cli.py --subreddits devops --time week --limit 100

# Save the full report as JSON
python cli.py --subreddits devops --output reports/devops.json

# Ask a question about what was just fetched (needs GEMINI_API_KEY)
python cli.py --subreddits devops --ask "what are people complaining about?"

# Skip the ML training step for a fast pass
python cli.py --subreddits devops --no-train --quiet
```

| Flag | Meaning |
| --- | --- |
| `--subreddits`, `-s` | Comma separated. `r/` prefixes are stripped for you. |
| `--time`, `-t` | `hour`, `day`, `week`, `month`, `year`, `all` (default `month`) |
| `--limit`, `-l` | Max posts per subreddit (default 100) |
| `--top` | Rows per section in the report (default 5) |
| `--output`, `-o` | Write the JSON report here |
| `--ask` | Ask a question; indexes the posts first |
| `--reindex` | Force the knowledge base to rebuild before asking |
| `--no-train` | Skip virality training/scoring |

Exit codes: `0` success, `1` fetch or indexing failure, `2` bad arguments or
missing credentials.

### Dashboard

```bash
streamlit run app.py
```

Four tabs: **Dashboard** (volume, topics, sentiment, posting hours), **Deep
Insights** (mod queue + virality sandbox), **Ask SubSense** (RAG chat), and
**MLOps** (retrain the model, rebuild the knowledge base).

### As a library

```python
from subsense_engine import RedditFetcher, run_pipeline, summarize

df = RedditFetcher().fetch_data(["startups"], time_filter="week", limit=100)
df = run_pipeline(df)
print(summarize(df))
```

---

## Persistence

Both artifacts are generated at runtime and are gitignored — nothing scraped is
committed to this repo.

- `virality_model.json` — the trained XGBoost model. Written automatically the
  first time you analyse 10+ posts, reloaded on the next start. If it was
  trained on a different feature set it is rejected with a warning rather than
  loaded, and retrains on the next run.
- `rag_store/` — `docs.json`, `embeddings.npy` and an optional FAISS index.
  Re-indexing is skipped when the same posts are already indexed (a fingerprint
  over post ids), so switching subreddits rebuilds but re-running the same query
  set does not burn API quota.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

82 tests, all offline: HTTP is served by a fake session, Gemini by a
deterministic stub embedder. No API key, no network and no real Reddit data is
needed to run them.

---

## Layout

```
app.py               Streamlit dashboard
cli.py               Headless entry point
config.py            Every tunable, read from the environment
subsense_engine.py   Fetcher, trend / mod / virality / RAG layers
tests/               Offline test suite with synthetic fixtures
```

## Notes and limits

- Reddit's `top` listings cap out around 1000 posts per subreddit; `--limit`
  above that silently plateaus.
- Sentiment is TextBlob's lexicon polarity — fast and dependency-light, but it
  does not understand sarcasm, which is a real limitation on Reddit.
- The virality model is trained on whatever you last fetched. A model trained on
  one community will not transfer well to another; retrain per corpus.
