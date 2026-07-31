"""Headless entry point for SubSense.

Runs the same intelligence pipeline the Streamlit dashboard uses, but from a
terminal or a cron job:

    python cli.py --subreddits startups,MachineLearning --time week --limit 100
    python cli.py --subreddits devops --output reports/devops.json
    python cli.py --subreddits devops --ask "what are people complaining about?"
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import pandas as pd

import config
from subsense_engine import (
    ModClassifier,
    RAGEngine,
    RedditFetcher,
    TrendEngine,
    ViralityPredictor,
    run_pipeline,
    summarize,
)

logger = logging.getLogger("subsense.cli")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="subsense",
        description="Analyse Reddit communities: topics, mod risk, virality and RAG chat.",
    )
    parser.add_argument("--subreddits", "-s", required=True,
                        help="Comma separated subreddits, e.g. 'startups,r/devops'.")
    parser.add_argument("--time", "-t", default="month", choices=list(config.TIME_FILTERS),
                        help="Reddit 'top' time window (default: month).")
    parser.add_argument("--limit", "-l", type=int, default=100,
                        help="Maximum posts to pull per subreddit (default: 100).")
    parser.add_argument("--top", type=int, default=5,
                        help="How many topics and risky posts to print (default: 5).")
    parser.add_argument("--output", "-o", default=None,
                        help="Write the full JSON report to this path.")
    parser.add_argument("--ask", default=None,
                        help="Ask a question about the fetched posts. Requires GEMINI_API_KEY.")
    parser.add_argument("--reindex", action="store_true",
                        help="Force the RAG knowledge base to be rebuilt before asking.")
    parser.add_argument("--no-train", action="store_true",
                        help="Skip virality training/scoring (faster, no model write).")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Only emit warnings and errors from the engine.")
    return parser.parse_args(argv)


def build_report(df: pd.DataFrame, top: int) -> Dict[str, Any]:
    summary = summarize(df)
    report: Dict[str, Any] = {"summary": summary}
    report["top_topics"] = summary.pop("top_topics")[:top]

    if df.empty:
        report["mod_queue"] = []
        report["top_posts"] = []
        return report

    risky = ModClassifier.high_risk(df).sort_values("mod_risk_score", ascending=False).head(top)
    report["mod_queue"] = [
        {
            "title": row["title"],
            "subreddit": row["subreddit"],
            "risk": float(row["mod_risk_score"]),
            "reasons": row["risk_reasons"],
            "url": row["url"],
        }
        for _, row in risky.iterrows()
    ]

    top_posts = df.sort_values("score", ascending=False).head(top)
    report["top_posts"] = [
        {
            "title": row["title"],
            "subreddit": row["subreddit"],
            "score": int(row["score"]),
            "comments": int(row["num_comments"]),
            "url": row["url"],
        }
        for _, row in top_posts.iterrows()
    ]
    return report


def _print_table(title: str, rows: List[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("  (none)")
        return
    for row in rows:
        print(f"  {row}")


def print_report(report: Dict[str, Any]) -> None:
    s = report["summary"]
    print("\n=== SubSense Report ===")
    print(f"  Posts analysed : {s['total_posts']}")
    print(f"  Communities    : {', '.join(s['subreddits']) or 'n/a'}")
    print(f"  Avg / median score : {s['avg_score']} / {s['median_score']}")
    print(f"  Total comments : {s['total_comments']}")
    print(f"  Media posts    : {s['media_posts']}")
    print(f"  High-risk posts: {s['high_risk_posts']}")
    print(f"  Avg sentiment  : {s['avg_sentiment']}")
    if s["busiest_hour"] is not None:
        print(f"  Busiest hour   : {s['busiest_hour']:02d}:00 UTC")

    _print_table("Trending topics", [
        f"{t['posts']:>3} posts | avg {t['avg_score']:>7} | {t['topic_keywords']}"
        for t in report["top_topics"]
    ])
    _print_table("Mod queue (highest risk)", [
        f"[{p['risk']:.1f}] r/{p['subreddit']} | {p['title'][:70]} ({p['reasons']})"
        for p in report["mod_queue"]
    ])
    _print_table("Top posts", [
        f"{p['score']:>6} pts | {p['comments']:>4} comments | r/{p['subreddit']} | {p['title'][:70]}"
        for p in report["top_posts"]
    ])


def answer_question(df: pd.DataFrame, question: str, reindex: bool) -> int:
    if not config.GEMINI_API_KEY:
        print("--ask needs GEMINI_API_KEY. Copy .env.example to .env and set it.", file=sys.stderr)
        return 2
    try:
        rag = RAGEngine(config.GEMINI_API_KEY)
    except Exception as e:
        print(f"Could not start the RAG engine: {e}", file=sys.stderr)
        return 2

    if not rag.index_data(df, force_reindex=reindex):
        print(f"Indexing failed: {rag.last_error}", file=sys.stderr)
        return 1

    answer, context = rag.query(question)
    print(f"\n=== Answer ===\n{answer}")
    if context:
        print(f"\n=== Sources ===\n{context}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(logging.WARNING if args.quiet else logging.INFO)

    subs = [s for s in (part.strip() for part in args.subreddits.split(",")) if s]
    if not subs:
        print("No subreddits given.", file=sys.stderr)
        return 2

    fetcher = RedditFetcher()
    df = fetcher.fetch_data(subs, time_filter=args.time, limit=args.limit)
    if df.empty:
        print(fetcher.last_error or
              "No posts fetched. Check the subreddit names and your connection.",
              file=sys.stderr)
        return 1

    df = run_pipeline(
        df,
        trend_engine=TrendEngine(),
        predictor=None if args.no_train else ViralityPredictor(),
        mod_classifier=ModClassifier(),
        score_virality=not args.no_train,
    )

    report = build_report(df, args.top)
    print_report(report)

    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")

    if args.ask:
        return answer_question(df, args.ask, args.reindex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
