import json

import pytest

import cli
from conftest import FakeResponse, FakeSession, listing, raw_post
from subsense_engine import ModClassifier, TrendEngine, empty_frame, run_pipeline


@pytest.fixture
def processed(posts_df):
    return run_pipeline(posts_df, trend_engine=TrendEngine(), mod_classifier=ModClassifier(),
                        score_virality=False)


def test_parse_args_defaults():
    args = cli.parse_args(["--subreddits", "devops"])
    assert args.time == "month"
    assert args.limit == 100
    assert args.ask is None


def test_parse_args_rejects_bad_time_window():
    with pytest.raises(SystemExit):
        cli.parse_args(["--subreddits", "devops", "--time", "fortnight"])


def test_build_report_sections(processed):
    report = cli.build_report(processed, top=3)
    assert set(report) == {"summary", "top_topics", "mod_queue", "top_posts"}
    assert report["summary"]["total_posts"] == 20
    assert len(report["top_topics"]) <= 3
    assert len(report["top_posts"]) == 3
    assert report["top_posts"][0]["score"] >= report["top_posts"][-1]["score"]


def test_build_report_is_json_serialisable(processed):
    report = cli.build_report(processed, top=3)
    assert json.loads(json.dumps(report))["summary"]["total_posts"] == 20


def test_build_report_on_empty_dataset():
    report = cli.build_report(empty_frame(), top=5)
    assert report["summary"]["total_posts"] == 0
    assert report["mod_queue"] == []
    assert report["top_posts"] == []


def test_print_report_does_not_crash(processed, capsys):
    cli.print_report(cli.build_report(processed, top=3))
    out = capsys.readouterr().out
    assert "SubSense Report" in out
    assert "Trending topics" in out


def test_main_runs_end_to_end_and_writes_json(monkeypatch, tmp_path, capsys):
    posts = [raw_post(f"p{i}", f"post about hiring engineers {i}", score=i * 5) for i in range(12)]
    session = FakeSession([FakeResponse(listing(posts, after=None))])
    # Swap in a fetcher backed by a fake session so no network is touched.
    monkeypatch.setattr(cli, "RedditFetcher", lambda: _fetcher_with(session))

    out_path = tmp_path / "nested" / "report.json"
    code = cli.main(["--subreddits", "synthetic", "--limit", "12", "--no-train",
                     "--output", str(out_path), "--quiet"])
    assert code == 0
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["summary"]["total_posts"] == 12
    assert "SubSense Report" in capsys.readouterr().out


def _fetcher_with(session):
    from subsense_engine import RedditFetcher
    return RedditFetcher(session=session, base_url="https://example.test", delay=0,
                         max_retries=1, client_id="", client_secret="")


def test_main_explains_an_empty_result(monkeypatch, capsys):
    session = FakeSession([FakeResponse(listing([], after=None))])
    monkeypatch.setattr(cli, "RedditFetcher", lambda: _fetcher_with(session))
    assert cli.main(["--subreddits", "synthetic", "--quiet"]) == 1
    assert "REDDIT_CLIENT_ID" in capsys.readouterr().err


def test_main_rejects_blank_subreddit_list(capsys):
    assert cli.main(["--subreddits", " , ,"]) == 2
    assert "No subreddits given" in capsys.readouterr().err


def test_ask_without_api_key_exits_cleanly(monkeypatch, processed, capsys):
    monkeypatch.setattr(cli.config, "GEMINI_API_KEY", "")
    assert cli.answer_question(processed, "what is trending?", reindex=False) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
