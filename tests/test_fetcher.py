import pandas as pd
import pytest
import requests

from conftest import FakeResponse, FakeSession, listing, raw_post
from subsense_engine import POST_COLUMNS, RedditFetcher, detect_media_type, empty_frame


def make_fetcher(responses):
    session = FakeSession(responses)
    fetcher = RedditFetcher(session=session, base_url="https://example.test", delay=0, max_retries=2)
    return fetcher, session


def test_empty_frame_carries_full_schema():
    df = empty_frame()
    assert list(df.columns) == list(POST_COLUMNS)
    assert df.empty


def test_normalize_produces_schema_and_utc_times():
    fetcher, _ = make_fetcher([])
    df = fetcher._normalize([raw_post("a1", "hello world", created_utc=1_700_000_000)])
    assert list(df.columns) == list(POST_COLUMNS)
    assert df.loc[0, "created_utc"] == pd.Timestamp("2023-11-14 22:13:20")
    assert df.loc[0, "title"] == "hello world"
    assert df["is_locked"].dtype == bool
    assert df["score"].dtype == "int64"


def test_normalize_of_nothing_is_still_well_formed():
    fetcher, _ = make_fetcher([])
    df = fetcher._normalize([])
    assert df.empty
    assert list(df.columns) == list(POST_COLUMNS)


def test_normalize_drops_crossposted_duplicates():
    fetcher, _ = make_fetcher([])
    df = fetcher._normalize([raw_post("dup", "one"), raw_post("dup", "one again")])
    assert len(df) == 1
    assert df.loc[0, "title"] == "one"


@pytest.mark.parametrize("payload,expected", [
    ({"is_video": True}, "video"),
    ({"gallery_data": {"items": []}}, "gallery"),
    ({"url": "https://i.redd.it/abc.PNG", "is_self": False}, "image"),
    ({"url": "https://i.redd.it/abc.jpg?width=100", "is_self": False}, "image"),
    ({"post_hint": "rich:video", "is_self": False}, "video"),
    ({"is_self": True}, "text"),
    ({"url": "https://news.site/story", "is_self": False}, "link"),
])
def test_detect_media_type(payload, expected):
    assert detect_media_type(payload) == expected


def test_fetch_data_rejects_unknown_time_filter():
    fetcher, _ = make_fetcher([])
    with pytest.raises(ValueError):
        fetcher.fetch_data(["python"], time_filter="fortnight")


def test_fetch_data_strips_r_prefix_and_tags_source():
    responses = [FakeResponse(listing([raw_post("a", "first")], after=None))]
    fetcher, session = make_fetcher(responses)
    df = fetcher.fetch_data(["r/Python"], limit=5)
    assert "/r/Python/top/.json" in session.requests[0]
    assert df.loc[0, "subreddit"] == "Python"


def test_fetch_data_with_no_valid_names_returns_empty_schema():
    fetcher, session = make_fetcher([])
    df = fetcher.fetch_data(["  ", "r/"], limit=5)
    assert df.empty
    assert list(df.columns) == list(POST_COLUMNS)
    assert session.requests == []


def test_pagination_follows_cursor_until_limit():
    page1 = listing([raw_post(f"a{i}", f"t{i}") for i in range(3)], after="c1")
    page2 = listing([raw_post(f"b{i}", f"t{i}") for i in range(3)], after="c2")
    fetcher, session = make_fetcher([FakeResponse(page1), FakeResponse(page2)])
    children = fetcher._fetch_paginated("https://example.test/r/x/top/.json?t=month", limit=5)
    assert len(children) == 5
    assert "after=c1" in session.requests[1]


def test_pagination_stops_on_repeated_cursor():
    # Reddit sometimes echoes the same cursor forever; the loop must not spin.
    page = listing([raw_post("a", "t")], after="same")
    fetcher, session = make_fetcher([FakeResponse(page) for _ in range(10)])
    children = fetcher._fetch_paginated("https://example.test/r/x/top/.json?t=month", limit=100)
    assert len(children) == 2
    assert len(session.requests) == 2


def test_pagination_stops_when_no_cursor():
    page = listing([raw_post("a", "t")], after=None)
    fetcher, session = make_fetcher([FakeResponse(page), FakeResponse(page)])
    children = fetcher._fetch_paginated("https://example.test/r/x/top/.json?t=month", limit=50)
    assert len(children) == 1
    assert len(session.requests) == 1


def test_retries_on_rate_limit_then_succeeds():
    ok = listing([raw_post("a", "t")], after=None)
    fetcher, session = make_fetcher([FakeResponse(None, status_code=429), FakeResponse(ok)])
    data = fetcher._get("https://example.test/x")
    assert data is not None
    assert len(session.requests) == 2


def test_gives_up_after_max_retries():
    fetcher, session = make_fetcher([FakeResponse(None, status_code=503) for _ in range(5)])
    assert fetcher._get("https://example.test/x") is None
    assert len(session.requests) == 2  # max_retries=2


def test_client_error_is_not_retried():
    fetcher, session = make_fetcher([FakeResponse(None, status_code=404) for _ in range(5)])
    assert fetcher._get("https://example.test/x") is None
    assert len(session.requests) == 1


def test_network_exception_is_swallowed_not_raised():
    fetcher, _ = make_fetcher([requests.ConnectionError("boom"), requests.ConnectionError("boom")])
    assert fetcher._get("https://example.test/x") is None


def test_fetch_data_survives_a_failing_subreddit():
    ok = FakeResponse(listing([raw_post("a", "good")], after=None))
    fetcher, _ = make_fetcher([FakeResponse(None, status_code=404), ok])
    df = fetcher.fetch_data(["broken", "working"], limit=5)
    assert len(df) == 1
    assert df.loc[0, "subreddit"] == "working"
