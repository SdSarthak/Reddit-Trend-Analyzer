import numpy as np
import pandas as pd
import pytest

from subsense_engine import (
    ModClassifier,
    TrendEngine,
    ViralityPredictor,
    coerce_schema,
    empty_frame,
    run_pipeline,
    summarize,
)


# --- TrendEngine -----------------------------------------------------------

def test_trend_engine_labels_every_post(posts_df):
    out = TrendEngine().extract_trends(posts_df)
    assert 'topic_cluster' in out.columns
    assert out['topic_keywords'].notna().all()
    assert out['topic_keywords'].str.len().gt(0).all()


def test_trend_engine_is_deterministic(posts_df):
    first = TrendEngine().extract_trends(posts_df)['topic_keywords'].tolist()
    second = TrendEngine().extract_trends(posts_df)['topic_keywords'].tolist()
    assert first == second


def test_trend_engine_separates_distinct_topics(posts_df):
    out = TrendEngine(max_clusters=2, posts_per_cluster=5).extract_trends(posts_df)
    hiring = out[out['title'].str.startswith('hiring')]['topic_cluster'].unique()
    ml = out[out['title'].str.startswith('machine')]['topic_cluster'].unique()
    assert len(hiring) == 1 and len(ml) == 1
    assert hiring[0] != ml[0]


def test_trend_engine_does_not_mutate_input(posts_df):
    before = list(posts_df.columns)
    TrendEngine().extract_trends(posts_df)
    assert list(posts_df.columns) == before


def test_trend_engine_handles_empty_input():
    out = TrendEngine().extract_trends(empty_frame())
    assert out.empty
    assert 'topic_keywords' in out.columns


def test_trend_engine_falls_back_when_vocabulary_is_all_stopwords():
    df = coerce_schema(pd.DataFrame([
        {"post_id": f"s{i}", "title": "the and of", "body": ""} for i in range(10)
    ]))
    out = TrendEngine().extract_trends(df)
    assert (out['topic_keywords'] == "General").all()


def test_top_topics_ranks_by_volume(posts_df):
    out = TrendEngine(max_clusters=2, posts_per_cluster=5).extract_trends(posts_df)
    topics = TrendEngine.top_topics(out)
    assert list(topics.columns) == ['topic_keywords', 'posts', 'avg_score', 'total_comments']
    assert topics['posts'].is_monotonic_decreasing


# --- ViralityPredictor -----------------------------------------------------

def test_features_are_identical_shape_for_train_and_inference(posts_df):
    train = ViralityPredictor.build_features(posts_df)
    single = ViralityPredictor.build_features(pd.DataFrame([
        {"title": "hello there", "media_type": "image", "created_utc": pd.Timestamp("2024-01-02 09:00")}
    ]))
    assert list(train.columns) == list(single.columns) == ViralityPredictor.FEATURES
    assert single.loc[0, 'title_len'] == 11
    assert single.loc[0, 'word_count'] == 2
    assert single.loc[0, 'has_media'] == 1
    assert single.loc[0, 'hour'] == 9


def test_build_features_tolerates_missing_columns():
    features = ViralityPredictor.build_features(pd.DataFrame({"title": ["a b c"]}))
    assert features.loc[0, 'hour'] == 12  # documented default when time is unknown
    assert features.loc[0, 'is_text'] == 1


def test_train_persists_and_reloads(tmp_path, posts_df):
    path = tmp_path / "model.json"
    predictor = ViralityPredictor(model_path=str(path))
    assert not predictor.is_trained

    scored = predictor.train_and_score(posts_df)
    assert predictor.is_trained
    assert path.exists()
    assert scored['predicted_score'].notna().all()

    reloaded = ViralityPredictor(model_path=str(path))
    assert reloaded.is_trained
    assert reloaded.predict_new("hiring engineers for our team", "text", 9) == pytest.approx(
        predictor.predict_new("hiring engineers for our team", "text", 9), rel=1e-5
    )


def test_predict_new_returns_zero_when_untrained(tmp_path):
    predictor = ViralityPredictor(model_path=str(tmp_path / "absent.json"))
    assert predictor.predict_new("anything", "text", 12) == 0.0


def test_small_dataset_is_not_trained_on(tmp_path, posts_df):
    predictor = ViralityPredictor(model_path=str(tmp_path / "model.json"))
    out = predictor.train_and_score(posts_df.head(3))
    assert not predictor.is_trained
    assert out['predicted_score'].isna().all()
    assert not (tmp_path / "model.json").exists()


def test_stale_model_with_other_features_is_rejected(tmp_path, posts_df):
    path = tmp_path / "stale.json"
    predictor = ViralityPredictor(model_path=str(path))
    predictor.model.fit(
        pd.DataFrame({"title_len": [1, 2, 3], "hour": [1, 2, 3]}), [1, 2, 3]
    )
    predictor.save_model()

    reloaded = ViralityPredictor(model_path=str(path))
    assert not reloaded.is_trained  # would have raised on every predict otherwise


def test_train_and_score_does_not_mutate_input(tmp_path, posts_df):
    predictor = ViralityPredictor(model_path=str(tmp_path / "model.json"))
    predictor.train_and_score(posts_df)
    assert 'predicted_score' not in posts_df.columns


# --- ModClassifier ---------------------------------------------------------

def build_risk_frame():
    rows = [
        {"post_id": "calm", "title": "a pleasant and wonderful guide", "upvote_ratio": 0.98,
         "is_locked": False, "is_stickied": False},
        {"post_id": "locked", "title": "a pleasant and wonderful guide", "upvote_ratio": 0.98,
         "is_locked": True, "is_stickied": False},
        {"post_id": "announcement", "title": "a pleasant and wonderful guide", "upvote_ratio": 0.98,
         "is_locked": True, "is_stickied": True},
        {"post_id": "contested", "title": "a pleasant and wonderful guide", "upvote_ratio": 0.40,
         "is_locked": False, "is_stickied": False},
        {"post_id": "toxic", "title": "this is terrible awful horrible and disgusting",
         "upvote_ratio": 0.30, "is_locked": True, "is_stickied": False},
    ]
    return coerce_schema(pd.DataFrame(rows))


def test_risk_scores_reflect_signals():
    out = ModClassifier().score_risk(build_risk_frame()).set_index('post_id')
    assert out.loc['calm', 'mod_risk_score'] == 0.0
    assert out.loc['locked', 'mod_risk_score'] == pytest.approx(0.8)
    assert out.loc['contested', 'mod_risk_score'] == pytest.approx(0.5)
    assert out.loc['toxic', 'mod_risk_score'] > out.loc['locked', 'mod_risk_score']


def test_stickied_lock_is_treated_as_an_announcement():
    out = ModClassifier().score_risk(build_risk_frame()).set_index('post_id')
    assert out.loc['announcement', 'mod_risk_score'] == 0.0
    assert out.loc['announcement', 'risk_reasons'] == "none"


def test_risk_reasons_are_explained():
    out = ModClassifier().score_risk(build_risk_frame()).set_index('post_id')
    assert out.loc['locked', 'risk_reasons'] == "locked thread"
    assert "low upvote ratio" in out.loc['toxic', 'risk_reasons']
    assert "negative sentiment" in out.loc['toxic', 'risk_reasons']


def test_high_risk_filter_uses_threshold():
    out = ModClassifier().score_risk(build_risk_frame())
    high = ModClassifier.high_risk(out)
    assert set(high['post_id']) == {'toxic'}


def test_score_risk_handles_empty_input():
    out = ModClassifier().score_risk(empty_frame())
    assert out.empty
    assert {'sentiment', 'mod_risk_score', 'risk_reasons'} <= set(out.columns)


def test_sentiment_is_bounded(posts_df):
    out = ModClassifier().score_risk(posts_df)
    assert out['sentiment'].between(-1, 1).all()


# --- Pipeline / summary ----------------------------------------------------

def test_run_pipeline_produces_all_derived_columns(tmp_path, posts_df):
    out = run_pipeline(posts_df, predictor=ViralityPredictor(model_path=str(tmp_path / "m.json")))
    for col in ('topic_keywords', 'mod_risk_score', 'sentiment', 'predicted_score'):
        assert col in out.columns


def test_summarize_reports_headline_metrics(tmp_path, posts_df):
    out = run_pipeline(posts_df, predictor=ViralityPredictor(model_path=str(tmp_path / "m.json")))
    summary = summarize(out)
    assert summary['total_posts'] == 20
    assert summary['subreddits'] == ['synthetic', 'synthetic2']
    assert summary['avg_score'] == pytest.approx(95.0)
    assert summary['media_posts'] == len(out[out['media_type'] != 'text'])
    assert summary['top_topics']
    assert isinstance(summary['busiest_hour'], int)


def test_summarize_of_empty_dataset_is_all_zeros():
    summary = summarize(empty_frame())
    assert summary['total_posts'] == 0
    assert summary['top_topics'] == []
    assert summary['busiest_hour'] is None
