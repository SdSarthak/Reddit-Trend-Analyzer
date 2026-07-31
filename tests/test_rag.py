import json

import numpy as np
import pandas as pd
import pytest

from conftest import FakeGeminiClient
from subsense_engine import RAGEngine, coerce_schema, empty_frame


def build_engine(tmp_path, client=None):
    return RAGEngine(persist_dir=str(tmp_path / "store"), client=client or FakeGeminiClient())


def corpus():
    rows = [
        {"post_id": "a", "subreddit": "cooking", "title": "sourdough bread starter tips",
         "body": "feeding schedules", "score": 120, "num_comments": 30},
        {"post_id": "b", "subreddit": "devops", "title": "kubernetes cluster autoscaling",
         "body": "node pools", "score": 90, "num_comments": 12},
        {"post_id": "c", "subreddit": "cooking", "title": "cast iron seasoning guide",
         "body": "oil and heat", "score": 45, "num_comments": 8},
    ]
    df = coerce_schema(pd.DataFrame(rows))
    df['created_utc'] = pd.Timestamp("2024-05-05 12:00")
    return df


def test_index_builds_and_reports_success(tmp_path):
    engine = build_engine(tmp_path)
    assert engine.index_data(corpus(), batch_delay=0) is True
    assert engine.is_indexed()
    assert len(engine.docs) == 3
    assert engine.last_error is None


def test_embeddings_are_l2_normalised(tmp_path):
    engine = build_engine(tmp_path)
    engine.index_data(corpus(), batch_delay=0)
    norms = np.linalg.norm(engine.doc_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_index_persists_and_reloads_across_instances(tmp_path):
    engine = build_engine(tmp_path)
    engine.index_data(corpus(), batch_delay=0)

    reopened = build_engine(tmp_path)
    assert reopened.is_indexed()
    assert len(reopened.docs) == 3
    assert reopened.fingerprint == engine.fingerprint


def test_persisted_docs_are_plain_json(tmp_path):
    # Timestamps and numpy scalars from `to_dict('records')` used to make the
    # whole save fail silently, losing the store.
    engine = build_engine(tmp_path)
    engine.index_data(corpus(), batch_delay=0)
    with open(engine._docs_file, encoding='utf-8') as f:
        payload = json.load(f)
    assert set(payload) == {'fingerprint', 'docs'}
    for doc in payload['docs']:
        for value in doc.values():
            assert isinstance(value, (str, int, float))


def test_reindex_is_skipped_for_the_same_corpus(tmp_path):
    client = FakeGeminiClient()
    engine = build_engine(tmp_path, client)
    engine.index_data(corpus(), batch_delay=0)
    calls = len(client.embed_calls)

    engine.index_data(corpus(), batch_delay=0)
    assert len(client.embed_calls) == calls  # no wasted API quota


def test_different_corpus_triggers_reindex(tmp_path):
    client = FakeGeminiClient()
    engine = build_engine(tmp_path, client)
    engine.index_data(corpus(), batch_delay=0)
    calls = len(client.embed_calls)

    other = corpus().assign(post_id=["x", "y", "z"])
    engine.index_data(other, batch_delay=0)
    assert len(client.embed_calls) > calls
    assert [d['post_id'] for d in engine.docs] == ["x", "y", "z"]


def test_force_reindex_overrides_the_fingerprint_check(tmp_path):
    client = FakeGeminiClient()
    engine = build_engine(tmp_path, client)
    engine.index_data(corpus(), batch_delay=0)
    calls = len(client.embed_calls)

    engine.index_data(corpus(), force_reindex=True, batch_delay=0)
    assert len(client.embed_calls) > calls


def test_indexing_empty_dataset_reports_an_error(tmp_path):
    engine = build_engine(tmp_path)
    assert engine.index_data(empty_frame(), batch_delay=0) is False
    assert "empty" in engine.last_error.lower()
    assert not engine.is_indexed()


def test_embedding_failure_is_captured_not_raised(tmp_path):
    class BrokenClient(FakeGeminiClient):
        def embed(self, texts, task_type="retrieval_document"):
            raise RuntimeError("quota exceeded")

    engine = build_engine(tmp_path, BrokenClient())
    assert engine.index_data(corpus(), batch_delay=0) is False
    assert "quota exceeded" in engine.last_error
    assert not engine.is_indexed()


def test_short_embedding_batch_is_rejected(tmp_path):
    class ShortClient(FakeGeminiClient):
        def embed(self, texts, task_type="retrieval_document"):
            return super().embed(texts, task_type)[:-1]

    engine = build_engine(tmp_path, ShortClient())
    assert engine.index_data(corpus(), batch_delay=0) is False
    assert "mismatch" in engine.last_error.lower()


def test_batching_covers_every_document(tmp_path):
    client = FakeGeminiClient()
    engine = build_engine(tmp_path, client)
    engine.index_data(corpus(), batch_size=2, batch_delay=0)
    document_batches = [c for c in client.embed_calls if c[1] == "retrieval_document"]
    assert [len(batch) for batch, _ in document_batches] == [2, 1]
    assert len(engine.doc_embeddings) == 3


def test_retrieval_ranks_the_relevant_document_first(tmp_path):
    engine = build_engine(tmp_path)
    engine.index_data(corpus(), batch_delay=0)
    hits = engine.retrieve("kubernetes cluster autoscaling", top_k=1)
    assert hits[0]['post_id'] == "b"


def test_retrieval_respects_top_k(tmp_path):
    engine = build_engine(tmp_path)
    engine.index_data(corpus(), batch_delay=0)
    assert len(engine.retrieve("bread", top_k=2)) == 2
    assert len(engine.retrieve("bread", top_k=99)) == 3  # clamped to corpus size


def test_retrieval_before_indexing_raises(tmp_path):
    engine = build_engine(tmp_path)
    with pytest.raises(RuntimeError):
        engine.retrieve("anything")


def test_query_returns_answer_and_cited_context(tmp_path):
    client = FakeGeminiClient()
    engine = build_engine(tmp_path, client)
    engine.index_data(corpus(), batch_delay=0)

    answer, context = engine.query("kubernetes cluster autoscaling", top_k=1)
    assert answer == "synthetic answer"
    assert "r/devops" in context
    assert "kubernetes cluster autoscaling" in client.prompts[0]


def test_query_without_an_index_explains_itself(tmp_path):
    engine = build_engine(tmp_path)
    answer, context = engine.query("anything")
    assert "Index not built" in answer
    assert context == ""


def test_query_surfaces_the_last_indexing_error(tmp_path):
    class BrokenClient(FakeGeminiClient):
        def embed(self, texts, task_type="retrieval_document"):
            raise RuntimeError("quota exceeded")

    engine = build_engine(tmp_path, BrokenClient())
    engine.index_data(corpus(), batch_delay=0)
    answer, _ = engine.query("anything")
    assert "quota exceeded" in answer


def test_query_failure_is_returned_not_raised(tmp_path):
    client = FakeGeminiClient()
    engine = build_engine(tmp_path, client)
    engine.index_data(corpus(), batch_delay=0)

    def boom(prompt):
        raise RuntimeError("model unavailable")

    client.generate = boom
    answer, context = engine.query("anything")
    assert "model unavailable" in answer
    assert context == ""


def test_numpy_fallback_matches_faiss_ranking(tmp_path):
    engine = build_engine(tmp_path)
    engine.index_data(corpus(), batch_delay=0)
    with_index = [d['post_id'] for d in engine.retrieve("sourdough bread starter", top_k=3)]

    engine.index = None  # simulate faiss being unavailable
    without_index = [d['post_id'] for d in engine.retrieve("sourdough bread starter", top_k=3)]
    assert with_index == without_index


def test_corrupt_store_is_discarded_rather_than_crashing(tmp_path):
    engine = build_engine(tmp_path)
    engine.index_data(corpus(), batch_delay=0)
    with open(engine._docs_file, 'w', encoding='utf-8') as f:
        f.write("{not json")

    reopened = build_engine(tmp_path)
    assert reopened.docs == []
    assert not reopened.is_indexed()


def test_legacy_bare_list_store_still_loads(tmp_path):
    engine = build_engine(tmp_path)
    engine.index_data(corpus(), batch_delay=0)
    with open(engine._docs_file, encoding='utf-8') as f:
        docs = json.load(f)['docs']
    with open(engine._docs_file, 'w', encoding='utf-8') as f:
        json.dump(docs, f)

    reopened = build_engine(tmp_path)
    assert len(reopened.docs) == 3
    assert reopened.fingerprint is None  # forces a refresh on next index
