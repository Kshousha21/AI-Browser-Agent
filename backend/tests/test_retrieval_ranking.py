"""Lexical, cosine, fusion, query bounds and exact source provenance."""
import pytest
from pydantic import ValidationError

from local_retrieval import RetrievalError, RetrievalSettings
from local_retrieval.ranking import fuse
from .retrieval_fixtures import fixture_service


@pytest.fixture
def corpus(tmp_path):
    values = fixture_service(tmp_path)
    values[0].index_all()
    return values


@pytest.mark.parametrize("query,topic", [
    ("xylophotonic9", "optics"), ("thermal packaging heat", "cooling"),
    ("agricultural irrigation crops", "agriculture"), ("Automotive LIDAR", "lidar"),
    ("Optische Verbindungen", "german"), ("co-packaged optics", "optics"),
])
def test_lexical_exact_multiple_scientific_and_german_terms(corpus, query, topic):
    service, _, documents, embedder = corpus
    embedder.calls.clear()
    result = service.search(query, mode="lexical")
    assert result.results[0].document_id == documents[topic].document_id
    assert result.results[0].bm25_rank == 1 and result.results[0].semantic_rank is None
    assert embedder.calls == []


def test_bm25_rare_term_beats_shared_keyword(tmp_path):
    service, _, documents, _ = fixture_service(tmp_path, {
        "rare": "packaging xenophotonic7", "common": "packaging packaging ordinary",
        "other": "packaging ordinary machinery", "more": "packaging generic example",
    })
    service.index_all()
    result = service.search("packaging xenophotonic7", mode="lexical")
    assert result.results[0].document_id == documents["rare"].document_id


def test_semantic_recovers_different_wording_and_hybrid_retains_exact_signal(corpus):
    service, _, documents, _ = corpus
    # 'hot' is absent from the source but shares a test-only feature with heat.
    assert service.search("hot", mode="lexical").returned_count == 0
    semantic = service.search("hot", mode="semantic")
    assert semantic.results[0].document_id == documents["cooling"].document_id
    hybrid = service.search("hot thermal", mode="hybrid")
    assert hybrid.results[0].document_id == documents["cooling"].document_id
    assert hybrid.results[0].bm25_rank == hybrid.results[0].semantic_rank == 1
    assert hybrid.results[0].hybrid_score == pytest.approx(2 / 61)


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_filter_and_provenance(corpus, mode):
    service, store, documents, _ = corpus
    identifier = documents["optics"].document_id
    result = service.search("optical light", document_ids=[identifier], mode=mode)
    assert result.returned_count == 1
    hit = result.results[0]
    original = store.chunks(identifier)[0]
    for name in type(original).model_fields:
        assert getattr(hit, name) == getattr(original, name)
    assert hit.trust == "untrusted_document_data"
    assert hit.char_end - hit.char_start == len(hit.text)
    assert "text" not in result.safe_metadata()["results"][0]
    assert "query" not in result.model_dump()


@pytest.mark.parametrize("kwargs", [
    {"query": ""}, {"query": " \n\t "}, {"query": "a" * 1001}, {"query": 12},
    {"query": "a\0b"}, {"top_k": 0}, {"top_k": 21}, {"top_k": True},
    {"top_k": "5"}, {"mode": "SQL"}, {"document_ids": []},
    {"document_ids": ["not-a-hash"]}, {"document_ids": ["a" * 64] * 2},
])
def test_strict_query_bounds_reject_without_embedding_or_storage(tmp_path, kwargs):
    service, _, _, embedder = fixture_service(tmp_path, {})
    arguments = {"query": "synthetic", **kwargs}
    with pytest.raises(RetrievalError, match="^invalid_arguments$"):
        service.search(**arguments)
    assert not service.index.path.exists() and not embedder.calls


def test_unknown_filter_does_not_broaden(corpus):
    service, _, documents, _ = corpus
    with pytest.raises(RetrievalError, match="^unknown_document$"):
        service.search("photonics", document_ids=[documents["optics"].document_id, "a" * 64])


@pytest.mark.parametrize("query", ["zzzznomatchzzzz", "!!!", '" OR * NEAR () : -'])
def test_lexical_no_match_and_fts_operators_are_literal(corpus, query):
    assert corpus[0].search(query, mode="lexical").returned_count == 0


def test_numeric_punctuation_query(tmp_path):
    service, _, documents, _ = fixture_service(tmp_path, {"numeric": "Synthetic sensor 1550 nm uses 12.5 mm packaging.", "other": "Unrelated fields and crops."})
    service.index_all()
    assert service.search("1550 nm", mode="lexical").results[0].document_id == documents["numeric"].document_id
    assert service.search("12.5", mode="lexical").results[0].document_id == documents["numeric"].document_id


def test_deterministic_rrf_and_ties_independent_of_insert_order(tmp_path):
    service, _, documents, _ = fixture_service(tmp_path, {"one": "shared packaging", "two": "shared packaging\n"})
    service.index_all()
    with service.index.connection() as connection:
        indexed = service.index.read(connection, service.embedder.identity)
    ids = list(indexed.chunks)
    assert len(ids) == 2
    result = fuse(indexed.chunks, ids, ids[::-1], {}, mode="hybrid", top_k=2, rrf_k=60)
    assert [hit.document_id for hit in result] == sorted(doc.document_id for doc in documents.values())
    assert result[0].hybrid_score == pytest.approx(1 / 61 + 1 / 62)
    again = fuse(dict(reversed(list(indexed.chunks.items()))), ids, ids[::-1], {}, mode="hybrid", top_k=2, rrf_k=60)
    assert result == again
    assert service.search("shared", mode="hybrid") == service.search("shared", mode="hybrid")


@pytest.mark.parametrize("values", [{"candidate_pool": 19}, {"candidate_pool": 201}, {"candidate_pool": True}, {"rrf_k": 0}, {"max_chunks": 0}])
def test_candidate_and_configuration_bounds(values):
    with pytest.raises(ValidationError):
        RetrievalSettings(**values)
