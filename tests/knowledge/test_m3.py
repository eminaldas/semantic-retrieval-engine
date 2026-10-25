from types import SimpleNamespace

import pytest

from backend.knowledge import eval as golden
from backend.knowledge import m3, projects


def test_embed_text_matches_what_sync_embeds():
    raw = "# Karar\n\nEsik 0,91 secildi.\n"
    record = projects.build_chunk_records("factprobe", "KARARLAR.md", raw, [("Esik 0,91 secildi.", "satır 3-3")],
                                          contexts=["Esik karari."])[0]
    assert m3.embed_text_from_payload(record) == record["embed_text"]


def test_sparse_weights_keep_max_and_drop_special_tokens():
    out = m3.sparse_weights([0, 5, 7, 5, 2, 9], [0.9, 0.1, 0.4, 0.3, 0.8, 0.0], skip_ids={0, 2})
    assert out == {5: 0.3, 7: 0.4}


class FakeHeads:
    def encode(self, texts):
        return [({11: 0.5}, [[0.0] * m3.COLBERT_DIM])]


class FakeClient:
    def __init__(self):
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=[SimpleNamespace(payload={"path": "a.md"}, score=0.7)])


@pytest.mark.parametrize("name, usings, query_vector", [
    ("bm25", ["dense", "bm25"], None),
    ("m3-sparse", ["dense", "m3_sparse"], None),
    ("hybrid3", ["dense", "bm25", "m3_sparse"], None),
    ("colbert", ["dense", "bm25", "m3_sparse"], "colbert"),
])
def test_retrievers_build_the_right_prefetch(name, usings, query_vector):
    client = FakeClient()
    pool = m3.make_retriever(name, FakeHeads())(client, [0.1, 0.2], "esik karari", limit=20)
    call = client.calls[0]
    assert call["collection_name"] == m3.COLLECTION_NAME
    assert [p.using for p in call["prefetch"]] == usings
    assert call["using"] == query_vector
    assert pool == [{"path": "a.md", "fusion_score": 0.7}]


def test_unknown_retriever():
    with pytest.raises(ValueError):
        m3.make_retriever("yok", FakeHeads())


def test_pool_hit_is_document_level():
    record = {"expected_paths": ["graf/README.md"]}
    assert golden.pool_hit(record, ["x.md", "graf\\README.md"])
    assert not golden.pool_hit(record, ["x.md"])
    assert golden.aggregate([{"answerable": True, "rank": 1, "abstained": False, "type": "t", "pool_hit": True},
                             {"answerable": True, "rank": None, "abstained": False, "type": "t",
                              "pool_hit": False}])["pool_recall"] == 0.5
