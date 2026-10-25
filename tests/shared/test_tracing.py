import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from backend.knowledge import search, store
from backend.mcp import server as srv
from backend.shared import tracing


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.install(provider)
    yield lambda: {s.name: s for s in exporter.get_finished_spans()}
    tracing.uninstall()


class TestSpan:
    def test_disabled_span_is_noop(self):
        with tracing.span("x", foo=1) as s:
            s.set("a", {"b": 1}).error(ValueError("yok"))  # hicbir sey yapmamali

    def test_attributes_kind_and_json(self, spans):
        with tracing.span("x", tracing.RETRIEVER, a=None) as s:
            s.update({"meta": {"k": "v"}, "ids": ["a", "b"], "mixed": [1, "a"], "n": 3})
        attrs = spans()["x"].attributes
        assert attrs["openinference.span.kind"] == "RETRIEVER"
        assert "a" not in attrs
        assert json.loads(attrs["meta"]) == {"k": "v"}
        assert tuple(attrs["ids"]) == ("a", "b")
        assert json.loads(attrs["mixed"]) == [1, "a"]

    def test_exception_marks_error_and_reraises(self, spans):
        with pytest.raises(RuntimeError):
            with tracing.span("x"):
                raise RuntimeError("bozuk")
        assert spans()["x"].status.status_code.name == "ERROR"

    def test_clip_respects_content_switch(self):
        provider = TracerProvider()
        tracing.install(provider, content=False)
        try:
            assert tracing.clip("gizli") is None
        finally:
            tracing.uninstall()
        assert tracing.clip("x" * 2000).endswith("...")


class FakeEmbedder:
    def encode(self, text):
        return [0.0, 1.0]


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.9 if "dogru" in text else 0.01 for _, text in pairs]


def _pool():
    return [
        {"chunk_id": "p:a.md#0001", "path": "a.md", "location": "satır 1-2", "text": "alakasiz", "fusion_score": 0.5},
        {"chunk_id": "p:b.md#0001", "path": "b.md", "location": "satır 1-2", "text": "dogru cevap", "fusion_score": 0.4},
    ]


class TestSearchSpans:
    def _run(self, monkeypatch, branches):
        monkeypatch.setattr(store, "hybrid_search", lambda *a, **k: _pool())
        monkeypatch.setattr(store, "branch_ranks", branches)
        monkeypatch.setattr(search, "fetch_chunks", lambda client, ids: {})
        return search.search(None, FakeEmbedder(), FakeCrossEncoder(), "soru", project="factprobe", top_k=2)

    def test_pipeline_spans_and_routing(self, spans, monkeypatch):
        results = self._run(monkeypatch, lambda *a, **k: {"dense": ["p:b.md#0001"],
                                                          "sparse": ["p:a.md#0001", "p:b.md#0001"]})
        assert results[0]["path"] == "b.md"
        got = spans()
        assert {"search", "embed", "hybrid_retrieve", "rerank"} <= set(got)
        retrieve = got["hybrid_retrieve"].attributes
        assert retrieve["rag.pool_both"] == 1 and retrieve["rag.pool_bm25_only"] == 1
        assert json.loads(retrieve["retrieval.documents.1.document.metadata"])["dense_rank"] == 1
        rerank = got["rerank"].attributes
        assert rerank["rag.top1_fusion_rank"] == 2  # reranker RRF'in ikincisini basa aldi
        assert rerank["reranker.output_documents.0.document.id"] == "p:b.md#0001"
        root = got["search"].attributes
        assert root["rag.top1_score"] == pytest.approx(0.9)
        assert root["rag.below_threshold"] == 1
        assert root["rag.abstained"] is False
        assert root["rag.project"] == "factprobe"
        assert got["embed"].parent.span_id == got["search"].context.span_id

    def test_branch_failure_does_not_break_search(self, spans, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("qdrant")
        assert self._run(monkeypatch, boom)
        assert "ConnectionError" in spans()["hybrid_retrieve"].attributes["rag.branch_error"]

    def test_branches_not_queried_when_disabled(self, monkeypatch):
        def fail(*a, **k):
            raise AssertionError("izleme kapaliyken ek sorgu atilmamali")
        assert self._run(monkeypatch, fail)


class FakeService:
    def search(self, query):
        return f"sonuc: {query}"

    def list_projects(self):
        raise RuntimeError("qdrant kapali")


class TestToolSpans:
    def test_tool_call_span(self, spans):
        srv.MCPServer(FakeService()).handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                             "params": {"name": "search", "arguments": {"query": "esik"}}})
        attrs = spans()["tool.search"].attributes
        assert attrs["tool.name"] == "search"
        assert json.loads(attrs["tool.parameters"]) == {"query": "esik"}
        assert attrs["output.value"] == "sonuc: esik"

    def test_tool_error_span(self, spans):
        response = srv.MCPServer(FakeService()).handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                                        "params": {"name": "list_projects", "arguments": {}}})
        assert response["result"]["isError"]
        span = spans()["tool.list_projects"]
        assert span.status.status_code.name == "ERROR"
        assert span.attributes["rag.error_kind"] == "internal"
