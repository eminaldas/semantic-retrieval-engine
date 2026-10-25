from types import SimpleNamespace

import pytest

from backend.knowledge import agent
from backend.knowledge.agent import AgentResult, ResearchAgent
from backend.shared import gemini_usage


class FakeService:
    def __init__(self):
        self.calls = []

    def resolve_project(self, project):
        return project or "factprobe"

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return "[1] KARARLAR_OZET.md (satır 3) · skor 0.90\nPostgres + pgvector"

    def get_document(self, path, project=None, max_chars=0):
        self.calls.append(("get_document", path, project))
        return "x" * 10_000

    def overview(self, project=None):
        return "harita"


def _response(text=None, calls=()):
    function_calls = [SimpleNamespace(name=name, args=args) for name, args in calls]
    parts = [SimpleNamespace(text=text, thought=False)] if text else []
    return SimpleNamespace(text=text, function_calls=function_calls or None,
                           usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=10),
                           candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=parts), finish_reason="STOP")])


class FakeGemini:
    """Sirayla hazir yanitlari doner; her cagrinin araclari acik mi kapali mi kaydeder."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.modes = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.modes.append(config.tool_config.function_calling_config.mode)
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def no_usage_log(monkeypatch):
    monkeypatch.setattr(gemini_usage, "log_usage", lambda *a, **k: None)  # gercek maliyet loguna yazma


def test_agent_searches_then_answers_with_citation():
    service = FakeService()
    gemini = FakeGemini([_response(calls=[("search", {"query": "graf db", "project": "fake-news"})]),
                         _response(text="Postgres + pgvector [KARARLAR_OZET.md].")])
    result = ResearchAgent(service, client=gemini).ask("Graf icin hangi veritabani?")
    assert result.answer.startswith("Postgres") and result.citations == ["KARARLAR_OZET.md"]
    # model proje secemez: ajan her zaman kendi projesinde arar
    assert service.calls == [("search", {"project": "factprobe", "query": "graf db"})]
    assert result.prompt_tokens == 200 and len(result.steps) == 1


def test_step_limit_forces_final_answer_without_tools():
    gemini = FakeGemini([_response(calls=[("overview", {})]), _response(calls=[("overview", {})]),
                         _response(text="Belgelerde bulamadim.")])
    result = ResearchAgent(FakeService(), client=gemini, max_steps=2).ask("?")
    assert [str(m) for m in gemini.modes][-1].endswith("NONE") and result.says_not_found


def test_empty_stop_gets_one_tool_free_retry():
    gemini = FakeGemini([_response(calls=[("overview", {})]), _response(), _response(text="Cevap [a.md].")])
    result = ResearchAgent(FakeService(), client=gemini).ask("?")
    assert result.answer == "Cevap [a.md]." and str(gemini.modes[-1]).endswith("NONE")


def test_quoted_document_text_is_not_a_not_found_answer():
    # d04: belge 'ŞİMDİLİK YOK' diyor; cevap bunu aktariyor, "bulamadim" demiyor.
    answer = "Belgelerde 'ŞİMDİLİK YOK, sonraya not' ifadesi yer almakta [GELECEK_ISLER.md]."
    assert not agent.AgentResult(answer).says_not_found
    assert not agent.AgentResult("Belgelerde “yok” diye işaretlenmiş [a.md].").says_not_found
    assert agent.AgentResult("Belgelerde bulamadım. Yakın olarak 'eşik' geçiyor [a.md].").says_not_found
    # ek kesme isaretleri arasi alinti sayilmaz, asil hukum korunur
    assert agent.AgentResult("KARARLAR'da ve belgelerde bu bilgi yok; HAT_PLANI'nda da gecmiyor.").says_not_found


def test_not_found_phrasings_only_count_at_the_start():
    assert agent.AgentResult("Proje belgelerinde bu konuda bir bilgi bulunmamaktadır.").says_not_found
    assert not agent.AgentResult("Postgres seçildi [a.md]. " + "x" * 400 + " Belgelerde bulamadım.").says_not_found


def test_tool_output_is_truncated_and_errors_become_text():
    a = ResearchAgent(FakeService(), client=None)
    assert len(a.run_tool("get_document", {"path": "a.md"})) == agent.MAX_TOOL_CHARS
    assert a.run_tool("rm", {}).startswith("Bilinmeyen arac")


def test_citations_parse_lists_and_locations():
    r = AgentResult(answer="a [KARARLAR_OZET.md, graf/README.md] b [hat/RAPOR_ESIK.md (satır 3-9)] c [1]")
    assert r.citations == ["KARARLAR_OZET.md", "graf/README.md", "hat/RAPOR_ESIK.md"]


def test_score_answer():
    cited = AgentResult(answer="Postgres [KARARLAR_OZET.md]")
    assert agent.score_answer({"expected_paths": ["KARARLAR_OZET.md"]}, cited)["ok"]
    assert not agent.score_answer({"expected_paths": ["graf/README.md"]}, cited)["ok"]
    refusal = AgentResult(answer="Belgelerde bulamadım.")
    assert agent.score_answer({"expected_paths": []}, refusal)["ok"]


def test_agent_trace_has_llm_and_tool_spans():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from backend.shared import tracing

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.install(provider)
    try:
        gemini = FakeGemini([_response(calls=[("search", {"query": "graf db"})]),
                             _response(text="Postgres + pgvector [KARARLAR_OZET.md].")])
        ResearchAgent(FakeService(), client=gemini).ask("Graf icin hangi veritabani?")
    finally:
        tracing.uninstall()
    spans = exporter.get_finished_spans()
    kinds = [s.attributes["openinference.span.kind"] for s in spans]
    assert kinds.count("LLM") == 2 and kinds.count("TOOL") == 1 and kinds.count("AGENT") == 1
    root = next(s for s in spans if s.name == "agent.ask")
    assert tuple(root.attributes["rag.citations"]) == ("KARARLAR_OZET.md",)
    assert root.attributes["llm.token_count.prompt"] == 200
    assert all(s.context.trace_id == root.context.trace_id for s in spans)
