from types import SimpleNamespace

from backend.knowledge import agent, longctx
from backend.shared import gemini_retry, gemini_usage

DOCS = {"KARARLAR_OZET.md": "Esik 0,91.", "graf/README.md": "Graf PostgreSQL."}


def test_corpus_and_cited_evidence():
    corpus = longctx.build_corpus(DOCS)
    assert corpus.index('yol="KARARLAR_OZET.md"') < corpus.index('yol="graf/README.md"')
    evidence = longctx.cited_evidence(DOCS, ["graf\\readme.md", "yok.md", "graf/README.md"])
    assert evidence == ["[graf/README.md]\nGraf PostgreSQL."]


def test_instruction_keeps_answer_rules_without_search_steps():
    assert "Belgelerde bulamadim." in longctx.INSTRUCTION and "get_document" not in longctx.INSTRUCTION


class FakeClient:
    def __init__(self, answer, cache_error=None):
        self.models, self.caches = self, self
        self.answer, self.cache_error, self.calls, self.deleted = answer, cache_error, [], []

    def create(self, model, config):
        if self.cache_error:
            raise self.cache_error
        return SimpleNamespace(name="cachedContents/1")

    def delete(self, name):
        self.deleted.append(name)

    def generate_content(self, model, contents, config):
        self.calls.append((contents, config))
        usage = SimpleNamespace(prompt_token_count=1000, candidates_token_count=20,
                                cached_content_token_count=900 if config.cached_content else 0)
        return SimpleNamespace(text=self.answer, usage_metadata=usage)


def _quiet(monkeypatch):
    monkeypatch.setattr(gemini_usage, "log_usage", lambda *a, **k: None)
    monkeypatch.setattr(gemini_retry, "call_with_retry", lambda fn, **k: fn())


def test_cached_corpus_sends_only_the_question(monkeypatch):
    _quiet(monkeypatch)
    client = FakeClient("Esik 0,91 [KARARLAR_OZET.md].")
    reader = longctx.LongContextReader(DOCS, client=client)
    result = reader.ask("esik kac?")
    reader.ask("ikinci soru")
    assert [c[0] for c in client.calls] == ["esik kac?", "ikinci soru"]
    assert result.citations == ["KARARLAR_OZET.md"] and result.evidence[0].startswith("[KARARLAR_OZET.md]")
    assert result.cached_tokens == 900
    assert agent.cost_usd(result) < agent.cost_usd(agent.AgentResult("", prompt_tokens=1000, output_tokens=20))
    reader.close()
    assert client.deleted == ["cachedContents/1"]


def test_without_cache_the_whole_corpus_goes_with_each_question(monkeypatch):
    _quiet(monkeypatch)
    client = FakeClient("Belgelerde bulamadim.", cache_error=RuntimeError("min token"))
    result = longctx.LongContextReader(DOCS, client=client).ask("butce?")
    contents, config = client.calls[0]
    assert "Graf PostgreSQL." in contents and contents.endswith("Soru: butce?")
    assert config.system_instruction == longctx.INSTRUCTION
    assert result.says_not_found and result.evidence == []
