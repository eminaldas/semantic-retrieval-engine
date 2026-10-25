from types import SimpleNamespace

from backend.knowledge import decompose
from backend.shared import gemini_retry, gemini_usage


def test_parse_drops_duplicates_and_question_itself():
    raw = '{"queries": ["YAYIN.md Gemini anahtari", "soru", "yayin.md gemini anahtari", "", "Dagitim adayi butce"]}'
    assert decompose.parse_queries(raw, "Soru") == ["YAYIN.md Gemini anahtari", "Dagitim adayi butce"]


def test_parse_single_part_or_broken_means_no_split():
    assert decompose.parse_queries('{"queries": ["tek"]}', "soru") == []
    assert decompose.parse_queries("bozuk", "soru") == []
    assert decompose.parse_queries('{"queries": "x"}', "soru") == []


def test_parse_caps_parts():
    raw = '{"queries": ["a", "b", "c", "d"]}'
    assert decompose.parse_queries(raw, "soru", max_parts=3) == ["a", "b", "c"]


class FakeClient:
    def __init__(self, text=None, error=None):
        self.models, self.text, self.error, self.prompts = self, text, error, []

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text, usage_metadata=None)


def test_decompose_calls_model_once(monkeypatch):
    monkeypatch.setattr(gemini_usage, "log_usage", lambda *a, **k: None)
    client = FakeClient('{"queries": ["a belgesi", "b belgesi"]}')
    assert decompose.decompose(client, "a ile b nasil?") == ["a belgesi", "b belgesi"]
    assert len(client.prompts) == 1 and "a ile b nasil?" in client.prompts[0]


def test_decompose_failure_or_no_client_falls_back_to_plain_search(monkeypatch):
    monkeypatch.setattr(gemini_usage, "log_usage", lambda *a, **k: None)
    monkeypatch.setattr(gemini_retry, "call_with_retry", lambda fn, **k: fn())
    assert decompose.decompose(FakeClient(error=RuntimeError("402")), "soru") == []
    assert decompose.decompose(None, "soru") == []
