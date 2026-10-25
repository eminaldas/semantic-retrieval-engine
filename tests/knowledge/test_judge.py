from types import SimpleNamespace

from backend.knowledge import agent, judge
from backend.knowledge.agent import AgentResult
from backend.shared import gemini_usage


def test_prompt_includes_reference_only_when_given():
    with_ref = judge.build_prompt("soru", ["kanit"], "cevap", reference=["0.91", "0,91"])
    without = judge.build_prompt("soru", [], "cevap")
    assert "BEKLENEN BILGI" in with_ref and "0.91 / 0,91" in with_ref
    assert "BEKLENEN BILGI" not in without and "hicbir arac" in without


def test_long_evidence_is_cut():
    prompt = judge.build_prompt("s", ["x" * (judge.MAX_EVIDENCE_CHARS + 500)], "c")
    assert "[... kesildi]" in prompt


def test_parse_verdict_valid_and_broken():
    ok = judge.parse_verdict('{"grounded": true, "correct": false, "unsupported": ["a","b","c","d"], "reason": "r"}')
    assert ok["grounded"] is True and ok["correct"] is False and len(ok["unsupported"]) == 3 and ok["error"] is None
    bad = judge.parse_verdict("bozuk")
    assert bad["grounded"] is None and bad["error"]


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.models = self
        self.prompts = []

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        return SimpleNamespace(text=self.text, usage_metadata=SimpleNamespace(prompt_token_count=1000,
                                                                              candidates_token_count=50))


def test_judge_record_skips_abstain_and_unanswerable(monkeypatch):
    monkeypatch.setattr(gemini_usage, "log_usage", lambda *a, **k: None)
    client = FakeClient('{"grounded": true, "correct": true, "unsupported": [], "reason": "ok"}')
    answerable = {"question": "q", "expected_paths": ["A.md"], "expected_text": "0.91"}
    unanswerable = {"question": "q", "expected_paths": [], "expected_text": None}
    answered = AgentResult("Esik 0.91 [A.md].", evidence=["A.md: esik 0,91"])
    not_found = AgentResult("Belgelerde bulamadim.")
    assert agent.judge_record(client, unanswerable, answered) is None
    assert agent.judge_record(client, answerable, not_found) is None
    verdict = agent.judge_record(client, answerable, answered)
    assert verdict["grounded"] and verdict["prompt_tokens"] == 1000
    assert "A.md: esik 0,91" in client.prompts[0] and "0.91" in client.prompts[0]


def test_judge_summary_ignores_errors():
    good = {"grounded": True, "correct": False, "error": None, "prompt_tokens": 1000, "output_tokens": 10}
    bad = {"grounded": None, "correct": None, "error": "x", "prompt_tokens": 0, "output_tokens": 0}
    summary = agent.judge_summary([good, bad, None])
    assert summary["judged"] == 1 and summary["judge_errors"] == 1
    assert summary["grounded_rate"] == 1.0 and summary["correct_rate"] == 0.0
    assert agent.judge_summary([None]) == {}
