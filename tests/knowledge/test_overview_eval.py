import json

from backend.knowledge import overview_eval as ov


def test_parse_points_drops_empty_and_unknown_paths():
    raw = json.dumps({"points": [{"text": "Hat 3 asamali", "paths": ["hat/HAT_PLANI.md", "yok.md"]},
                                 {"text": "  ", "paths": []}]})
    assert ov.parse_points(raw, {"hat/HAT_PLANI.md": "..."}) == [{"text": "Hat 3 asamali", "paths": ["hat/HAT_PLANI.md"]}]


def test_parse_points_bad_json_is_empty():
    assert ov.parse_points("bozuk") == []


def test_parse_coverage_requires_one_verdict_per_point():
    assert ov.parse_coverage('{"covered": [true, false], "contradictions": []}', 2)["covered"] == [True, False]
    assert ov.parse_coverage('{"covered": [true], "contradictions": []}', 2)["error"]


def test_coverage_and_source_recall():
    assert ov.coverage({"covered": [True, False, True, True]}) == 0.75
    assert ov.coverage({"covered": None}) is None
    points = [{"text": "a", "paths": ["A.md", "hat/B.md"]}, {"text": "b", "paths": ["hat/B.md"]}]
    assert ov.source_recall(points, ["hat\\B.md"]) == 0.5
    assert ov.source_recall([{"text": "a", "paths": []}], ["A.md"]) is None


def test_cover_prompt_numbers_points_and_clips_text():
    prompt = ov.build_cover_prompt("soru", [{"text": "bir"}, {"text": "iki"}], "x" * 50, max_chars=10)
    assert "1. bir\n2. iki" in prompt and "[... kesildi]" in prompt


def test_aggregate_skips_missing_values():
    rows = [{"context_coverage": 0.5, "answer_coverage": None, "contradictions": ["c"]},
            {"context_coverage": 1.0, "answer_coverage": 0.8, "contradictions": [], "judge_error": "x"}]
    summary = ov.aggregate(rows)
    assert summary["context_coverage"] == 0.75 and summary["answer_coverage"] == 0.8
    assert summary["contradictions"] == 1 and summary["judge_errors"] == 1
