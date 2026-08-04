import json

import pytest

from backend.shared import gemini_usage


class FakeUsageMetadata:
    def __init__(self, prompt_token_count=None, candidates_token_count=None, total_token_count=None):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count


class FakeResponse:
    def __init__(self, usage_metadata=None):
        self.usage_metadata = usage_metadata


class TestLogUsage:
    def test_writes_one_line_with_known_model_cost(self, tmp_path):
        log_file = tmp_path / "usage.jsonl"
        response = FakeResponse(FakeUsageMetadata(prompt_token_count=1000, candidates_token_count=500, total_token_count=1500))
        gemini_usage.log_usage("step2_extractor", "gemini-flash-lite-latest", response, log_path=str(log_file))
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["call_site"] == "step2_extractor"
        assert entry["model"] == "gemini-flash-lite-latest"
        assert entry["prompt_tokens"] == 1000
        assert entry["output_tokens"] == 500
        assert entry["total_tokens"] == 1500
        assert entry["estimated_cost_usd"] == pytest.approx(0.0003)
        assert "timestamp" in entry

    def test_unknown_model_records_null_cost(self, tmp_path):
        log_file = tmp_path / "usage.jsonl"
        response = FakeResponse(FakeUsageMetadata(prompt_token_count=100, candidates_token_count=50, total_token_count=150))
        gemini_usage.log_usage("some_site", "bilinmeyen-model-x", response, log_path=str(log_file))
        entry = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert entry["estimated_cost_usd"] is None

    def test_missing_usage_metadata_skips_logging(self, tmp_path):
        log_file = tmp_path / "usage.jsonl"
        response = FakeResponse(usage_metadata=None)
        gemini_usage.log_usage("step2_extractor", "gemini-flash-lite-latest", response, log_path=str(log_file))
        assert not log_file.exists()

    def test_appends_to_existing_file(self, tmp_path):
        log_file = tmp_path / "usage.jsonl"
        response = FakeResponse(FakeUsageMetadata(prompt_token_count=10, candidates_token_count=10, total_token_count=20))
        gemini_usage.log_usage("a", "gemini-flash-lite-latest", response, log_path=str(log_file))
        gemini_usage.log_usage("b", "gemini-flash-lite-latest", response, log_path=str(log_file))
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_pro_model_uses_tiered_pricing_above_200k(self, tmp_path):
        log_file = tmp_path / "usage.jsonl"
        response = FakeResponse(FakeUsageMetadata(prompt_token_count=250_000, candidates_token_count=1000, total_token_count=251_000))
        gemini_usage.log_usage("x", "gemini-pro-latest", response, log_path=str(log_file))
        entry = json.loads(log_file.read_text(encoding="utf-8").strip())
        # 250_000/1e6*2.50 + 1000/1e6*15.00 = 0.625 + 0.015 = 0.64
        assert entry["estimated_cost_usd"] == pytest.approx(0.64)


class TestAggregateUsage:
    def test_missing_file_returns_zeroed_totals(self, tmp_path):
        missing = tmp_path / "does_not_exist.jsonl"
        result = gemini_usage.aggregate_usage(str(missing))
        assert result == {
            "total_calls": 0, "total_tokens": 0, "total_cost_usd": 0.0,
            "all_costs_known": True, "by_call_site": [],
        }

    def test_aggregates_multiple_call_sites(self, tmp_path):
        log_file = tmp_path / "usage.jsonl"
        log_file.write_text(
            json.dumps({"call_site": "a", "model": "m", "prompt_tokens": 100, "output_tokens": 50, "total_tokens": 150, "estimated_cost_usd": 0.001}) + "\n" +
            json.dumps({"call_site": "a", "model": "m", "prompt_tokens": 200, "output_tokens": 100, "total_tokens": 300, "estimated_cost_usd": 0.002}) + "\n" +
            json.dumps({"call_site": "b", "model": "m", "prompt_tokens": 10, "output_tokens": 5, "total_tokens": 15, "estimated_cost_usd": 0.0001}) + "\n",
            encoding="utf-8",
        )
        result = gemini_usage.aggregate_usage(str(log_file))
        assert result["total_calls"] == 3
        assert result["total_tokens"] == 465
        assert result["total_cost_usd"] == pytest.approx(0.0031)
        assert result["all_costs_known"] is True
        by_site = {r["call_site"]: r for r in result["by_call_site"]}
        assert by_site["a"]["calls"] == 2
        assert by_site["a"]["tokens"] == 450
        assert by_site["a"]["cost_usd"] == pytest.approx(0.003)
        assert by_site["b"]["cost_usd"] == pytest.approx(0.0001)

    def test_unknown_cost_entries_flag_all_costs_known_false(self, tmp_path):
        log_file = tmp_path / "usage.jsonl"
        log_file.write_text(
            json.dumps({"call_site": "a", "model": "m", "prompt_tokens": 100, "output_tokens": 50, "total_tokens": 150, "estimated_cost_usd": 0.001}) + "\n" +
            json.dumps({"call_site": "b", "model": "bilinmeyen", "prompt_tokens": 10, "output_tokens": 5, "total_tokens": 15, "estimated_cost_usd": None}) + "\n",
            encoding="utf-8",
        )
        result = gemini_usage.aggregate_usage(str(log_file))
        assert result["all_costs_known"] is False
        assert result["total_cost_usd"] == pytest.approx(0.001)
        by_site = {r["call_site"]: r for r in result["by_call_site"]}
        assert by_site["b"]["cost_usd"] is None
        assert by_site["b"]["calls"] == 1

    def test_skips_blank_lines(self, tmp_path):
        log_file = tmp_path / "usage.jsonl"
        log_file.write_text(
            json.dumps({"call_site": "a", "model": "m", "prompt_tokens": 1, "output_tokens": 1, "total_tokens": 2, "estimated_cost_usd": 0.0}) + "\n\n",
            encoding="utf-8",
        )
        result = gemini_usage.aggregate_usage(str(log_file))
        assert result["total_calls"] == 1
