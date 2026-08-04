import time

import pytest
from google.genai import errors

from backend.shared import gemini_retry


class FakeResponse:
    def __init__(self, headers=None):
        self.headers = headers or {}


class TestCallWithRetry:
    def test_succeeds_on_first_try(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = gemini_retry.call_with_retry(fn, pacing_key="test_first_try")
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_429_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        attempts = []

        def fn():
            attempts.append(1)
            if len(attempts) < 3:
                raise errors.ClientError(429, {"error": {"message": "rate limited"}})
            return "ok"

        result = gemini_retry.call_with_retry(fn, pacing_key="test_429")
        assert result == "ok"
        assert len(attempts) == 3

    def test_retries_on_server_error(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        attempts = []

        def fn():
            attempts.append(1)
            if len(attempts) < 2:
                raise errors.ServerError(503, {"error": {"message": "unavailable"}})
            return "ok"

        result = gemini_retry.call_with_retry(fn, pacing_key="test_5xx")
        assert result == "ok"
        assert len(attempts) == 2

    def test_retries_on_network_error(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        attempts = []

        def fn():
            attempts.append(1)
            if len(attempts) < 2:
                raise ConnectionError("getaddrinfo failed")
            return "ok"

        result = gemini_retry.call_with_retry(fn, pacing_key="test_network")
        assert result == "ok"
        assert len(attempts) == 2

    def test_permanent_client_error_fails_immediately(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        attempts = []

        def fn():
            attempts.append(1)
            raise errors.ClientError(400, {"error": {"message": "bad request"}})

        with pytest.raises(RuntimeError):
            gemini_retry.call_with_retry(fn, pacing_key="test_400")
        assert len(attempts) == 1

    def test_exhausts_all_attempts_then_raises(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        attempts = []

        def fn():
            attempts.append(1)
            raise errors.ClientError(429, {"error": {"message": "rate limited"}})

        with pytest.raises(RuntimeError):
            gemini_retry.call_with_retry(fn, pacing_key="test_exhaust")
        assert len(attempts) == gemini_retry.MAX_ATTEMPTS

    def test_honors_retry_after_header(self, monkeypatch):
        slept = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        attempts = []

        def fn():
            attempts.append(1)
            if len(attempts) < 2:
                response = FakeResponse(headers={"Retry-After": "3"})
                raise errors.ClientError(429, {"error": {"message": "rate limited"}}, response=response)
            return "ok"

        result = gemini_retry.call_with_retry(fn, pacing_key="test_retry_after")
        assert result == "ok"
        assert slept == [3.0]


class TestMinIntervalPacing:
    def test_enforces_minimum_interval_between_calls(self):
        gemini_retry.call_with_retry(lambda: "first", min_interval_seconds=0.1, pacing_key="test_pacing")
        start = time.monotonic()
        gemini_retry.call_with_retry(lambda: "second", min_interval_seconds=0.1, pacing_key="test_pacing")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.09

    def test_different_pacing_keys_do_not_interfere(self):
        gemini_retry.call_with_retry(lambda: "a", min_interval_seconds=1.0, pacing_key="test_key_a")
        start = time.monotonic()
        gemini_retry.call_with_retry(lambda: "b", min_interval_seconds=1.0, pacing_key="test_key_b")
        elapsed = time.monotonic() - start
        assert elapsed < 0.5
