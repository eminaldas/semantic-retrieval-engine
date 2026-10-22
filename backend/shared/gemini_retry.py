"""Gemini API cagrilarini saran ortak retry/backoff + proaktif hiz sinirlama.
extractor.py, entity_linker.py, query_engine.py,
conversation_memory.py, self_correction.py, eval_hallucination.py
tarafindan kullanilir. docs/superpowers/specs/2026-08-02-gemini-rate-limit-design.md."""

import random
import time

from google.genai import errors

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 30.0
JITTER_SECONDS = 1.0
RETRY_AFTER_CAP_SECONDS = 60.0

_last_call_at = {}


def _is_retryable(exc):
    """429 (rate limit) ve 5xx (sunucu hatasi) + APIError disi hatalar (network/DNS/
    timeout - genellikle gecici) yeniden denenir; diger 4xx hatalari (400/401/403 gibi
    kalici hatalar) denenmez."""
    if isinstance(exc, errors.APIError):
        return exc.code == 429 or exc.code >= 500
    return True


def _retry_after_seconds(exc):
    """429 yanitinda Retry-After header'i varsa onu kullanir (60sn ile sinirlanir)."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        return min(float(value), RETRY_AFTER_CAP_SECONDS)
    except (TypeError, ValueError):
        return None


def call_with_retry(fn, min_interval_seconds=0.0, pacing_key="default"):
    """fn: parametresiz bir cagrilabilir (genellikle bir closure), gercek
    client.models.generate_content(...) + response parse islemini yapar.

    min_interval_seconds > 0 ise, ayni pacing_key ile yapilan bir onceki cagridan
    bu yana en az bu kadar sure gecmesini bekler (proaktif hiz sinirlama - sadece
    extractor.py'nin toplu dongusu kullanir; diger tekil-cagri siteleri
    varsayilan 0.0 ile hicbir gecikme eklemez).

    Son denemeden sonra hala basarisizsa RuntimeError firlatir (mevcut
    'except RuntimeError' yakalama noktalariyla geriye donuk uyumlu)."""
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if min_interval_seconds > 0:
            last_call = _last_call_at.get(pacing_key)
            if last_call is not None:
                remaining = min_interval_seconds - (time.monotonic() - last_call)
                if remaining > 0:
                    time.sleep(remaining)
            _last_call_at[pacing_key] = time.monotonic()

        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_ATTEMPTS or not _is_retryable(exc):
                raise RuntimeError(f"{attempt} denemede basarisiz: {last_error}") from last_error
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = min(BASE_DELAY_SECONDS * (2 ** (attempt - 1)), MAX_DELAY_SECONDS)
                delay += random.uniform(0, JITTER_SECONDS)
            time.sleep(delay)
