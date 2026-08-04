"""Gemini API cagrilarinin token kullanimi/tahmini maliyetini izler.
gemini_retry.py ile ayni kategori (paylasilan yardimci modul, pipeline adimi degil).
docs/superpowers/specs/2026-08-04-gemini-cost-tracking-design.md."""

import json
import os
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_PATH = os.path.join(BASE_DIR, "artifacts", "gemini_usage_log.jsonl")

# Kaynak: https://ai.google.dev/gemini-api/docs/pricing (2026-08-04'te dogrulandi,
# standart/pay-as-you-go metin fiyatlandirmasi, $/1M token). Zamanla eskiyebilir -
# yeni bir model kullanilirsa veya fiyatlar degisirse burasi elle guncellenmeli.
# Her giris bir tier listesi: (prompt_token esigi ya da None=sinirsiz, girdi $, cikti $).
# Tier'lar sirali taranir, prompt_tokens esige esit/altindaysa o tier kullanilir.
PRICING = {
    "gemini-flash-lite-latest": [(None, 0.10, 0.40)],
    "gemini-flash-latest": [(None, 0.30, 2.50)],
    "gemini-pro-latest": [(200_000, 1.25, 10.00), (None, 2.50, 15.00)],
}


def _estimate_cost(model, prompt_tokens, output_tokens):
    tiers = PRICING.get(model)
    if tiers is None or prompt_tokens is None or output_tokens is None:
        return None
    for threshold, input_price, output_price in tiers:
        if threshold is None or prompt_tokens <= threshold:
            return (prompt_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
    return None


def log_usage(call_site, model, response, log_path=DEFAULT_LOG_PATH):
    """response.usage_metadata'dan token sayilarini cikarip log_path'e bir satir ekler.
    usage_metadata eksik/bozuksa sessizce hicbir sey yapmaz - asil Gemini cagrisini
    asla kesintiye ugratmaz."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    prompt_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    total_tokens = usage.total_token_count
    if prompt_tokens is None or output_tokens is None or total_tokens is None:
        return

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "call_site": call_site,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": _estimate_cost(model, prompt_tokens, output_tokens),
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def aggregate_usage(log_path=DEFAULT_LOG_PATH):
    """log_path'i okuyup call_site-bazli kirilim + toplam dondurur (saf fonksiyon,
    sadece dosya G/C). Dosya yoksa sifirlanmis bos bir sonuc doner."""
    if not os.path.exists(log_path):
        return {"total_calls": 0, "total_tokens": 0, "total_cost_usd": 0.0, "all_costs_known": True, "by_call_site": []}

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    sites = {}
    for e in entries:
        site = sites.setdefault(e["call_site"], {"calls": 0, "tokens": 0, "cost_usd": 0.0, "unknown": False})
        site["calls"] += 1
        site["tokens"] += e["total_tokens"]
        if e["estimated_cost_usd"] is None:
            site["unknown"] = True
        else:
            site["cost_usd"] += e["estimated_cost_usd"]

    all_costs_known = not any(s["unknown"] for s in sites.values())
    total_cost_usd = sum(s["cost_usd"] for s in sites.values() if not s["unknown"])

    by_call_site = [
        {
            "call_site": call_site,
            "calls": s["calls"],
            "tokens": s["tokens"],
            "cost_usd": None if s["unknown"] else s["cost_usd"],
        }
        for call_site, s in sorted(sites.items())
    ]

    return {
        "total_calls": len(entries),
        "total_tokens": sum(e["total_tokens"] for e in entries),
        "total_cost_usd": total_cost_usd,
        "all_costs_known": all_costs_known,
        "by_call_site": by_call_site,
    }
