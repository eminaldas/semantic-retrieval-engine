"""Buyuk corpus'ta /api/query (general mode) latency'sini olcen, tek seferlik
bir arac. dashboard.py'nin calisir durumda olmasini gerektirir (docker compose
up -d + python dashboard.py). Task 4'un golden set sorularini yeniden kullanir.
docs/superpowers/specs/2026-07-29-scale-test-design.md."""

import json
import os
import time

import requests

from backend.shared.paths import PROJECT_ROOT

if __name__ == "__main__":
    BASE_DIR = PROJECT_ROOT
    GOLDEN_SET_FILE = os.path.join(BASE_DIR, "eval", "golden_scale_retrieval.jsonl")
    RESULTS_FILE = os.path.join(BASE_DIR, "artifacts", "scale_test_results.json")

    questions = []
    with open(GOLDEN_SET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line)["question"])

    latencies = []
    for question in questions:
        start = time.perf_counter()
        response = requests.post(
            "http://localhost:8000/api/query",
            json={"question": question},
            timeout=60,
        )
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        print(f"  {elapsed:.2f}s (status={response.status_code}): {question}")

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    print(f"\n=== Ozet ({len(latencies)} soru) ===")
    print(f"  ortalama: {avg_latency:.2f}s")
    print(f"  en yavas: {max_latency:.2f}s")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    existing = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing["query_latency"] = {"average_seconds": avg_latency, "max_seconds": max_latency, "samples": latencies}
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\nBASARILI! Sonuclar {RESULTS_FILE}'e yazildi.")
