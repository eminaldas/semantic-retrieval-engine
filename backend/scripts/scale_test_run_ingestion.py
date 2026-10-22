"""sources_scale_test/'i gercek pipeline'dan (step1->2->3->4) gecirip her
adimin suresini olcer. step6 (self-correction) bu olcumun kapsami disinda -
tum chunk'lar ilk kez eklendigi icin candidate bulmaz, olculecek bir sinyal yok.
docs/superpowers/specs/2026-07-29-scale-test-design.md."""

import json
import os
import subprocess
import sys
import time

from backend.shared.paths import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
SCALE_TEST_SOURCES_DIR = os.path.join(BASE_DIR, "sources_scale_test")
RESULTS_FILE = os.path.join(BASE_DIR, "artifacts", "scale_test_results.json")

if __name__ == "__main__":
    test_files = sorted(
        os.path.join(SCALE_TEST_SOURCES_DIR, f)
        for f in os.listdir(SCALE_TEST_SOURCES_DIR)
        if os.path.isfile(os.path.join(SCALE_TEST_SOURCES_DIR, f))
    )
    print(f"{len(test_files)} test dosyasi bulundu.", flush=True)

    timings = {}

    start = time.time()
    for i, file_path in enumerate(test_files, start=1):
        result = subprocess.run([sys.executable, "-m", "backend.pipeline.chunker", file_path], cwd=BASE_DIR)
        if result.returncode != 0:
            print(f"UYARI: {file_path} step1'de basarisiz oldu, atlaniyor.", flush=True)
        print(f"  [{i}/{len(test_files)}] step1 tamamlandi: {os.path.basename(file_path)}", flush=True)
    timings["chunker_total_seconds"] = time.time() - start

    for step_module in ["backend.pipeline.extractor", "backend.pipeline.graph_writer", "backend.pipeline.vector_writer"]:
        print(f"\n=== {step_module} calistiriliyor ===", flush=True)
        start = time.time()
        result = subprocess.run([sys.executable, "-m", step_module], cwd=BASE_DIR)
        elapsed = time.time() - start
        timings[f"{step_module.replace('.py', '')}_total_seconds"] = elapsed
        print(f"{step_module}: {elapsed:.1f}s (exit code {result.returncode})", flush=True)
        if result.returncode != 0:
            print(f"UYARI: {step_module} basarisiz oldu (exit code {result.returncode}). Devam ediliyor.", flush=True)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    existing = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing["ingestion_timings"] = timings
    existing["test_document_count"] = len(test_files)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\nZamanlama sonuclari {RESULTS_FILE}'e yazildi.")
