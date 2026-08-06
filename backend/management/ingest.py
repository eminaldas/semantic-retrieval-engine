"""Tum ingestion hattini (step1->2->3->4->6) tek komutla zincirler.
docs/superpowers/specs/2026-07-26-auto-ingestion-design.md.

Bu bir 'stepN_' pipeline modulu degil - eval_retrieval.py/dashboard.py gibi
talep-uzerine calisan bir orkestrasyon araci. Her adimi subprocess ile
calistirir - mevcut script'lerin __main__'larini yeniden yazmaz/import etmez.

Adimlar modul-yolu ile (-m) cagriliyor, dosya-yolu ile degil - boylece
backend/ paketinin ic import'lari (orn. self_correction'in graph_writer'dan
RELATION_TYPES almasi) subprocess icinde de dogru calisiyor."""

import subprocess
import sys

from backend.shared.paths import PROJECT_ROOT

STEPS = [
    "backend.pipeline.chunker",
    "backend.pipeline.extractor",
    "backend.pipeline.graph_writer",
    "backend.pipeline.vector_writer",
    "backend.pipeline.self_correction",
]

if __name__ == "__main__":
    for step_module in STEPS:
        print(f"\n=== {step_module} calistiriliyor ===", flush=True)
        result = subprocess.run([sys.executable, "-m", step_module], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"\nHATA: {step_module} basarisiz oldu (exit code {result.returncode}). Ingestion durduruldu.", flush=True)
            sys.exit(result.returncode)

    print("\nBASARILI! Ingestion tamamlandi (step1->2->3->4->6).", flush=True)
