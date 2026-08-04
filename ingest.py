"""Tum ingestion hattini (step1->2->3->4->6) tek komutla zincirler.
docs/superpowers/specs/2026-07-26-auto-ingestion-design.md.

Bu bir 'stepN_' pipeline modulu degil - eval_retrieval.py/dashboard.py gibi
talep-uzerine calisan bir orkestrasyon araci. Her adimi subprocess ile
calistirir - mevcut script'lerin __main__'larini yeniden yazmaz/import etmez."""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    "step1_chunker.py",
    "step2_extractor.py",
    "step3_graph_writer.py",
    "step4_vector_writer.py",
    "step6_self_correction.py",
]

if __name__ == "__main__":
    for step in STEPS:
        step_path = os.path.join(BASE_DIR, step)
        print(f"\n=== {step} calistiriliyor ===", flush=True)
        result = subprocess.run([sys.executable, step_path], cwd=BASE_DIR)
        if result.returncode != 0:
            print(f"\nHATA: {step} basarisiz oldu (exit code {result.returncode}). Ingestion durduruldu.", flush=True)
            sys.exit(result.returncode)

    print("\nBASARILI! Ingestion tamamlandi (step1->2->3->4->6).", flush=True)
