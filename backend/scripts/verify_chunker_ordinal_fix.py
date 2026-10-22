"""chunker.py::split_into_sentences'in rakam+nokta duzeltmesini
gercek dosyalar uzerinde dogrulayan, bir kereye mahsus dogrulama scripti.
docs/superpowers/specs/2026-08-01-chunker-ordinal-split-fix-design.md.
artifacts/ veya Neo4j/Qdrant'a hicbir yazma yapmaz - salt okunur dogrulama."""

import os
import re

from sentence_transformers import SentenceTransformer

from backend.pipeline import chunker
from backend.shared.paths import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT

print("=== 1) Legal test dokumaninda yalitilmis rakam+nokta parcasi kalmamis mi ===")
legal_path = os.path.join(BASE_DIR, "sources_legal_test", "banking_authorizations.txt")
legal_text = chunker.load_document(legal_path)
legal_sentences = chunker.split_into_sentences(legal_text)

bare_ordinal_fragments = [s for s in legal_sentences if re.fullmatch(r"\d{1,3}\.", s.strip())]
if bare_ordinal_fragments:
    print(f"HATA: hala yalitilmis rakam+nokta parcasi var: {bare_ordinal_fragments}")
else:
    print(f"OK: {len(legal_sentences)} cumle, hicbiri yalitilmis rakam+nokta parcasi degil.")

target = next(
    (s for s in legal_sentences if "1. Grup C Sınıfı ile 2. Grup C Sınıfı yetkilileri" in s and "50.000" in s),
    None,
)
if target is None:
    print("HATA: beklenen tam cumle bulunamadi - hala bolunmus olabilir.")
else:
    print(f"OK: hedef cumle butun halde: {target!r}")

print("\n=== 2) Ana corpus'un chunk sayilari degismemis mi (bu 3 dosyada ilgili kalip yok) ===")
print("Model yukleniyor...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

expected_chunk_counts = {
    "mimari_v1.txt": 4,
    "wikipedia_deadlock.txt": 1,
    "wikipedia_postgresql.txt": 11,
}
for filename, expected_count in expected_chunk_counts.items():
    path = os.path.join(BASE_DIR, "sources", filename)
    chunks = chunker.chunk_document(path, model)
    status = "OK" if len(chunks) == expected_count else "FARK VAR"
    print(f"{status}: {filename} -> {len(chunks)} chunk (beklenen: {expected_count})")

print("\nDogrulama tamamlandi.")
