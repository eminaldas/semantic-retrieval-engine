"""Buyuk test corpus'unda vektor-retrieval kalitesini olcen, tek seferlik bir
arac. Mevcut eval_retrieval.py'nin path-match kismi MODE_B'ye ozel oldugu
icin burada kullanilmiyor - sadece qe.vector_seed_search yeniden kullaniliyor.
docs/superpowers/specs/2026-07-29-scale-test-design.md."""

import json


def load_golden_set(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_hit(expected_source_id, seed_chunk_ids, chunks_by_id):
    """Beklenen source_id, dondurulen chunk_id'lerin herhangi birinde var mi?"""
    retrieved_source_ids = {chunks_by_id[cid]["source_id"] for cid in seed_chunk_ids if cid in chunks_by_id}
    return expected_source_id in retrieved_source_ids


def aggregate_hit_rate(hits):
    if not hits:
        return 0.0
    return sum(hits) / len(hits)


import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from backend.query import step5_query_engine as qe
from backend.shared.paths import PROJECT_ROOT


if __name__ == "__main__":
    BASE_DIR = PROJECT_ROOT
    GOLDEN_SET_FILE = os.path.join(BASE_DIR, "eval", "golden_scale_retrieval.jsonl")
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    RESULTS_FILE = os.path.join(BASE_DIR, "artifacts", "scale_test_results.json")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    golden_set = load_golden_set(GOLDEN_SET_FILE)
    chunks_by_id = qe.load_chunks_by_id(CHUNKS_FILE)
    qdrant_client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

    print("Embedding modeli yukleniyor...")
    embedding_model = SentenceTransformer(qe.EMBEDDING_MODEL_NAME)

    hits = []
    per_question = []
    for record in golden_set:
        seed_chunk_ids = qe.vector_seed_search(qdrant_client, embedding_model, record["question"], 20)
        hit = compute_hit(record["expected_source_id"], seed_chunk_ids, chunks_by_id)
        hits.append(hit)
        per_question.append({"question": record["question"], "expected_source_id": record["expected_source_id"], "hit": hit})
        mark = "OK" if hit else "FAIL"
        print(f"  [{mark}] {record['question']}")

    hit_rate = aggregate_hit_rate(hits)
    print(f"\n=== Ozet ({len(hits)} soru) ===")
    print(f"  hit_rate: {hit_rate:.2f}")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    existing = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing["retrieval_quality"] = {"hit_rate": hit_rate, "per_question": per_question}
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\nBASARILI! Sonuclar {RESULTS_FILE}'e yazildi.")
