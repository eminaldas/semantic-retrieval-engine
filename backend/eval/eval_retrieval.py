"""Retrieval hit-rate degerlendirme araci - golden bir soru setiyle
step5_query_engine.py'nin graf+vektor+rerank cikisini olcer.
docs/superpowers/specs/2026-07-24-evaluation-harness-design.md'deki tasarima gore.

Not: bu bir 'stepN_' pipeline modulu degil - istendiginde calistirilan
bir kalite-olcum araci (dashboard.py gibi ayri duruyor). Gemini'ye hic
dokunmaz, sadece retrieval kalitesini olcer."""

import json


def load_golden_set(path):
    """eval/golden_retrieval.jsonl'u okur, bos satirlari atlar."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_metrics(record, candidate_paths, ranked_evidence):
    """Tek bir golden kayit icin path_match/top1_hit/recall hesaplar (saf fonksiyon,
    Neo4j/Qdrant'a hic dokunmaz - cagiran taraf candidate_paths/ranked_evidence'i onceden hesaplar)."""
    path_found = len(candidate_paths) > 0
    path_match = path_found == record["expect_path"]

    expected = set(record["expected_chunk_ids"])
    top_ids = [chunk["chunk_id"] for chunk, _score in ranked_evidence]

    top1_hit = bool(top_ids) and top_ids[0] in expected
    found_count = len(expected & set(top_ids))
    recall = found_count / len(expected)

    return {
        "question": record["question"],
        "cause_entity_id": record["cause_entity_id"],
        "target_entity_id": record["target_entity_id"],
        "path_match": path_match,
        "top1_hit": top1_hit,
        "recall": recall,
    }


def aggregate_results(results):
    """Soru-bazli sonuc listesinden toplu orani hesaplar. Bos liste (orn. bos/yanlis
    yola isaret eden --golden-set) icin sifira bolme hatasi yerine 0.0 doner."""
    n = len(results)
    if n == 0:
        return {"path_match_rate": 0.0, "top1_hit_rate": 0.0, "mean_recall": 0.0}

    return {
        "path_match_rate": sum(r["path_match"] for r in results) / n,
        "top1_hit_rate": sum(r["top1_hit"] for r in results) / n,
        "mean_recall": sum(r["recall"] for r in results) / n,
    }


import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.query import step5_query_engine as qe
from backend.shared.paths import PROJECT_ROOT


def evaluate_record(record, session, qdrant_client, embedding_model, cross_encoder, chunks_by_id, max_hops, seed_topk, top_n):
    """Tek bir golden kayit icin gercek path+vektor+rerank calistirir, sonra compute_metrics'e devreder."""
    candidate_paths = qe.find_candidate_paths(session, record["cause_entity_id"], record["target_entity_id"], max_hops)
    seed_chunk_ids = qe.vector_seed_search(qdrant_client, embedding_model, record["question"], seed_topk)
    evidence_pool = qe.build_evidence_pool(candidate_paths, seed_chunk_ids, chunks_by_id)
    ranked_evidence = qe.rerank_evidence(cross_encoder, record["question"], evidence_pool, top_n)
    return compute_metrics(record, candidate_paths, ranked_evidence)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval hit-rate degerlendirme araci")
    parser.add_argument("--golden-set", default=None, help="golden set jsonl yolu (varsayilan: eval/golden_retrieval.jsonl)")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--seed-topk", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    BASE_DIR = PROJECT_ROOT
    GOLDEN_SET_FILE = args.golden_set or os.path.join(BASE_DIR, "eval", "golden_retrieval.jsonl")
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    LOG_FILE = os.path.join(BASE_DIR, "artifacts", "eval_retrieval_log.jsonl")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    golden_set = load_golden_set(GOLDEN_SET_FILE)
    chunks_by_id = qe.load_chunks_by_id(CHUNKS_FILE)

    neo4j_driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD")),
    )
    qdrant_client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

    print("Modeller yukleniyor...")
    embedding_model = SentenceTransformer(qe.EMBEDDING_MODEL_NAME)
    cross_encoder = CrossEncoder(qe.CROSS_ENCODER_MODEL_NAME)

    results = []
    with neo4j_driver.session() as session:
        for record in golden_set:
            result = evaluate_record(
                record, session, qdrant_client, embedding_model, cross_encoder,
                chunks_by_id, args.max_hops, args.seed_topk, args.top_n,
            )
            results.append(result)
            path_mark = "OK" if result["path_match"] else "FAIL"
            top1_mark = "OK" if result["top1_hit"] else "FAIL"
            print(f"  [path_match={path_mark}] [top1_hit={top1_mark}] recall={result['recall']:.2f}  {result['question']}")
    neo4j_driver.close()

    aggregate = aggregate_results(results)
    print(f"\n=== Ozet ({len(results)} soru) ===")
    print(f"  path_match_rate: {aggregate['path_match_rate']:.2f}")
    print(f"  top1_hit_rate:   {aggregate['top1_hit_rate']:.2f}")
    print(f"  mean_recall:     {aggregate['mean_recall']:.2f}")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate": aggregate,
        "per_question": results,
    }
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(f"\nBASARILI! Log: {LOG_FILE}")
