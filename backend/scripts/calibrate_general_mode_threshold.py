"""GENERAL_MODE_RELEVANCE_THRESHOLD'u canli olcmek icin bir seferlik betik.
eval/golden_retrieval.jsonl (cevaplanmali) + eval/golden_refusal.jsonl (reddetmeli)
sorularinin gercek top-1 rerank skorlarini yazdirir - docs/superpowers/specs/
2026-08-07-retrieval-confidence-gate-design.md #4. Qdrant'in ayakta olmasi gerekir
(docker compose up -d). Kalici bir pipeline adimi degil, tek seferlik kalibrasyon araci -
backend/scripts/'un mevcut konvansiyonuyla ayni (bkz. verify_chunker_ordinal_fix.py)."""

import json
import os

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.query import query_engine as qe
from backend.shared.paths import PROJECT_ROOT


def load_questions(path):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line)["question"])
    return questions


def top_score_for(question, qdrant_client, embedding_model, cross_encoder, chunks_by_id):
    seed_chunk_ids = qe.vector_seed_search(qdrant_client, embedding_model, question, 20)
    evidence_pool = [chunks_by_id[cid] for cid in seed_chunk_ids if cid in chunks_by_id]
    ranked = qe.rerank_evidence(cross_encoder, question, evidence_pool)
    return ranked[0][1] if ranked else None


if __name__ == "__main__":
    BASE_DIR = PROJECT_ROOT
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    GOLDEN_RETRIEVAL_FILE = os.path.join(BASE_DIR, "eval", "golden_retrieval.jsonl")
    GOLDEN_REFUSAL_FILE = os.path.join(BASE_DIR, "eval", "golden_refusal.jsonl")

    chunks_by_id = qe.load_chunks_by_id(CHUNKS_FILE)
    qdrant_client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

    print("Modeller yukleniyor...")
    embedding_model = SentenceTransformer(qe.EMBEDDING_MODEL_NAME)
    cross_encoder = CrossEncoder(qe.CROSS_ENCODER_MODEL_NAME)

    print("\n=== Cevaplanmali (golden_retrieval.jsonl) ===")
    answerable_scores = []
    for record in [json.loads(l) for l in open(GOLDEN_RETRIEVAL_FILE, encoding="utf-8") if l.strip()]:
        score = top_score_for(record["question"], qdrant_client, embedding_model, cross_encoder, chunks_by_id)
        answerable_scores.append(score)
        print(f"  {score:.3f}  {record['question']}")

    print("\n=== Reddetmeli (golden_refusal.jsonl) ===")
    refusal_scores = []
    for question in load_questions(GOLDEN_REFUSAL_FILE):
        score = top_score_for(question, qdrant_client, embedding_model, cross_encoder, chunks_by_id)
        refusal_scores.append(score)
        print(f"  {score:.3f}  {question}")

    print(f"\nCevaplanmali min skor: {min(answerable_scores):.3f}")
    print(f"Reddetmeli max skor:   {max(refusal_scores):.3f}")
    if min(answerable_scores) > max(refusal_scores):
        print("Bosluk var - iki deger arasinda bir THRESHOLD secilebilir.")
    else:
        print("UYARI: gruplar cakisiyor - tek bir sabit esik ikisini net ayiramaz, kontrollore bildir.")
