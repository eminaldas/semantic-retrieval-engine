"""resolve_entity_id'nin entity sayisi arttikca nasil davrandigini olcen,
izole bir mikro-benchmark. Gercek Gemini pipeline'ini gerektirmez - sentetik
entity'ler kullanir, hepsi SCALE_TEST_ENTITY tipinde (test sonunda kolayca
temizlenir). docs/superpowers/specs/2026-07-29-scale-test-design.md."""

import json
import os
import time
import uuid

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from backend.pipeline import step3_graph_writer as gw
from backend.shared.paths import PROJECT_ROOT

SCALE_TEST_TYPE = "SCALE_TEST_ENTITY"
MEASURE_POINTS = [50, 100, 200, 400, 800]


def resolve_and_write(tx, name, qdrant_client, embedding_model):
    """Gercek write_extraction_record'un entity kismini taklit eder: once coz, sonra yaz."""
    entity_id = gw.resolve_entity_id(name, SCALE_TEST_TYPE, [], tx, qdrant_client, embedding_model)
    _, normalized_name = gw.normalize_entity_id(name, SCALE_TEST_TYPE)
    entity = {"name": name, "type": SCALE_TEST_TYPE, "aliases": [], "confidence": 0.9}
    gw.upsert_entity(tx, entity, entity_id, normalized_name)
    return entity_id


if __name__ == "__main__":
    BASE_DIR = PROJECT_ROOT
    RESULTS_FILE = os.path.join(BASE_DIR, "artifacts", "scale_test_results.json")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD")),
    )
    qdrant_client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

    print("Embedding modeli yukleniyor...")
    embedding_model = SentenceTransformer(gw.EMBEDDING_MODEL_NAME)

    max_n = max(MEASURE_POINTS)
    measurements = []

    with driver.session() as session:
        session.run("MATCH (e:Entity {type: $type}) DETACH DELETE e", type=SCALE_TEST_TYPE)

        for i in range(1, max_n + 1):
            name = f"ScaleTestEntity_{i}_{uuid.uuid4().hex[:8]}"
            start = time.perf_counter()
            session.execute_write(resolve_and_write, name, qdrant_client, embedding_model)
            elapsed = time.perf_counter() - start
            if i in MEASURE_POINTS:
                measurements.append({"existing_entity_count": i, "resolve_and_write_seconds": round(elapsed, 4)})
                print(f"  entity_count={i}: {elapsed:.4f}s")

        session.run("MATCH (e:Entity {type: $type}) DETACH DELETE e", type=SCALE_TEST_TYPE)
        print(f"Temizlendi: {SCALE_TEST_TYPE} tipindeki tum test entity'leri silindi.")

    driver.close()

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    existing = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing["entity_resolution_benchmark"] = measurements
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\nBASARILI! Sonuclar {RESULTS_FILE}'e yazildi.")
