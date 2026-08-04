"""sources_scale_test/'in ana sisteme yazdigi tum veriyi guvenli sekilde
temizler. Sadece bu teste ozel source_id'leri siler - orijinal 3 dokumana
dokunmaz. docs/superpowers/specs/2026-07-29-scale-test-design.md."""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

import manage_sources

ORIGINAL_SOURCE_IDS = {"mimari_v1.txt", "wikipedia_deadlock.txt", "wikipedia_postgresql.txt"}

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    EXTRACTION_FILE = os.path.join(BASE_DIR, "artifacts", "extraction.jsonl")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    neo4j_driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD")),
    )
    qdrant_client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

    chunks = manage_sources.load_jsonl(CHUNKS_FILE)
    all_source_ids = {c["source_id"] for c in chunks}
    test_source_ids = all_source_ids - ORIGINAL_SOURCE_IDS

    print(f"{len(test_source_ids)} test kaynagi silinecek.")

    for source_id in sorted(test_source_ids):
        result = manage_sources.remove_source(source_id, CHUNKS_FILE, EXTRACTION_FILE, neo4j_driver, qdrant_client)
        if result:
            print(f"  silindi: {source_id} ({result['removed_chunk_count']} chunk, {result['removed_entity_count']} munhasir entity)")

    neo4j_driver.close()
    print("\nBASARILI! Temizlik tamamlandi.")
