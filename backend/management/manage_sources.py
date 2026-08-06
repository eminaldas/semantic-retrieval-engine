"""Kaynak/dokuman yonetimi: list/remove/update alt-komutlari.
docs/superpowers/specs/2026-07-26-source-management-design.md.

Bu bir 'stepN_' pipeline modulu degil - ingest.py/eval_*.py gibi talep-uzerine
calisan bir arac. Kendi kucuk load_jsonl/write_jsonl kopyasina sahip - diger
eval_*.py araclarinin ayni deseniyle tutarli (birbirinden import etmiyorlar)."""

import json
import subprocess
import sys

from backend.pipeline import graph_writer
from backend.pipeline import vector_writer


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def group_chunks_by_source(chunks):
    """chunks.jsonl kayitlarini source_id'ye gore gruplar - list komutunun temel veri kaynagi."""
    groups = {}
    for chunk in chunks:
        source_id = chunk["source_id"]
        if source_id not in groups:
            groups[source_id] = {"chunk_count": 0, "ingested_at": chunk.get("ingested_at"), "chunk_ids": []}
        groups[source_id]["chunk_count"] += 1
        groups[source_id]["chunk_ids"].append(chunk["chunk_id"])
    return groups


def group_extraction_by_source(chunks, extraction_records):
    """extraction.jsonl kayitlarini (chunk_id uzerinden) source_id'ye gore gruplar,
    o dokumanin katkidigi ham entity/relation sayisini hesaplar (Neo4j'deki
    nihai birlesmis sayi degil - sadece bu dokumanin urettigi ham miktar)."""
    chunk_id_to_source = {c["chunk_id"]: c["source_id"] for c in chunks}
    groups = {}
    for record in extraction_records:
        source_id = chunk_id_to_source.get(record["chunk_id"])
        if source_id is None:
            continue
        if source_id not in groups:
            groups[source_id] = {"entity_count": 0, "relation_count": 0}
        groups[source_id]["entity_count"] += len(record["entities"])
        groups[source_id]["relation_count"] += len(record["relations"])
    return groups


def chunk_ids_for_source(chunks, source_id):
    """Bir source_id'ye ait tum chunk_id'leri dondurur - remove komutunun hedef listesi."""
    return [c["chunk_id"] for c in chunks if c["source_id"] == source_id]


def prune_evidence_chunk_ids(evidence_ids, removed_ids):
    """Bir iliskinin evidence_chunk_ids'inden silinen dokumana ait olanlari cikarir.
    Sonuc bos kalirsa (tum kaniti silinen dokumandan geliyorsa) cagiran taraf
    iliskiyi silmeli - 'kanitsiz kenar olamaz' ilkesi."""
    removed_set = set(removed_ids)
    return [cid for cid in evidence_ids if cid not in removed_set]


def cmd_list(chunks_file, extraction_file):
    chunks = load_jsonl(chunks_file)
    extraction = load_jsonl(extraction_file)
    chunk_groups = group_chunks_by_source(chunks)
    extraction_groups = group_extraction_by_source(chunks, extraction)

    if not chunk_groups:
        print("Sistemde henuz hicbir dokuman yok.")
        return

    for source_id in sorted(chunk_groups):
        c = chunk_groups[source_id]
        e = extraction_groups.get(source_id, {"entity_count": 0, "relation_count": 0})
        ingested = c["ingested_at"] or "bilinmiyor"
        print(
            f"{source_id}  {c['chunk_count']} chunk  {e['entity_count']} entity  "
            f"{e['relation_count']} relation  ingested: {ingested}"
        )


def remove_source_from_graph(session, chunk_ids):
    """Neo4j'den bir kaynagin chunk/entity/relation verisini guvenli sekilde siler.
    Sira: (1) Chunk'lar silinir (MENTIONED_IN kenarlari otomatik gider),
    (2) etkilenen iliskilerin evidence_chunk_ids'i budanir (prune_evidence_chunk_ids,
    saf fonksiyon) - bos kalirsa iliski silinir, (3) artik hicbir MENTIONED_IN'i
    kalmayan (yetim) Entity'ler silinir - baska dokumanda gecenlere dokunulmaz.
    Silinen entity_id'leri (Qdrant temizligi icin cagiran tarafa) dondurur."""
    session.run("MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids DETACH DELETE c", chunk_ids=chunk_ids)

    result = session.run(
        """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE ANY(cid IN r.evidence_chunk_ids WHERE cid IN $chunk_ids)
        RETURN elementId(r) AS rel_id, r.evidence_chunk_ids AS evidence_chunk_ids
        """,
        chunk_ids=chunk_ids,
    )
    affected_relations = [(record["rel_id"], record["evidence_chunk_ids"]) for record in result]

    for rel_id, evidence_ids in affected_relations:
        pruned = prune_evidence_chunk_ids(evidence_ids, chunk_ids)
        if pruned:
            session.run(
                "MATCH ()-[r]->() WHERE elementId(r) = $rel_id SET r.evidence_chunk_ids = $pruned",
                rel_id=rel_id, pruned=pruned,
            )
        else:
            session.run("MATCH ()-[r]->() WHERE elementId(r) = $rel_id DELETE r", rel_id=rel_id)

    result = session.run(
        """
        MATCH (e:Entity)
        WHERE NOT (e)-[:MENTIONED_IN]->(:Chunk)
        WITH e, e.entity_id AS entity_id
        DETACH DELETE e
        RETURN entity_id
        """
    )
    return [record["entity_id"] for record in result]


def remove_source(source_id, chunks_file, extraction_file, neo4j_driver, qdrant_client):
    """Bir kaynagi ve ona munhasir chunk/entity/relation verisini siler. print etmez -
    hem CLI (cmd_remove) hem API (dashboard.py::delete_source) bunu kullanir.
    Donus: None (source_id bulunamadi) ya da {"removed_chunk_count", "removed_entity_count"}."""
    chunks = load_jsonl(chunks_file)
    removed_chunk_ids = chunk_ids_for_source(chunks, source_id)
    if not removed_chunk_ids:
        return None

    with neo4j_driver.session() as session:
        removed_entity_ids = remove_source_from_graph(session, removed_chunk_ids)

    chunk_point_ids = [vector_writer.chunk_point_id(cid) for cid in removed_chunk_ids]
    qdrant_client.delete(collection_name="chunks", points_selector=chunk_point_ids)

    if removed_entity_ids:
        entity_point_ids = [graph_writer.entity_point_id(eid) for eid in removed_entity_ids]
        qdrant_client.delete(collection_name="entities", points_selector=entity_point_ids)

    remaining_chunks = [c for c in chunks if c["source_id"] != source_id]
    write_jsonl(remaining_chunks, chunks_file)

    extraction = load_jsonl(extraction_file)
    removed_set = set(removed_chunk_ids)
    remaining_extraction = [r for r in extraction if r["chunk_id"] not in removed_set]
    write_jsonl(remaining_extraction, extraction_file)

    return {"removed_chunk_count": len(removed_chunk_ids), "removed_entity_count": len(removed_entity_ids)}


def cmd_remove(source_id, chunks_file, extraction_file, neo4j_driver, qdrant_client):
    print(f"'{source_id}' siliniyor...")
    result = remove_source(source_id, chunks_file, extraction_file, neo4j_driver, qdrant_client)
    if result is None:
        print(f"HATA: '{source_id}' sistemde bulunamadi.")
        return
    print(
        f"BASARILI! {result['removed_chunk_count']} chunk, {result['removed_entity_count']} munhasir entity silindi "
        f"(baska dokumanlarda da gecen entity/iliskiler korundu)."
    )


def cmd_update(file_path, chunks_file, extraction_file, neo4j_driver, qdrant_client, base_dir):
    import os

    source_id = os.path.basename(file_path)
    cmd_remove(source_id, chunks_file, extraction_file, neo4j_driver, qdrant_client)

    steps = [
        [sys.executable, "-m", "backend.pipeline.chunker", file_path],
        [sys.executable, "-m", "backend.pipeline.extractor"],
        [sys.executable, "-m", "backend.pipeline.graph_writer"],
        [sys.executable, "-m", "backend.pipeline.vector_writer"],
    ]
    for step_cmd in steps:
        step_module = step_cmd[2]
        print(f"\n=== {step_module} calistiriliyor ===", flush=True)
        result = subprocess.run(step_cmd, cwd=base_dir)
        if result.returncode != 0:
            print(f"\nHATA: {step_module} basarisiz oldu (exit code {result.returncode}). Update durduruldu.", flush=True)
            sys.exit(result.returncode)

    print(f"\nBASARILI! '{source_id}' guncellendi (silindi + yeniden ingest edildi).", flush=True)


if __name__ == "__main__":
    import argparse
    import os

    from backend.shared.paths import PROJECT_ROOT

    BASE_DIR = PROJECT_ROOT
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    EXTRACTION_FILE = os.path.join(BASE_DIR, "artifacts", "extraction.jsonl")

    parser = argparse.ArgumentParser(description="Kaynak/dokuman yonetimi: list/remove/update")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="Sistemdeki dokumanlari listeler (Neo4j/Qdrant gerekmez)")
    remove_parser = subparsers.add_parser("remove", help="Bir kaynagi ve ona munhasir verisini siler")
    remove_parser.add_argument("source_id", help="orn. mimari_v1.txt")
    update_parser = subparsers.add_parser("update", help="Bir kaynagi siler ve guncel halini yeniden ingest eder")
    update_parser.add_argument("file_path", help="orn. sources/mimari_v1.txt")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(CHUNKS_FILE, EXTRACTION_FILE)
    elif args.command in ("remove", "update"):
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
        from qdrant_client import QdrantClient

        load_dotenv(os.path.join(BASE_DIR, ".env"))
        neo4j_driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD")),
        )
        qdrant_client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

        if args.command == "remove":
            cmd_remove(args.source_id, CHUNKS_FILE, EXTRACTION_FILE, neo4j_driver, qdrant_client)
        else:
            cmd_update(args.file_path, CHUNKS_FILE, EXTRACTION_FILE, neo4j_driver, qdrant_client, BASE_DIR)

        neo4j_driver.close()
