import json
import os
import re
import sys
import uuid

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Ontology v0.1 (docs/01_architecture_v0.1.md #4) - relation_type dogrudan Cypher
# sorgusuna enterpole ediliyor (Neo4j iliski tipleri parametre olarak verilemez),
# bu yuzden interpole etmeden once mutlaka bu whitelist'e karsi dogrulaniyor.
RELATION_TYPES = {
    "TRIGGERS", "CAUSES", "LEADS_TO", "IMPACTS", "RISKS", "DEPENDS_ON",
    "USES", "DEGRADES", "IMPROVES", "MITIGATES", "PREVENTS",
}

ENTITY_COLLECTION_NAME = "entities"
ENTITY_SIMILARITY_THRESHOLD = 0.85  # docs/superpowers/specs/2026-07-22-entity-resolution-design.md #1

# entity_id -> Qdrant point id turetmek icin sabit namespace (uuid5, deterministik).
# step4_vector_writer.py'nin CHUNK_ID_NAMESPACE'iyle ayni desen, farkli bir sabit -
# chunk point id'leriyle hic cakismamasi icin kasten farkli UUID.
ENTITY_ID_NAMESPACE = uuid.UUID("a3f8c2e1-7b4d-4f9a-9c1e-6d2b8f0a5c3d")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # step1/step4 ile ayni model olmali

# Turkce ozel karakterler [^a-z0-9] regex'i tarafindan "gecersiz" sayilip sessizce
# siliniyordu (orn. "I/O Darbogazi" -> "i_o_darbo_az", "Odeme Veri Tabani" -> "deme_veri_taban").
# Regex'ten once ASCII karsiliklarina cevriliyor ki anlam kaybolmasin / farkli kelimeler
# ayni (yanlis) id'ye cakismasin.
TURKISH_TRANSLITERATION = str.maketrans({
    "ğ": "g", "Ğ": "g",
    "ı": "i", "İ": "i",
    "ş": "s", "Ş": "s",
    "ö": "o", "Ö": "o",
    "ü": "u", "Ü": "u",
    "ç": "c", "Ç": "c",
})


def normalize_entity_id(name, entity_type):
    """docs/01_architecture_v0.1.md #6.1: entity_id = <TYPE>::<normalized_name>"""
    transliterated = name.strip().translate(TURKISH_TRANSLITERATION).lower()
    normalized_name = re.sub(r"[^a-z0-9]+", "_", transliterated).strip("_")
    return f"{entity_type}::{normalized_name}", normalized_name


def entity_point_id(entity_id):
    """Ayni entity_id her zaman ayni point'e denk gelir - tekrar calistirma duplicate yaratmaz (upsert)."""
    return str(uuid.uuid5(ENTITY_ID_NAMESPACE, entity_id))


def ensure_entity_collection(client, vector_size):
    if not client.collection_exists(ENTITY_COLLECTION_NAME):
        client.create_collection(
            collection_name=ENTITY_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )


def upsert_entity_embedding(qdrant_client, entity_id, name, entity_type, embedding_model):
    vector = embedding_model.encode(name)
    vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    qdrant_client.upsert(
        collection_name=ENTITY_COLLECTION_NAME,
        points=[models.PointStruct(
            id=entity_point_id(entity_id),
            vector=vector,
            payload={"entity_id": entity_id, "type": entity_type, "name": name},
        )],
    )


def resolve_entity_id(name, entity_type, aliases, tx, qdrant_client, embedding_model, threshold=ENTITY_SIMILARITY_THRESHOLD):
    """docs/superpowers/specs/2026-07-22-entity-resolution-design.md #2: once tam
    eslesme (isim/alias kesisimi - LLM'in kendi cikardigi aliases'a guveniyoruz),
    sonra embedding yedegi (parafraz/ceviri varyasyonlari icin), sonra yeni id."""
    candidate_names = {normalize_entity_id(name, entity_type)[1]}
    for alias in aliases:
        candidate_names.add(normalize_entity_id(alias, entity_type)[1])

    existing = tx.run(
        "MATCH (e:Entity {type: $type}) RETURN e.entity_id AS entity_id, "
        "e.normalized_name AS normalized_name, e.aliases AS aliases",
        type=entity_type,
    )
    for record in existing:
        existing_names = {record["normalized_name"]}
        for alias in (record["aliases"] or []):
            existing_names.add(normalize_entity_id(alias, entity_type)[1])
        if candidate_names & existing_names:
            return record["entity_id"]

    vector = embedding_model.encode(name)
    vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    hits = qdrant_client.query_points(
        collection_name=ENTITY_COLLECTION_NAME,
        query=vector,
        limit=1,
        query_filter=models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value=entity_type))]),
    ).points
    if hits and hits[0].score >= threshold:
        return hits[0].payload["entity_id"]

    return normalize_entity_id(name, entity_type)[0]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_constraints(session):
    """docs/01_architecture_v0.1.md #7.4"""
    session.run(
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
    )
    session.run(
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
    )
    session.run(
        "CREATE INDEX entity_normalized_name IF NOT EXISTS "
        "FOR (e:Entity) ON (e.normalized_name)"
    )
    session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")


def chunk_exists(tx, chunk_id):
    """step6_self_correction.py'nin dogru bir 'eski/yeni' ayrimi yapabilmesi icin:
    bu chunk_id, upsert_chunk cagrilmadan ONCE grafta zaten var miydi?"""
    result = tx.run("MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN c", chunk_id=chunk_id)
    return result.single() is not None


def upsert_chunk(tx, chunk):
    tx.run(
        """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        ON CREATE SET c.source_id = $source_id, c.location = $location, c.quote = $quote
        """,
        chunk_id=chunk["chunk_id"],
        source_id=chunk["source_id"],
        location=chunk["location"],
        quote=chunk["quote"],
    )


def upsert_entity(tx, entity, entity_id, normalized_name):
    """docs/01_architecture_v0.1.md #6.2: yeni alias birlestir, confidence icin max al.
    entity_id/normalized_name artik disaridan (resolve_entity_id'den) geliyor - bu
    fonksiyon artik kendi id'sini hesaplamiyor, sadece verilen id'ye yaziyor."""
    tx.run(
        """
        MERGE (e:Entity {entity_id: $entity_id})
        ON CREATE SET
            e.name = $name, e.type = $type, e.normalized_name = $normalized_name,
            e.aliases = $aliases, e.confidence = $confidence
        ON MATCH SET
            e.confidence = CASE WHEN $confidence > e.confidence THEN $confidence ELSE e.confidence END,
            e.aliases = e.aliases + [a IN $aliases WHERE NOT a IN e.aliases]
        """,
        entity_id=entity_id,
        name=entity["name"],
        type=entity["type"],
        normalized_name=normalized_name,
        aliases=entity.get("aliases", []),
        confidence=entity["confidence"],
    )
    return entity_id


def upsert_mentioned_in(tx, entity_id, chunk_id):
    tx.run(
        """
        MATCH (e:Entity {entity_id: $entity_id}), (c:Chunk {chunk_id: $chunk_id})
        MERGE (e)-[:MENTIONED_IN]->(c)
        """,
        entity_id=entity_id,
        chunk_id=chunk_id,
    )


def upsert_relation(tx, relation, chunk_id, resolved_ids):
    """docs/01_architecture_v0.1.md #6.2: duplicate edge -> evidence_chunk_ids birikir, confidence max alinir.
    docs/superpowers/specs/2026-07-22-entity-resolution-design.md: source/target icin once
    resolved_ids'e (bu chunk'in entities[] listesi islenirken cozulmus id'ler) bakilir - boylece
    entity-resolution'in birlestirdigi bir entity'ye kuran relation'lar da DOGRU id'ye gider.
    resolved_ids'te yoksa (kontrat disi bir durum) naive normalize_entity_id'ye duser."""
    relation_type = relation["relation_type"]
    if relation_type not in RELATION_TYPES:
        raise ValueError(f"Ontoloji disi relation_type: {relation_type}")

    source_key = (relation["source"]["name"], relation["source"]["type"])
    target_key = (relation["target"]["name"], relation["target"]["type"])
    source_id = resolved_ids.get(source_key) or normalize_entity_id(*source_key)[0]
    target_id = resolved_ids.get(target_key) or normalize_entity_id(*target_key)[0]

    query = f"""
        MATCH (a:Entity {{entity_id: $source_id}}), (b:Entity {{entity_id: $target_id}})
        MERGE (a)-[r:{relation_type}]->(b)
        ON CREATE SET r.confidence = $confidence, r.evidence_chunk_ids = [$chunk_id], r.status = 'ACTIVE'
        ON MATCH SET
            r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
            r.evidence_chunk_ids = CASE WHEN $chunk_id IN r.evidence_chunk_ids
                THEN r.evidence_chunk_ids ELSE r.evidence_chunk_ids + $chunk_id END
    """
    tx.run(query, source_id=source_id, target_id=target_id, confidence=relation["confidence"], chunk_id=chunk_id)


def write_extraction_record(tx, chunk, record, qdrant_client, embedding_model):
    upsert_chunk(tx, chunk)

    resolved_ids = {}
    for entity in record["entities"]:
        entity_id = resolve_entity_id(
            entity["name"], entity["type"], entity.get("aliases", []), tx, qdrant_client, embedding_model
        )
        _, normalized_name = normalize_entity_id(entity["name"], entity["type"])
        upsert_entity(tx, entity, entity_id, normalized_name)
        upsert_entity_embedding(qdrant_client, entity_id, entity["name"], entity["type"], embedding_model)
        upsert_mentioned_in(tx, entity_id, chunk["chunk_id"])
        resolved_ids[(entity["name"], entity["type"])] = entity_id

    for relation in record["relations"]:
        upsert_relation(tx, relation, chunk["chunk_id"], resolved_ids)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    EXTRACTION_FILE = os.path.join(BASE_DIR, "artifacts", "extraction.jsonl")
    NEW_CHUNK_IDS_FILE = os.path.join(BASE_DIR, "artifacts", "new_chunk_ids.json")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    if not password:
        print("HATA: NEO4J_PASSWORD tanimli degil. .env dosyasina bir sifre yaz (bkz. .env.example)")
        print("ve docker compose up -d ile Neo4j'yi ayni sifreyle baslat.")
        sys.exit(1)

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    for required_file in (CHUNKS_FILE, EXTRACTION_FILE):
        if not os.path.exists(required_file):
            print(f"HATA: {required_file} bulunamadi. Once step1_chunker.py ve step2_extractor.py calistirilmali.")
            sys.exit(1)

    print("Neo4j'ye ve Qdrant'a baglaniliyor, embedding modeli yukleniyor...")
    chunks_by_id = {c["chunk_id"]: c for c in load_jsonl(CHUNKS_FILE)}
    extraction_records = list(load_jsonl(EXTRACTION_FILE))

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    qdrant_client = QdrantClient(url=qdrant_url)
    ensure_entity_collection(qdrant_client, embedding_model.get_embedding_dimension())

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            session.execute_write(ensure_constraints)

            entity_count = 0
            relation_count = 0
            new_chunk_ids = []
            for i, record in enumerate(extraction_records, start=1):
                chunk_id = record["chunk_id"]
                chunk = chunks_by_id.get(chunk_id)
                if chunk is None:
                    print(f"  [{i}/{len(extraction_records)}] {chunk_id}: UYARI - chunks.jsonl'da yok, atlaniyor")
                    continue

                if not session.execute_read(chunk_exists, chunk_id):
                    new_chunk_ids.append(chunk_id)

                session.execute_write(write_extraction_record, chunk, record, qdrant_client, embedding_model)
                entity_count += len(record["entities"])
                relation_count += len(record["relations"])
                print(
                    f"  [{i}/{len(extraction_records)}] {chunk_id}: "
                    f"{len(record['entities'])} entity, {len(record['relations'])} relation yazildi"
                )

            with open(NEW_CHUNK_IDS_FILE, "w", encoding="utf-8") as f:
                json.dump(new_chunk_ids, f, ensure_ascii=False)

            print(f"\nBASARILI! upsert_report: {len(chunks_by_id)} chunk, {entity_count} entity-yazimi, {relation_count} relation-yazimi (duplicate'lar merge edildi).")
            print(f"Yeni chunk'lar ({len(new_chunk_ids)}): {NEW_CHUNK_IDS_FILE}")
    finally:
        driver.close()
