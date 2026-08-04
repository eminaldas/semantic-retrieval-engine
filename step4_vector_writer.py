import json
import os
import sys
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "chunks"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # step1_chunker.py ile ayni model olmali

# chunk_id -> Qdrant point id turetmek icin sabit namespace (uuid5, deterministik).
# Qdrant point id'leri int/UUID olmak zorunda; string chunk_id dogrudan kullanilamiyor.
CHUNK_ID_NAMESPACE = uuid.UUID("f1e6a3b0-9c1a-4f7d-8e3a-2b6f7a9c0d1e")


def chunk_point_id(chunk_id):
    """Ayni chunk_id her zaman ayni point'e denk gelir - tekrar calistirma duplicate yaratmaz (upsert)."""
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, chunk_id))


def load_chunks(chunks_file):
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_collection(client, vector_size):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )


def build_point(chunk, vector):
    return models.PointStruct(
        id=chunk_point_id(chunk["chunk_id"]),
        vector=vector.tolist(),
        payload={
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk["source_id"],
            "location": chunk["location"],
            "quote": chunk["quote"],
            "text": chunk["text"],
        },
    )


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    if not os.path.exists(CHUNKS_FILE):
        print(f"HATA: {CHUNKS_FILE} bulunamadi. Once step1_chunker.py calistirilmali.")
        sys.exit(1)

    print("Model yukleniyor ve Qdrant'a baglaniliyor...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vector_size = model.get_embedding_dimension()

    try:
        client = QdrantClient(url=qdrant_url)
        ensure_collection(client, vector_size)
    except Exception as exc:
        print(f"HATA: Qdrant'a baglanilamadi ({qdrant_url}): {exc}")
        print("docker compose up -d ile Qdrant'in ayakta oldugundan emin ol.")
        sys.exit(1)

    chunks = list(load_chunks(CHUNKS_FILE))
    vectors = model.encode([c["text"] for c in chunks])
    points = [build_point(chunk, vector) for chunk, vector in zip(chunks, vectors)]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"\nBASARILI! {len(points)} chunk '{COLLECTION_NAME}' koleksiyonuna yazildi (vektor boyutu={vector_size}).")
