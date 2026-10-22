"""Proje belgeleri icin Qdrant deposu: 'project_chunks' koleksiyonu, iki adli vektor -
'dense' (multilingual MiniLM) + 'bm25' (kendi sparse vektorumuz, IDF'i Qdrant hesaplar).

Parcalarin tek kaynagi Qdrant payload'i - cekirdek korpustaki gibi bir JSONL'a
yazilmiyor; boylece proje basina artimli ekleme/silme filtreyle yapilabiliyor."""

import uuid

from qdrant_client import models

from backend.knowledge import bm25, projects

COLLECTION_NAME = "project_chunks"
DENSE = "dense"
SPARSE = "bm25"
POINT_NAMESPACE = uuid.UUID("5d0c7f3e-2b8a-4c61-9e4f-8a1b2c3d4e5f")
PAYLOAD_FIELDS = ("chunk_id", "project", "source_id", "path", "location", "section", "text", "doc_type", "modified_at",
                  "indexed_at", "dirs", "superseded_by")
PAYLOAD_INDEXES = {
    "project": models.PayloadSchemaType.KEYWORD,
    "source_id": models.PayloadSchemaType.KEYWORD,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "path": models.PayloadSchemaType.KEYWORD,
    "dirs": models.PayloadSchemaType.KEYWORD,
    "modified_at": models.PayloadSchemaType.DATETIME,
}


def point_id(chunk_id):
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def ensure_collection(client, vector_size):
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        existing = info.config.params.vectors[DENSE].size
        if existing != vector_size:
            raise ValueError(
                f"'{COLLECTION_NAME}' {existing} boyutlu vektorlerle kurulmus, model {vector_size} uretiyor. "
                f"Embedding modeli degismis: koleksiyonu silip tum projeleri --full ile yeniden indeksleyin."
            )
        present = set(info.payload_schema or {})
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={DENSE: models.VectorParams(size=vector_size, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        present = set()
    for field, schema in PAYLOAD_INDEXES.items():  # eski koleksiyonlara sonradan eklenen indeksler de kurulur
        if field not in present:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)


def build_point(record, dense_vector):
    indices, values = bm25.document_sparse_vector(record["embed_text"])
    return models.PointStruct(
        id=point_id(record["chunk_id"]),
        vector={
            DENSE: list(map(float, dense_vector)),
            SPARSE: models.SparseVector(indices=indices, values=values),
        },
        payload={k: record.get(k) for k in PAYLOAD_FIELDS},
    )


def project_filter(project=None, source_id=None, doc_type=None, path_prefix=None, modified_after=None,
                   modified_before=None):
    """path_prefix: klasor ('graf/' ya da 'graf') ya da tek dosya ('graf/x.md').
    modified_after/before: 'YYYY-MM-DD' (iki uc dahil)."""
    conditions = []
    for key, value in (("project", project), ("source_id", source_id), ("doc_type", doc_type)):
        if value:
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    if path_prefix:
        prefix = path_prefix.strip().replace("\\", "/")
        prefix = prefix[2:] if prefix.startswith("./") else prefix.lstrip("/")
        if prefix.lower().endswith(projects.SUPPORTED_EXTENSIONS):
            conditions.append(models.FieldCondition(key="path", match=models.MatchValue(value=prefix)))
        else:
            prefix = prefix.rstrip("/") + "/"
            conditions.append(models.FieldCondition(key="dirs", match=models.MatchValue(value=prefix)))
    if modified_after or modified_before:
        conditions.append(models.FieldCondition(key="modified_at", range=models.DatetimeRange(
            gte=f"{modified_after}T00:00:00Z" if modified_after else None,
            lte=f"{modified_before}T23:59:59Z" if modified_before else None,
        )))
    return models.Filter(must=conditions) if conditions else None


def delete_source(client, source_id):
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(filter=project_filter(source_id=source_id)),
    )


def upsert_records(client, records, dense_vectors, batch_size=128):
    points = [build_point(r, v) for r, v in zip(records, dense_vectors)]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION_NAME, points=points[start:start + batch_size])
    return len(points)


def hybrid_search(client, dense_query, query_text, flt=None, limit=30, prefetch_limit=60):
    """Dense ve BM25 adaylarini ayri ayri toplayip Reciprocal Rank Fusion ile
    birlestirir. Dense anlami, BM25 dosya/fonksiyon adlari gibi birebir terimleri yakalar.
    flt: project_filter ciktisi (proje, tur, klasor, tarih)."""
    q_indices, q_values = bm25.query_sparse_vector(query_text)
    prefetch = [models.Prefetch(query=list(map(float, dense_query)), using=DENSE, filter=flt, limit=prefetch_limit)]
    if q_indices:
        prefetch.append(models.Prefetch(
            query=models.SparseVector(indices=q_indices, values=q_values), using=SPARSE, filter=flt, limit=prefetch_limit,
        ))
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [dict(hit.payload, fusion_score=hit.score) for hit in result.points]


def scroll_all(client, flt, fields):
    """Filtreye uyan tum noktalarin istenen payload alanlari (sayfali scroll)."""
    rows, offset = [], None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME, scroll_filter=flt, limit=256, offset=offset,
            with_payload=list(fields), with_vectors=False,
        )
        rows.extend(p.payload for p in points)
        if offset is None:
            return rows


def summarize_sources(rows):
    """scroll_all satirlarindan belge listesi: path, tur, parca sayisi, son degisiklik."""
    sources = {}
    for row in rows:
        entry = sources.setdefault(row["source_id"], {
            "source_id": row["source_id"], "path": row.get("path"), "doc_type": row.get("doc_type"),
            "chunk_count": 0, "modified_at": row.get("modified_at"),
        })
        entry["chunk_count"] += 1
    return sorted(sources.values(), key=lambda e: e["path"] or "")
