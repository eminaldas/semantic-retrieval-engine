"""BGE-M3'un ogrenilmis sparse ve ColBERT (cok vektorlu) ciktilari ile getirme denemesi.

Dense vektoru zaten BGE-M3 uretiyor; ayni modelin son katman ciktisindan iki kucuk kafa daha
hesaplanabiliyor (FlagEmbedding BGEM3FlagModel ile ayni formul, ek model yuklemeden):
  sparse:  token basina relu(sparse_linear(h)), ayni token icin en buyuk agirlik. Kendi BM25'imizin
           tersine agirliklari ogrenilmis; esanlamli/ekli kelimede BM25'ten iyi olmasi beklenir.
  colbert: normalize(colbert_linear(h[1:])), token basina 1024 boyut; skor MaxSim (late interaction).

Canli koleksiyona dokunmaz. 'build' mevcut project_chunks noktalarini (ayni id, payload, dense ve
BM25 vektoru) ayri bir koleksiyona kopyalar ve iki yeni vektoru ekler; Gemini ya da yeniden parcalama
yok. Eval bu koleksiyonda farkli getirme yollarini dener:

    python -m backend.knowledge.m3 build factprobe
    python -m backend.knowledge.eval factprobe --retriever m3-sparse --label m3-sparse

Yollar (RETRIEVERS): bm25 (canli ile ayni: dense+BM25 RRF, kopyanin dogrulamasi), m3-sparse
(dense+M3 sparse), hybrid3 (dense+BM25+M3 sparse), colbert (ucunun adaylari ColBERT MaxSim ile
siralanir). Hepsinde aday havuzu sonra ayni cross-encoder'a gider; fark havuza kimin girdigidir."""

import sys

from qdrant_client import models

from backend.knowledge import store

COLLECTION_NAME = "project_chunks_m3"
M3_SPARSE = "m3_sparse"
COLBERT = "colbert"
COLBERT_DIM = 1024
HEADS_REPO = "BAAI/bge-m3"
RETRIEVERS = ("bm25", "m3-sparse", "hybrid3", "colbert")
UPSERT_BATCH = 4


def embed_text_from_payload(payload):
    """sync'te vektorlenen metnin aynisi (projects.build_chunk_records): onek + baglam + parca."""
    from backend.knowledge.projects import context_prefix

    headings = payload["section"].split(" > ") if payload.get("section") else []
    prefix = context_prefix(payload["project"], payload["path"], headings)
    if payload.get("chunk_context"):
        prefix += f"\nBaglam: {payload['chunk_context']}"
    return prefix + "\n" + payload["text"]


def sparse_weights(token_ids, weights, skip_ids):
    """Token basina en buyuk agirlik; ozel tokenlar ve sifir agirliklar atilir. Doner: {id: agirlik}."""
    out = {}
    for tid, w in zip(token_ids, weights):
        if tid in skip_ids or w <= 0:
            continue
        if w > out.get(tid, 0.0):
            out[tid] = float(w)
    return out


class M3Heads:
    """SentenceTransformer ile yuklu BGE-M3'un son katmanindan sparse ve ColBERT ciktilari."""

    def __init__(self, embedder):
        import torch
        from huggingface_hub import hf_hub_download

        self.embedder = embedder
        self.transformer = embedder[0].auto_model
        self.tokenizer = embedder[0].tokenizer
        self.max_length = embedder.max_seq_length
        device, dtype = self.transformer.device, self.transformer.dtype
        hidden = self.transformer.config.hidden_size
        self.sparse_linear = torch.nn.Linear(hidden, 1)
        self.colbert_linear = torch.nn.Linear(hidden, COLBERT_DIM)
        for layer, name in ((self.sparse_linear, "sparse_linear.pt"), (self.colbert_linear, "colbert_linear.pt")):
            layer.load_state_dict(torch.load(hf_hub_download(HEADS_REPO, name), map_location="cpu"))
            layer.to(device=device, dtype=dtype).eval()
        t = self.tokenizer
        self.skip_ids = {i for i in (t.cls_token_id, t.eos_token_id, t.pad_token_id, t.unk_token_id) if i is not None}

    def encode(self, texts, batch_size=32):
        """Doner: [(sparse {id: agirlik}, colbert [[float]*1024, ...]), ...] metin sirasiyla."""
        import torch

        out = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.transformer.device)
            with torch.no_grad():
                hidden = self.transformer(**batch).last_hidden_state
                token_w = torch.relu(self.sparse_linear(hidden)).squeeze(-1).float().cpu().tolist()
                colbert = torch.nn.functional.normalize(self.colbert_linear(hidden[:, 1:]).float(), dim=-1).cpu()
            mask = batch["attention_mask"].cpu()
            ids = batch["input_ids"].cpu().tolist()
            for row in range(len(ids)):
                n = int(mask[row].sum())
                out.append((sparse_weights(ids[row][:n], token_w[row][:n], self.skip_ids),
                            colbert[row, :n - 1].tolist()))  # h[1:]: CLS haric, padding haric
        return out


def _sparse(weights):
    return models.SparseVector(indices=list(weights), values=list(weights.values()))


def ensure_collection(client, dense_size):
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            store.DENSE: models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
            # Sadece ikinci asamada siralama icin; HNSW kurulmaz (m=0), float16 ile yarim yer.
            COLBERT: models.VectorParams(size=COLBERT_DIM, distance=models.Distance.COSINE,
                                         multivector_config=models.MultiVectorConfig(
                                             comparator=models.MultiVectorComparator.MAX_SIM),
                                         hnsw_config=models.HnswConfigDiff(m=0), datatype=models.Datatype.FLOAT16),
        },
        sparse_vectors_config={store.SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF),
                               M3_SPARSE: models.SparseVectorParams()},
    )
    for field, schema in store.PAYLOAD_INDEXES.items():
        client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)


def build(client, heads, project, batch_size=64, log=print):
    """Canli koleksiyondaki proje noktalarini kopyalayip M3 sparse + ColBERT ekler. Doner: nokta sayisi."""
    flt = store.project_filter(project=project)
    total, offset = 0, None
    while True:
        points, offset = client.scroll(store.COLLECTION_NAME, scroll_filter=flt, limit=batch_size, offset=offset,
                                       with_payload=True, with_vectors=True)
        if not points:
            break
        if total == 0:
            ensure_collection(client, len(points[0].vector[store.DENSE]))
        encoded = heads.encode([embed_text_from_payload(p.payload) for p in points])
        batch = [models.PointStruct(id=p.id, payload=p.payload, vector={
                     store.DENSE: p.vector[store.DENSE], store.SPARSE: p.vector[store.SPARSE],
                     M3_SPARSE: _sparse(sparse), COLBERT: colbert})
                 for p, (sparse, colbert) in zip(points, encoded)]
        # ColBERT parca basina token x 1024 sayi; 64'luk tek istek Qdrant'in 32 MB JSON sinirini asiyordu.
        for start in range(0, len(batch), UPSERT_BATCH):
            client.upsert(COLLECTION_NAME, points=batch[start:start + UPSERT_BATCH])
        total += len(points)
        log(f"  {total} parca")
        if offset is None:
            break
    return total


def make_retriever(name, heads, prefetch_limit=60):
    """search.search(retriever=...) icin: (client, dense_query, query_text, flt, limit) -> aday listesi."""
    if name not in RETRIEVERS:
        raise ValueError(f"bilinmeyen getirme yolu: {name} ({', '.join(RETRIEVERS)})")

    def retrieve(client, dense_query, query_text, flt=None, limit=20):
        from backend.knowledge import bm25

        sparse, colbert = heads.encode([query_text])[0]
        prefetch = [models.Prefetch(query=list(map(float, dense_query)), using=store.DENSE, filter=flt,
                                    limit=prefetch_limit)]
        q_indices, q_values = bm25.query_sparse_vector(query_text)
        if name in ("bm25", "hybrid3", "colbert") and q_indices:
            prefetch.append(models.Prefetch(query=models.SparseVector(indices=q_indices, values=q_values),
                                            using=store.SPARSE, filter=flt, limit=prefetch_limit))
        if name in ("m3-sparse", "hybrid3", "colbert") and sparse:
            prefetch.append(models.Prefetch(query=_sparse(sparse), using=M3_SPARSE, filter=flt, limit=prefetch_limit))
        if name == "colbert":
            query, using = colbert, COLBERT
        else:
            query, using = models.FusionQuery(fusion=models.Fusion.RRF), None
        result = client.query_points(collection_name=COLLECTION_NAME, prefetch=prefetch, query=query, using=using,
                                     limit=limit, with_payload=True)
        return [dict(hit.payload, fusion_score=hit.score) for hit in result.points]

    return retrieve


def main():
    from backend.knowledge.runtime import Runtime
    from backend.knowledge.service import KnowledgeService

    if len(sys.argv) < 3 or sys.argv[1] != "build":
        print("Kullanim: python -m backend.knowledge.m3 build <proje>")
        return 1
    runtime = Runtime()
    project = KnowledgeService(runtime).resolve_project(sys.argv[2])
    client = runtime.client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)  # deneme koleksiyonu; her build bastan kurar
    total = build(client, M3Heads(runtime.embedder()), project)
    print(f"{COLLECTION_NAME}: {total} parca ({project})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
