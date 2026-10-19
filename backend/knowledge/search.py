"""Claude'a (MCP) donen arama: hibrit aday toplama -> cross-encoder rerank ->
az ama guclu parca. 'Lost in the middle'a karsi baglama 30 aday degil, en iyi
birkac parca konuyor; esigin altindakiler 'low_confidence' ile isaretleniyor."""

from backend.knowledge import store

# Ayni cross-encoder, ama cekirdek korpusun -1.5'i (query_engine) Markdown agirlikli
# proje belgelerinde fazla katiydi: dogru belgeden gelen parcalar -2..-3 bandinda,
# korpus-disi bir soru (pizza tarifi) -3.6'da kaldi. Gecici deger; nehaber soru
# setiyle kalibre edilecek. Sadece isaretler, hicbir parcayi dusurmez.
RELEVANCE_THRESHOLD = -3.0


def rerank(cross_encoder, query, candidates, top_k):
    if not candidates:
        return []
    scores = cross_encoder.predict([(query, c["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [dict(c, rerank_score=float(s)) for c, s in ranked]


def search(qdrant_client, embedding_model, cross_encoder, query, project=None, doc_type=None, top_k=6, candidates=20,
           threshold=RELEVANCE_THRESHOLD):
    dense_query = embedding_model.encode(query)
    pool = store.hybrid_search(qdrant_client, dense_query, query, project=project, doc_type=doc_type, limit=candidates)
    ranked = rerank(cross_encoder, query, pool, top_k)
    for item in ranked:
        item["low_confidence"] = item["rerank_score"] < threshold
    return ranked


def format_results(results):
    """Claude'un okuyacagi metin: her parca kaynak + konum + bolum ile, skora gore sirali."""
    if not results:
        return "Bu soruyla eslesen bir belge parcasi bulunamadi."
    if all(r["low_confidence"] for r in results):
        header = "UYARI: Hicbir parca guclu bir eslesme degil; belgeler bu soruyu ele almiyor olabilir.\n\n"
    else:
        header = ""
    blocks = []
    for i, r in enumerate(results, start=1):
        section = f" · {r['section']}" if r.get("section") else ""
        flag = " · zayif eslesme" if r["low_confidence"] else ""
        blocks.append(
            f"[{i}] {r['path']} ({r['location']}){section} · skor {r['rerank_score']:.2f}{flag}\n{r['text'].strip()}"
        )
    return header + "\n\n---\n\n".join(blocks)
