"""Claude'a (MCP) donen arama: hibrit aday toplama -> cross-encoder rerank ->
az ama guclu parca. 'Lost in the middle'a karsi baglama 30 aday degil, en iyi
birkac parca konuyor; esigin altindakiler 'low_confidence' ile isaretleniyor.

Proje belgelerinin parcalari kucuk (ortanca ~110 karakter, cogu tek madde). Esleme
kucuk parcayla yapilir, Claude'a ise parcanin komsulariyla birlikte ~700 karakterlik
bir pencere gider (small-to-big); ayni yere dusen sonuclar tek pencerede birlesir."""

import re

from backend.knowledge import store

# factprobe altin setiyle kalibre edildi (2026-10-22, 36 cevapli + 12 cevapsiz soru):
# cevapli sorularin en iyi parcasi en dusuk -0.97, cevapsizlarin en yuksegi -1.57.
# -1.25 ikisinin ortasi; eski -3.0 cevapsiz sorularin 7/12'sini guclu gosteriyordu.
# Model degisirse yeniden kalibre edilmeli. Sadece isaretler, hicbir parcayi dusurmez.
RELEVANCE_THRESHOLD = -1.25
WINDOW_CHARS = 700  # sonuc basina pencere; eslesen parca bundan uzunsa tek basina gider
WINDOW_RADIUS = 3  # her yone en fazla kac komsu parca
MAX_OUTPUT_CHARS = 4000  # tum cevabin butcesi; en az bir sonuc her zaman gider

_LINES_RE = re.compile(r"(\d+)\s*-\s*(\d+)")


def rerank(cross_encoder, query, candidates, top_k):
    if not candidates:
        return []
    scores = cross_encoder.predict([(query, c["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [dict(c, rerank_score=float(s)) for c, s in ranked]


def _split_chunk_id(chunk_id):
    source_id, _, index = chunk_id.rpartition("#")
    return source_id, int(index)


def neighbor_ids(chunk_id, radius=WINDOW_RADIUS):
    source_id, index = _split_chunk_id(chunk_id)
    return [f"{source_id}#{i:04d}" for i in range(max(1, index - radius), index + radius + 1) if i != index]


def _merge_location(locations):
    """'satır 3-5' + 'satır 6-9' -> 'satır 3-9'; bicim tanınmazsa ilk konum."""
    ranges = [_LINES_RE.search(loc or "") for loc in locations]
    if not all(ranges):
        return locations[0]
    start = min(int(m.group(1)) for m in ranges)
    end = max(int(m.group(2)) for m in ranges)
    return _LINES_RE.sub(f"{start}-{end}", locations[0], count=1)


def build_windows(results, chunks_by_id, window_chars=WINDOW_CHARS, radius=WINDOW_RADIUS):
    """Her sonuca komsu parcalari (once yakin olanlar, sirayla once/sonra) pencere dolana
    kadar ekler. Daha iyi sirali bir sonucun penceresine dusen sonuc atlanir. Saf fonksiyon:
    chunks_by_id = {chunk_id: {"text", "location"}} cagiran tarafindan doldurulur."""
    windows, taken = [], set()
    for r in results:
        if r["chunk_id"] in taken:
            continue
        source_id, index = _split_chunk_id(r["chunk_id"])
        members, size = {index: r}, len(r["text"])
        for step in range(1, radius + 1):
            for i in (index - step, index + step):
                neighbor = chunks_by_id.get(f"{source_id}#{i:04d}")
                if neighbor is None or f"{source_id}#{i:04d}" in taken or size + len(neighbor["text"]) > window_chars:
                    continue
                members[i] = neighbor
                size += len(neighbor["text"])
        ordered = [members[i] for i in sorted(members)]
        taken.update(f"{source_id}#{i:04d}" for i in members)
        windows.append(dict(r, context="\n".join(m["text"].strip() for m in ordered),
                            location=_merge_location([m["location"] for m in ordered])))
    return windows


def fetch_chunks(client, chunk_ids):
    if not chunk_ids:
        return {}
    points = client.retrieve(store.COLLECTION_NAME, ids=[store.point_id(c) for c in chunk_ids],
                             with_payload=["chunk_id", "text", "location"], with_vectors=False)
    return {p.payload["chunk_id"]: p.payload for p in points}


def search(qdrant_client, embedding_model, cross_encoder, query, project=None, doc_type=None, top_k=5, candidates=20,
           threshold=RELEVANCE_THRESHOLD):
    dense_query = embedding_model.encode(query)
    pool = store.hybrid_search(qdrant_client, dense_query, query, project=project, doc_type=doc_type, limit=candidates)
    # Pencereler birlesince sonuc azalir; birkac yedek aday tutulur.
    ranked = rerank(cross_encoder, query, pool, top_k + 3)
    for item in ranked:
        item["low_confidence"] = item["rerank_score"] < threshold
    wanted = {n for r in ranked for n in neighbor_ids(r["chunk_id"])} - {r["chunk_id"] for r in ranked}
    chunks_by_id = fetch_chunks(qdrant_client, sorted(wanted))
    chunks_by_id.update({r["chunk_id"]: r for r in ranked})
    return build_windows(ranked, chunks_by_id)[:top_k]


def format_results(results, max_chars=MAX_OUTPUT_CHARS):
    """Claude'un okuyacagi metin: her pencere kaynak + konum + bolum ile, skora gore sirali."""
    if not results:
        return "Bu soruyla eslesen bir belge parcasi bulunamadi."
    if all(r["low_confidence"] for r in results):
        header = ("UYARI: Hicbir parca guclu bir eslesme degil; belgeler bu soruyu buyuk ihtimalle ele almiyor. "
                  "Cevap uydurma: kullaniciya belgelerde bulunamadigini soyle.\n\n")
    else:
        header = ""
    blocks, used = [], len(header)
    for i, r in enumerate(results, start=1):
        section = f" · {r['section']}" if r.get("section") else ""
        flag = " · zayif eslesme" if r["low_confidence"] else ""
        block = (f"[{i}] {r['path']} ({r['location']}){section} · skor {r['rerank_score']:.2f}{flag}\n"
                 f"{r.get('context', r['text']).strip()}")
        if blocks and used + len(block) > max_chars:
            blocks.append(f"[{len(results) - i + 1} sonuc daha, karakter butcesi doldu; daha dar bir soruyla arayin]")
            break
        blocks.append(block)
        used += len(block) + 7
    return header + "\n\n---\n\n".join(blocks)
