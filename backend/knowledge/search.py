"""Claude'a (MCP) donen arama: hibrit aday toplama -> cross-encoder rerank ->
az ama guclu parca. 'Lost in the middle'a karsi baglama 30 aday degil, en iyi
birkac parca konuyor; esigin altindakiler 'low_confidence' ile isaretleniyor.

Proje belgelerinin parcalari kucuk (ortanca ~110 karakter, cogu tek madde). Esleme
kucuk parcayla yapilir, Claude'a ise parcanin komsulariyla birlikte ~700 karakterlik
bir pencere gider (small-to-big); ayni yere dusen sonuclar tek pencerede birlesir."""

import re

from backend.knowledge import store
from backend.knowledge.runtime import CROSS_ENCODER_MODEL_NAME, EMBEDDING_MODEL_NAME
from backend.shared import tracing

# bge-reranker-v2-m3 skoru 0-1 arasi. factprobe altin setinde (2026-10-22, 42 cevapli + 12
# alakasiz soru, Gemini baglam cumleleriyle) alakasizlarin en iyi parcasi <= 0.041, cevaplilarinki
# >= 0.069: 0.05 ikisini ayiriyor (yanlis uyari 0, kacan alakasiz 0). Aralik dar; model ya da
# baglam uretimi degisirse yeniden kalibre edilmeli. Sadece isaretler, hicbir parcayi dusurmez.
RELEVANCE_THRESHOLD = 0.05
WINDOW_CHARS = 700  # sonuc basina pencere; eslesen parca bundan uzunsa tek basina gider
WINDOW_RADIUS = 3  # her yone en fazla kac komsu parca
MAX_OUTPUT_CHARS = 4000  # tum cevabin butcesi; en az bir sonuc her zaman gider
SUPERSEDED_FACTOR = 0.5  # yerine yeni not yazilmis notun skoru bu katsayiyla carpilir
# Bir belgeden en fazla kac sonuc one alinir; None = sinir yok. Iki belgeyi birlestiren sorularda
# ayni belgenin parcalari ilk sonuclari doldurup ikinci belgeyi disarida birakiyordu.
MAX_PER_SOURCE = None

_LINES_RE = re.compile(r"(\d+)\s*-\s*(\d+)")


def rerank_text(candidate):
    """Cross-encoder'in gordugu metin: varsa LLM baglam cumlesi + parca. factprobe altin setinde
    (2026-10-22) baglamla rerank alakasiz sorulari tam ayirdi (abstain 0.92 -> 1.00), hit@5 0.93 -> 0.95."""
    if candidate.get("chunk_context"):
        return f"{candidate['chunk_context']}\n{candidate['text']}"
    return candidate["text"]


def rerank(cross_encoder, query, candidates, top_k):
    if not candidates:
        return []
    scores = cross_encoder.predict([(query, rerank_text(c)) for c in candidates])
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


def demote_superseded(ranked, factor=SUPERSEDED_FACTOR):
    """Yerine yeni not yazilmis notlarin skoru dusurulur ve yeniden siralanir; silinmez,
    cunku Claude'un eski kararin ne oldugunu gormesi de gerekebilir."""
    for item in ranked:
        if item.get("superseded_by"):
            item["rerank_score"] *= factor
    return sorted(ranked, key=lambda r: r["rerank_score"], reverse=True)


def diversify(ranked, max_per_source, limit):
    """Sirayi bozmadan her belgeden en fazla max_per_source sonuc one alir; yer kalirsa elenenler
    sona eklenir, yani sonuc sayisi hic azalmaz. Doner: (liste, one alinmayan sonuc sayisi)."""
    if not max_per_source:
        return ranked[:limit], 0
    kept, held, counts = [], [], {}
    for item in ranked:
        key = item.get("path")
        if counts.get(key, 0) < max_per_source:
            kept.append(item)
            counts[key] = counts.get(key, 0) + 1
        else:
            held.append(item)
    pushed = sum(1 for item in held if item in ranked[:limit])
    return (kept + held)[:limit], pushed


def interleave(ranked_lists, limit):
    """Parca sorgularin siralarini sirayla birlestirir (once her listenin 1.'si, sonra 2.'si...);
    ayni parca bir kez girer. Boylece her sorgunun en iyi sonucu ilk siralarda yer alir; skorlar
    farkli sorgulara gore oldugu icin tek siraya dizilmez."""
    merged, seen = [], set()
    for row in range(max((len(r) for r in ranked_lists), default=0)):
        for ranked in ranked_lists:
            if row < len(ranked) and ranked[row]["chunk_id"] not in seen:
                seen.add(ranked[row]["chunk_id"])
                merged.append(ranked[row])
    return merged[:limit]


def _retrieve_and_rerank(qdrant_client, embedding_model, cross_encoder, query, flt, candidates, limit,
                         max_per_source, retriever):
    with tracing.span("embed", tracing.EMBEDDING, **{"embedding.model_name": EMBEDDING_MODEL_NAME}):
        dense_query = embedding_model.encode(query)
    with tracing.span("hybrid_retrieve", tracing.RETRIEVER, **{"input.value": tracing.clip(query)}) as span:
        pool = retriever(qdrant_client, dense_query, query, flt=flt, limit=candidates)
        branch_meta = _trace_branches(span, qdrant_client, dense_query, query, flt, pool)
        span.update(tracing.document_attributes("retrieval.documents", pool, "fusion_score", branch_meta))
    with tracing.span("rerank", tracing.RERANKER, **{"reranker.model_name": CROSS_ENCODER_MODEL_NAME,
                                                     "reranker.query": tracing.clip(query),
                                                     "reranker.top_k": limit}) as span:
        ranked, pushed = diversify(demote_superseded(rerank(cross_encoder, query, pool, len(pool))),
                                   max_per_source, limit)
        span.set("rag.max_per_source", max_per_source).set("rag.diversity_pushed", pushed)
        if tracing.enabled():
            span.update(tracing.document_attributes("reranker.input_documents", pool, "fusion_score"))
            span.update(tracing.document_attributes("reranker.output_documents", ranked, "rerank_score"))
            span.update(rerank_shift(pool, ranked))
    return ranked


def search(qdrant_client, embedding_model, cross_encoder, query, project=None, doc_type=None, top_k=5, candidates=20,
           threshold=RELEVANCE_THRESHOLD, path_prefix=None, modified_after=None, modified_before=None,
           max_per_source=MAX_PER_SOURCE, subqueries=None, retriever=None):
    """subqueries: soru parcalari (decompose.decompose). Verilirse soru ve her parca ayri getirilip
    kendi sorgusuna gore rerank edilir, sonra sirayla birlestirilir.
    retriever: aday toplama (client, dense_query, query, flt, limit) -> liste; varsayilan
    store.hybrid_search. Deneme yollari icin (m3.make_retriever)."""
    filters = {k: v for k, v in dict(project=project, doc_type=doc_type, path_prefix=path_prefix,
                                     modified_after=modified_after, modified_before=modified_before).items() if v}
    queries = [query] + [q for q in (subqueries or []) if q.strip() and q.strip() != query.strip()]
    with tracing.span("search", tracing.CHAIN, **{"input.value": tracing.clip(query), "metadata": filters,
                                                  "rag.project": project, "rag.top_k": top_k,
                                                  "rag.threshold": threshold,
                                                  "rag.subqueries": queries[1:] or None}) as root:
        flt = store.project_filter(project=project, doc_type=doc_type, path_prefix=path_prefix,
                                   modified_after=modified_after, modified_before=modified_before)
        # Pencereler birlesince sonuc azalir; birkac yedek aday tutulur.
        ranked_lists = [_retrieve_and_rerank(qdrant_client, embedding_model, cross_encoder, q, flt, candidates,
                                             top_k + 3, max_per_source, retriever or store.hybrid_search)
                        for q in queries]
        ranked = ranked_lists[0] if len(ranked_lists) == 1 else interleave(ranked_lists, top_k + 3)
        for item in ranked:
            item["low_confidence"] = item["rerank_score"] < threshold
        wanted = {n for r in ranked for n in neighbor_ids(r["chunk_id"])} - {r["chunk_id"] for r in ranked}
        chunks_by_id = fetch_chunks(qdrant_client, sorted(wanted))
        chunks_by_id.update({r["chunk_id"]: r for r in ranked})
        windows = build_windows(ranked, chunks_by_id)
        results = windows[:top_k]
        root.update(outcome_attributes(ranked, windows, results, threshold))
        return results


def _trace_branches(span, client, dense_query, query, flt, pool):
    """Izleme acikken dense ve BM25 kollarinin ayri siralarini span'e yazar; hata aramayi bozmaz."""
    if not tracing.enabled():
        return None
    try:
        branches = store.branch_ranks(client, dense_query, query, flt=flt)
    except Exception as exc:
        span.set("rag.branch_error", f"{type(exc).__name__}: {exc}")
        return None
    attrs, meta = branch_routing(branches, pool)
    span.update(attrs)
    return meta


def branch_routing(branches, pool):
    """Aday havuzundaki her parcanin dense/BM25 sirasi ve hangi koldan geldigi.
    Doner: (span ozetleri, chunk_id -> {"dense_rank", "bm25_rank"})."""
    dense = {cid: i for i, cid in enumerate(branches["dense"], start=1)}
    sparse = {cid: i for i, cid in enumerate(branches["sparse"], start=1)}
    meta, counts = {}, {"both": 0, "dense_only": 0, "bm25_only": 0}
    for item in pool:
        cid = item["chunk_id"]
        d, s = dense.get(cid), sparse.get(cid)
        meta[cid] = {"dense_rank": d, "bm25_rank": s}
        if d and s:
            counts["both"] += 1
        elif d:
            counts["dense_only"] += 1
        elif s:
            counts["bm25_only"] += 1
    return {"rag.pool_size": len(pool), "rag.bm25_terms": bool(branches["sparse"]),
            **{f"rag.pool_{k}": v for k, v in counts.items()}}, meta


def rerank_shift(pool, ranked):
    """Reranker'in RRF sirasini ne kadar degistirdigi: ilk sonucun RRF'teki yeri ve ortalama kayma."""
    fusion_rank = {c["chunk_id"]: i for i, c in enumerate(pool, start=1)}
    shifts = [abs(fusion_rank.get(r["chunk_id"], len(pool)) - i) for i, r in enumerate(ranked, start=1)]
    return {
        "rag.top1_fusion_rank": fusion_rank.get(ranked[0]["chunk_id"]) if ranked else None,
        "rag.mean_rank_shift": round(sum(shifts) / len(shifts), 2) if shifts else None,
        "rag.superseded_demoted": sum(1 for r in ranked if r.get("superseded_by")),
    }


def outcome_attributes(ranked, windows, results, threshold):
    """Aramanin sonucu: skorlar, esige mesafe, 'bulunamadi' karari, donen kaynaklar."""
    scores = [r["rerank_score"] for r in ranked]
    return {
        "rag.result_count": len(results),
        "rag.top1_score": scores[0] if scores else None,
        "rag.top2_score": scores[1] if len(scores) > 1 else None,
        "rag.threshold_margin": round(scores[0] - threshold, 4) if scores else None,
        "rag.below_threshold": sum(1 for s in scores if s < threshold),
        "rag.abstained": all(r["low_confidence"] for r in results),
        "rag.windows_merged": len(ranked) - len(windows),  # komsu pencereye dusup birlesen sonuc
        "output.value": "\n".join(f"{r['path']} | {r.get('location', '')} | {r['rerank_score']:.3f}" for r in results),
    }


def result_flags(r):
    """Sonuc basligindaki uyarilar. 'stale' cagiran tarafindan (diske bakarak) doldurulur."""
    flags = []
    if r["low_confidence"]:
        flags.append("zayif eslesme")
    if r.get("superseded_by"):
        flags.append(f"eski not, yerini {r['superseded_by']} aldi")
    if r.get("stale"):
        flags.append(f"guncel degil: {r['stale']}")
    return flags


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
        date = f" · {r['modified_at'][:10]}" if r.get("modified_at") else ""
        flags = "".join(f" · {f}" for f in result_flags(r))
        block = (f"[{i}] {r['path']} ({r['location']}){section}{date} · skor {r['rerank_score']:.2f}{flags}\n"
                 f"{r.get('context', r['text']).strip()}")
        if blocks and used + len(block) > max_chars:
            blocks.append(f"[{len(results) - i + 1} sonuc daha, karakter butcesi doldu; daha dar bir soruyla arayin]")
            break
        blocks.append(block)
        used += len(block) + 7
    return header + "\n\n---\n\n".join(blocks)
