"""Proje belgesi aramasinin (MCP search) altin setle olcumu.

    python -m backend.knowledge.eval factprobe [--golden-set yol] [--top-k 5] [--label not]

Altin set: knowledge_eval/<proje>.jsonl (gitignore'da; proje icerigi tasir). Her satir
{"id", "type", "question", "expected_paths": [...], "expected_text": str|[str]|null}.
expected_paths ve expected_text "herhangi biri" demektir: ayni bilgi birden cok belgede yaziliysa
hepsi listelenir. expected_paths bos ise soru cevapsizdir: dogru davranis tum sonuclarin zayif
isaretli olmasi. Istege bagli alanlar:
  "hops": [{"paths": [...], "text": ...}, ...]  iki ya da daha fazla belgeyi birlestiren soru; her
          adim ayri ayri ilk k sonucta bulunmali (multihop_recall)
  "origin": "human" | "synthetic"  sentetik sorular insan sorularindan ayri raporlanir
  "reviewed": "2026-10-25"  kaydin belgelere karsi son kontrol tarihi (belge degisirse yeniden bakilir)
Gemini'ye dokunmaz; sadece getirmeyi olcer. Her kosu artifacts/knowledge_eval/<proje>.jsonl'a
eklenir ve bir oncekiyle karsilastirilir."""

import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone


def load_golden_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm_path(path):
    return path.replace("\\", "/").strip("/").lower()


def _norm_text(text):
    # Turkce kucuk harf: casefold "İ"yi "i̇" yapar, "ŞİMDİLİK" ile "şimdilik" eslesmez.
    text = unicodedata.normalize("NFC", text).replace("İ", "i").replace("I", "ı").lower()
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)  # Turkce ondalik virgul: "0,91" == "0.91"
    return re.sub(r"\s+", " ", text)


def is_hit(record, result):
    """Sonuc beklenen belgelerden birinden mi geliyor (ve beklenen ifade varsa parcada geciyor mu)."""
    if _norm_path(result["path"]) not in {_norm_path(p) for p in record["expected_paths"]}:
        return False
    expected = record.get("expected_text")
    if not expected:
        return True
    # Ayni bilgi birden cok yerde yazili olabilir: listeden herhangi biri yeter.
    options = [expected] if isinstance(expected, str) else expected
    text = _norm_text(result.get("context", result["text"]))  # Claude'un gordugu pencere
    return any(_norm_text(option) in text for option in options)


def score_record(record, results):
    """Tek soru icin saf olcum: ilk dogru sonucun sirasi ve zayif-eslesme davranisi."""
    all_weak = all(r["low_confidence"] for r in results)
    if not record["expected_paths"]:
        return {"id": record["id"], "type": record["type"], "answerable": False, "rank": None,
                "abstained": all_weak}
    rank = next((i for i, r in enumerate(results, start=1) if is_hit(record, r)), None)
    score = {"id": record["id"], "type": record["type"], "answerable": True, "rank": rank,
             "abstained": all_weak, "origin": record.get("origin", "human")}
    if record.get("hops"):
        hops = [{"expected_paths": h["paths"], "expected_text": h.get("text")} for h in record["hops"]]
        score["hops_found"] = sum(1 for h in hops if any(is_hit(h, r) for r in results))
        score["hops_total"] = len(hops)
    return score


def aggregate(scores):
    answerable = [s for s in scores if s["answerable"]]
    unanswerable = [s for s in scores if not s["answerable"]]

    def rate(items, pred):
        return sum(1 for s in items if pred(s)) / len(items) if items else 0.0

    summary = {
        "n": len(scores),
        "hit@1": rate(answerable, lambda s: s["rank"] == 1),
        "hit@3": rate(answerable, lambda s: s["rank"] is not None and s["rank"] <= 3),
        "hit@5": rate(answerable, lambda s: s["rank"] is not None and s["rank"] <= 5),
        "mrr": sum(1 / s["rank"] for s in answerable if s["rank"]) / len(answerable) if answerable else 0.0,
        # Cevabi olan soruda "hicbiri guclu degil" demek: Claude'u gereksiz yere vazgecirir.
        "false_abstain": rate(answerable, lambda s: s["abstained"]),
        # Cevapsiz soruda zayif uyarisi vermek: uydurmaya karsi asil emniyet.
        "abstain_accuracy": rate(unanswerable, lambda s: s["abstained"]),
    }
    by_type = {}
    for s in answerable:
        by_type.setdefault(s["type"], []).append(s)
    summary["hit@3_by_type"] = {t: rate(items, lambda s: s["rank"] is not None and s["rank"] <= 3)
                                for t, items in sorted(by_type.items())}
    pooled = [s for s in answerable if "pool_hit" in s]
    if pooled:
        summary["pool_recall"] = rate(pooled, lambda s: s["pool_hit"])
    multihop = [s for s in answerable if s.get("hops_total")]
    if multihop:
        # Cok adimli soruda tek belge yetmez: tum adimlar ayni anda ilk k'da olmali.
        summary["multihop_recall"] = rate(multihop, lambda s: s["hops_found"] == s["hops_total"])
    by_origin = {}
    for s in answerable:
        by_origin.setdefault(s.get("origin", "human"), []).append(s)
    if len(by_origin) > 1:
        summary["hit@3_by_origin"] = {o: rate(items, lambda s: s["rank"] is not None and s["rank"] <= 3)
                                      for o, items in sorted(by_origin.items())}
    return summary


def _percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


def _gemini_client():
    from dotenv import load_dotenv
    from google import genai

    from backend.shared.paths import PROJECT_ROOT

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def pool_hit(record, pool_paths):
    """Beklenen belgelerden biri cross-encoder'a giden aday havuzunda mi (belge duzeyinde). Reranker
    havuzda olmayani one alamaz; getirme yollarinin farki burada gorunur."""
    expected = {_norm_path(p) for p in record["expected_paths"]}
    return any(_norm_path(p) in expected for p in pool_paths)


def run_eval(project_name, golden_path=None, top_k=5, label="", max_per_source=None, decompose_questions=False,
             runtime=None, verbose=True, retriever=None):
    """Altin seti bir kez kosar, kaydi loga ekler ve ozeti dondurur (main ve gate.py kullanir)."""
    from backend.knowledge import decompose, search, store
    from backend.knowledge.runtime import Runtime
    from backend.knowledge.service import KnowledgeService
    from backend.shared import tracing
    from backend.shared.paths import PROJECT_ROOT

    golden_path = golden_path or os.path.join(PROJECT_ROOT, "knowledge_eval", f"{project_name}.jsonl")
    log_path = os.path.join(PROJECT_ROOT, "artifacts", "knowledge_eval", f"{project_name}.jsonl")
    records = load_golden_set(golden_path)

    runtime = runtime or Runtime()
    project = KnowledgeService(runtime).resolve_project(project_name)
    client, embedder, reranker = runtime.client(), runtime.embedder(), runtime.reranker()
    search.search(client, embedder, reranker, "isinma", project=project, top_k=1)  # model yuklemesi olcume girmesin
    tracing.init_tracing("rag-eval")  # her soru Phoenix'te ayri iz; kacan sorular oradan incelenir
    gemini = _gemini_client() if decompose_questions else None
    if retriever:
        from backend.knowledge import m3

        base_retriever = m3.make_retriever(retriever, m3.M3Heads(embedder))
    else:
        base_retriever = store.hybrid_search
    pools = []

    def capture(*args, **kwargs):  # havuzu (rerank oncesi adaylar) olcum icin saklar
        pool = base_retriever(*args, **kwargs)
        pools.append([c["path"] for c in pool])
        return pool

    scores, latencies, chars = [], [], []
    for record in records:
        with tracing.span("eval_case", tracing.CHAIN, **{
                "input.value": record["question"], "metadata": {"id": record["id"], "type": record["type"],
                                                                "label": label, "project": project},
                "rag.expected_paths": record["expected_paths"] or None}) as span:
            start = time.perf_counter()
            subqueries = decompose.decompose(gemini, record["question"]) if gemini else None
            pools.clear()
            results = search.search(client, embedder, reranker, record["question"], project=project, top_k=top_k,
                                    subqueries=subqueries, retriever=capture,
                                    **({"max_per_source": max_per_source} if max_per_source else {}))
            latencies.append(time.perf_counter() - start)
            chars.append(len(search.format_results(results)))
            score = score_record(record, results)
            if score["answerable"]:
                score["pool_hit"] = pool_hit(record, [p for pool in pools for p in pool])
            span.update({"rag.answerable": score["answerable"], "rag.hit_rank": score["rank"],
                         "rag.abstained": score["abstained"],
                         "rag.passed": (score["rank"] is not None and score["rank"] <= 3) if score["answerable"]
                         else score["abstained"]})
        score["top"] = [f"{r['path']} {r['rerank_score']:.2f}" for r in results[:3]]
        if subqueries:
            score["subqueries"] = subqueries
        score["top1_score"] = results[0]["rerank_score"] if results else None
        scores.append(score)
        mark = ("OK " if score["rank"] and score["rank"] <= 3 else "-- ") if score["answerable"] else \
               ("OK " if score["abstained"] else "-- ")
        if verbose:
            print(f"{mark}{record['id']} rank={score['rank']} zayif={score['abstained']}  {record['question']}")

    summary = aggregate(scores)
    summary["latency_p50"] = _percentile(latencies, 50)
    summary["latency_p95"] = _percentile(latencies, 95)
    summary["chars_mean"] = sum(chars) / len(chars)

    previous = None
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        previous = json.loads(lines[-1])["summary"] if lines else None

    if verbose:
        print(f"\n=== {project_name}: {summary['n']} soru ===")
        for key in ("hit@1", "hit@3", "hit@5", "mrr", "false_abstain", "abstain_accuracy",
                    "latency_p50", "latency_p95", "chars_mean"):
            delta = f"  (onceki {previous[key]:.2f})" if previous and key in previous else ""
            print(f"  {key:17} {summary[key]:.2f}{delta}")
        print(f"  hit@3 tur bazinda: {summary['hit@3_by_type']}")
        for key in ("pool_recall", "multihop_recall", "hit@3_by_origin"):
            if key in summary:
                print(f"  {key}: {summary[key]}")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "label": label,
                            "golden_set": os.path.basename(golden_path),
                            "top_k": top_k, "max_per_source": max_per_source,
                            "decompose": decompose_questions, "retriever": retriever or "default",
                            "summary": summary, "per_question": scores},
                           ensure_ascii=False) + "\n")
    if verbose:
        print(f"\nKayit: {log_path}")
    tracing.flush()
    return summary, scores


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Proje aramasi altin set olcumu")
    parser.add_argument("project")
    parser.add_argument("--golden-set", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--label", default="", help="kosuya not (orn. 'baseline', 'bge-m3')")
    parser.add_argument("--max-per-source", type=int, default=None, help="belge basina en fazla sonuc (A/B icin)")
    parser.add_argument("--retriever", default=None,
                        help="aday toplama deneme yolu (m3.RETRIEVERS); once: python -m backend.knowledge.m3 build")
    parser.add_argument("--decompose", action="store_true",
                        help="soruyu Gemini ile parcalara bolup her parcayi ayri ara (sure parcalamayi da icerir)")
    args = parser.parse_args()
    run_eval(args.project, golden_path=args.golden_set, top_k=args.top_k, label=args.label,
             max_per_source=args.max_per_source, decompose_questions=args.decompose, retriever=args.retriever)


if __name__ == "__main__":
    main()
