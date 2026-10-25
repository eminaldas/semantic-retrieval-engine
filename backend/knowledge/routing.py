"""Proje ipucu (plan #5): proje verilmeden gelen soru baska bir projeye belirgin sekilde daha iyi
uyuyorsa sonuclarin sonuna not duser. Sonuclar yine varsayilan projeden gelir; diger projenin
icerigi gosterilmez (factprobe'da calisirken fake-news sonucu gelmemeli). Not, soruyu dogru
projede yeniden aratmak icin: "project='fake-news' ile ara".

Her proje icin ayni sorgu vektoruyle kucuk bir aday havuzu getirilir ve cross-encoder ile puanlanir;
projenin skoru en iyi rerank skorudur. Diger projenin skoru varsayilaninkini HINT_MARGIN kadar gecer
ve HINT_MIN_SCORE'un ustundeyse ipucu verilir. Esikler olcumle secilir:

    python -m backend.knowledge.routing factprobe fake-news

Her projenin altin setlerindeki sorular (cevapsizlar dahil) kendi projesinin sorusu sayilir;
varsayilan projenin sorusunda ipucu yanlis alarm, diger projenin sorusunda ipucu isabettir."""

import argparse
import glob
import json
import os
import time

from backend.knowledge import search, store
from backend.shared import tracing

PROBE_CANDIDATES = 10
HINT_MARGIN = 0.2
HINT_MIN_SCORE = 0.3
SKIP_SCORE = 1.0  # rerank skoru sigmoid; varsayilan skor + marj bunu gecerse ipucu imkansiz, yoklama atlanir
MARGINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def probe(client, embedder, cross_encoder, query, projects, candidates=PROBE_CANDIDATES, dense_query=None):
    """Doner: {proje: en iyi rerank skoru}; adayi olmayan proje 0.0."""
    if dense_query is None:
        dense_query = embedder.encode(query)
    scores = {}
    for name in projects:
        pool = store.hybrid_search(client, dense_query, query, flt=store.project_filter(project=name),
                                   limit=candidates)
        ranked = search.rerank(cross_encoder, query, pool, 1)
        scores[name] = ranked[0]["rerank_score"] if ranked else 0.0
    return scores


def pick_hint(scores, current, margin=HINT_MARGIN, min_score=HINT_MIN_SCORE):
    """Varsayilandan belirgin daha iyi uyan proje ya da None."""
    others = {p: s for p, s in scores.items() if p != current}
    if not others:
        return None
    best = max(others, key=others.get)
    if others[best] >= min_score and others[best] >= scores.get(current, 0.0) + margin:
        return best
    return None


def hint_text(project, scores, current):
    return (f"Not: Bu soru '{project}' projesinin belgelerine daha iyi uyuyor "
            f"(en iyi eslesme {scores[project]:.2f}, '{current}' icinde {scores.get(current, 0.0):.2f}). "
            f"O projede aramak icin project='{project}' verin.")


def project_hint(client, embedder, cross_encoder, query, current, projects, current_score=None):
    """Servis icin: ipucu metni ya da None. Hata aramayi bozmaz.
    current_score: aramanin kendi en iyi skoru; verilirse varsayilan proje yeniden yoklanmaz ve
    SKIP_SCORE'un ustundeyse hic yoklama yapilmaz (ipucu icin digerinin 1'i asmasi gerekirdi)."""
    others = [p for p in projects if p != current]
    if not current or not others:
        return None
    if current_score is not None and current_score + HINT_MARGIN >= SKIP_SCORE:
        return None
    with tracing.span("project_hint", tracing.CHAIN, **{"input.value": tracing.clip(query),
                                                        "rag.project": current}) as span:
        try:
            scores = probe(client, embedder, cross_encoder, query,
                           others if current_score is not None else [current] + others)
        except Exception as exc:
            span.set("rag.error", f"{type(exc).__name__}: {exc}")
            return None
        if current_score is not None:
            scores[current] = current_score
        hinted = pick_hint(scores, current)
        span.update({"rag.project_scores": json.dumps(scores), "rag.hint": hinted})
    return hint_text(hinted, scores, current) if hinted else None


# --- olcum -----------------------------------------------------------------

def golden_files(project, eval_dir):
    """<proje>.jsonl ve <proje>_*.jsonl (yedekler ve adaylar haric)."""
    paths = [os.path.join(eval_dir, f"{project}.jsonl")] + sorted(glob.glob(os.path.join(eval_dir, f"{project}_*.jsonl")))
    return [p for p in paths if os.path.exists(p) and "_candidates" not in p and "_v1_" not in p]


def sweep(rows, current, margins=MARGINS, min_score=HINT_MIN_SCORE):
    """rows: [{"project": gercek proje, "scores": {...}}]. Her marj icin yanlis alarm ve isabet orani."""
    own = [r for r in rows if r["project"] == current]
    foreign = [r for r in rows if r["project"] != current]
    out = []
    for margin in margins:
        false_alarm = sum(1 for r in own if pick_hint(r["scores"], current, margin, min_score))
        hit = sum(1 for r in foreign if pick_hint(r["scores"], current, margin, min_score) == r["project"])
        out.append({"margin": margin, "min_score": min_score,
                    "false_alarm": false_alarm / len(own) if own else None,
                    "hint_recall": hit / len(foreign) if foreign else None,
                    "own_n": len(own), "foreign_n": len(foreign)})
    return out


def run(current, others, eval_dir, runtime=None, sets=None):
    """sets: {proje: [yol, ...]}; verilmeyen proje icin golden_files."""
    from backend.knowledge.eval import load_golden_set
    from backend.knowledge.runtime import Runtime

    runtime = runtime or Runtime()
    client, embedder, reranker = runtime.client(), runtime.embedder(), runtime.reranker()
    projects = [current] + others
    probe(client, embedder, reranker, "isinma", projects)  # model yuklemesi sureye girmesin
    rows, latencies = [], []
    for project in projects:
        for path in (sets or {}).get(project) or golden_files(project, eval_dir):
            seen = set()
            for record in load_golden_set(path):
                if record["question"] in seen:
                    continue
                seen.add(record["question"])
                start = time.perf_counter()
                scores = probe(client, embedder, reranker, record["question"], projects)
                latencies.append(time.perf_counter() - start)
                rows.append({"id": record["id"], "set": os.path.basename(path), "project": project,
                             "answerable": bool(record["expected_paths"]), "question": record["question"],
                             "scores": scores})
    latencies.sort()
    return rows, sweep(rows, current), latencies[len(latencies) // 2] if latencies else None


def main():
    from backend.shared.paths import PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Proje ipucu olcumu")
    parser.add_argument("current", help="varsayilan proje (orn. factprobe)")
    parser.add_argument("others", nargs="+", help="ipucu verilebilecek projeler")
    parser.add_argument("--eval-dir", default=os.path.join(PROJECT_ROOT, "knowledge_eval"))
    parser.add_argument("--set", action="append", default=[], metavar="PROJE=YOL",
                        help="bir projenin soru dosyasi (orn. fake-news=knowledge_eval/fake-news_candidates.jsonl)")
    args = parser.parse_args()
    sets = {}
    for item in args.set:
        name, _, path = item.partition("=")
        sets.setdefault(name, []).append(path)
    rows, table, p50 = run(args.current, args.others, args.eval_dir, sets=sets)
    for row in table:
        print(f"marj {row['margin']:.2f}: yanlis alarm {row['false_alarm']} ({row['own_n']} soru), "
              f"isabet {row['hint_recall']} ({row['foreign_n']} soru)")
    print(f"probe p50 {p50:.2f} sn (tum projeler)")
    for r in rows:
        hinted = pick_hint(r["scores"], args.current)
        wrong = (r["project"] == args.current and hinted) or (r["project"] != args.current and hinted != r["project"])
        if wrong:
            print(f"  -- {r['project']}/{r['id']} ipucu={hinted} {r['scores']}  {r['question']}")
    log_path = os.path.join(PROJECT_ROOT, "artifacts", "knowledge_eval", "routing.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"current": args.current, "others": args.others, "sweep": table, "probe_p50": p50,
                            "rows": rows}, ensure_ascii=False) + "\n")
    print(f"Kayit: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
