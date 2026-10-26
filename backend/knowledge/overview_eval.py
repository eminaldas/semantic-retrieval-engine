"""Genel bakis sorulari olcumu (plan #9): "projede neler denendi", "hat/ klasorunun sonucu ne" gibi
tek bir parcada cevabi olmayan, bircok belgeyi ozetlemeyi isteyen sorular.

Altin set kayitlarinin aksine burada beklenen tek bir belge yok. Referans, projenin tum belgelerini
okuyan uzun baglam modelinden (longctx) bir kez alinir: 4-6 anahtar nokta ve dayandiklari belgeler.
Olcum, bu noktalarin ne kadarinin karsilandigidir (kapsama):

  context_coverage: search aracinin Claude'a dondurdugu metin (tek cagri) noktalarin kacini iceriyor.
                    Ozet katmani en cok bunu degistirir.
  answer_coverage:  ask ajaninin cevabi noktalarin kacini iceriyor (--ask ile; Gemini ajani ucretli).

    python -m backend.knowledge.overview_eval refs factprobe sorular.txt     # bir kez, referans uretir
    python -m backend.knowledge.overview_eval run factprobe --label baseline [--ask]

refs: sorular.txt'de satir basina bir soru. Kayitlar knowledge_eval/<proje>_overview.jsonl'a
origin=longctx, reviewed=null olarak yazilir; noktalar elle gozden gecirilebilir. Kapsama yargici
ayni Gemini yargic modelidir; bir nokta "ayni bilgi farkli kelimelerle" verilmisse karsilanmis sayilir."""

import argparse
import json
import os
import time

from backend.shared import tracing

JUDGE_MODEL = "gemini-flash-latest"
CALL_SITE = "knowledge_overview_eval"
MAX_TEXT_CHARS = 24000
TOP_K = 8

REF_PROMPT = """Soru: {question}

Bu soruyu projenin TUM belgelerine dayanarak cevapla. Cevabi 4-6 anahtar noktaya bol: her nokta tek
cumle, kendi basina anlasilir, belgelerdeki somut bilgiyi (ad, sayi, sonuc, karar) icersin. Her nokta
icin dayandigi belge yollarini ver. Belgelerde olmayan bir sey ekleme; genel cumleler yazma."""

REF_SCHEMA = {
    "type": "object",
    "properties": {"points": {"type": "array", "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}},
        "required": ["text", "paths"]}}},
    "required": ["points"],
}

COVER_PROMPT = """Bir proje belgeleri asistaninin bir soruya dondurdugu METNI degerlendiriyorsun.

SORU:
{question}

ANAHTAR NOKTALAR (iyi bir cevabin icermesi gerekenler):
{points}

METIN:
{text}

Her anahtar nokta icin karar ver: METIN bu noktanin bilgisini iceriyor mu? Ayni bilgi farkli
kelimelerle ya da daha ayrintili verilmisse evet. Sadece konuya deginip asil bilgiyi (sayi, sonuc,
karar) vermiyorsa hayir. covered listesi noktalarla ayni sirada ve ayni uzunlukta olsun.
contradictions: METIN'de bir noktayla celisen ifadeler (yoksa bos liste). JSON dondur."""

COVER_SCHEMA = {
    "type": "object",
    "properties": {"covered": {"type": "array", "items": {"type": "boolean"}},
                   "contradictions": {"type": "array", "items": {"type": "string"}}},
    "required": ["covered", "contradictions"],
}


def _generate(client, model, contents, schema, config_extra=None):
    from google.genai import types

    from backend.shared import gemini_retry, gemini_usage

    config = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                         response_schema=schema, **(config_extra or {}))

    def _call():
        response = client.models.generate_content(model=model, contents=contents, config=config)
        gemini_usage.log_usage(CALL_SITE, model, response)
        return response

    return gemini_retry.call_with_retry(_call, pacing_key=CALL_SITE)


def parse_points(raw, known_paths=None):
    """Referans ciktisi -> [{"text", "paths"}]; bos metinli noktalar atilir, bilinmeyen yollar silinir."""
    try:
        items = json.loads(raw).get("points", [])
    except (ValueError, AttributeError):
        return []
    known = {p.replace("\\", "/").lower(): p for p in (known_paths or [])}
    points = []
    for item in items if isinstance(items, list) else []:
        text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
        if not text:
            continue
        paths = [str(p) for p in item.get("paths", [])]
        if known:
            paths = [known[p.replace("\\", "/").lower()] for p in paths if p.replace("\\", "/").lower() in known]
        points.append({"text": text, "paths": paths})
    return points


def parse_coverage(raw, n_points):
    """Yargic ciktisi -> {"covered": [bool]*n, "contradictions": [...]} ya da hata."""
    try:
        data = json.loads(raw)
        covered = [bool(x) for x in data["covered"]]
        if len(covered) != n_points:
            raise ValueError(f"{len(covered)} karar, {n_points} nokta")
        return {"covered": covered, "contradictions": [str(x) for x in data.get("contradictions", [])][:3],
                "error": None}
    except (ValueError, KeyError, TypeError) as exc:
        return {"covered": None, "contradictions": [], "error": f"{type(exc).__name__}: {exc}"}


def build_cover_prompt(question, points, text, max_chars=MAX_TEXT_CHARS):
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... kesildi]"
    numbered = "\n".join(f"{i}. {p['text']}" for i, p in enumerate(points, start=1))
    return COVER_PROMPT.format(question=question, points=numbered, text=text or "(bos)")


def judge_coverage(client, question, points, text, model=JUDGE_MODEL):
    with tracing.span("coverage_judge", tracing.LLM, **{"llm.model_name": model,
                                                        "input.value": tracing.clip(question)}) as span:
        response = _generate(client, model, build_cover_prompt(question, points, text), COVER_SCHEMA)
        verdict = parse_coverage(response.text or "", len(points))
        span.set("rag.covered", json.dumps(verdict["covered"]))
    return verdict


def coverage(verdict):
    if not verdict or verdict["covered"] is None:
        return None
    return sum(verdict["covered"]) / len(verdict["covered"])


def source_recall(points, cited):
    """Referans noktalarinin dayandigi belgelerden kaci kaynak olarak gosterilmis/getirilmis."""
    expected = {p.replace("\\", "/").lower() for point in points for p in point["paths"]}
    if not expected:
        return None
    got = {c.replace("\\", "/").lower() for c in cited}
    return len(expected & got) / len(expected)


def aggregate(rows):
    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    return {"n": len(rows), "context_coverage": mean("context_coverage"),
            "context_source_recall": mean("context_source_recall"),
            "answer_coverage": mean("answer_coverage"), "answer_source_recall": mean("answer_source_recall"),
            "contradictions": sum(len(r.get("contradictions", [])) for r in rows),
            "judge_errors": sum(1 for r in rows if r.get("judge_error"))}


# --- komutlar ----------------------------------------------------------------

def make_references(project, questions, out_path, runtime=None):
    from backend.knowledge.agent import _long_context_reader
    from backend.knowledge.runtime import Runtime
    from backend.knowledge.service import KnowledgeService

    service = KnowledgeService(runtime or Runtime())
    reader = _long_context_reader(service, project)
    existing = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            existing = {json.loads(line)["question"] for line in f if line.strip()}
    try:
        cache = reader._cached_content()
        extra = {"cached_content": cache} if cache else {"system_instruction": reader.corpus}
        with open(out_path, "a", encoding="utf-8") as f:
            for i, question in enumerate(questions, start=1):
                if question in existing:
                    continue
                response = _generate(reader._gemini(), reader.model, REF_PROMPT.format(question=question),
                                     REF_SCHEMA, extra)
                points = parse_points(response.text or "", reader.docs)
                record = {"id": f"o{i:02d}", "question": question, "points": points, "origin": "longctx",
                          "reviewed": None}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{record['id']} {len(points)} nokta  {question}", flush=True)
    finally:
        reader.close()


def _run_records(records, rows, service, project, top_k, gemini, agent):
    from backend.knowledge.agent import cost_usd

    for record in records:
        points = record["points"]
        start = time.perf_counter()
        text = service.search(record["question"], project=project, top_k=top_k)
        row = {"id": record["id"], "search_seconds": round(time.perf_counter() - start, 3),
               "context_chars": len(text)}
        verdict = judge_coverage(gemini, record["question"], points, text)
        row["context_coverage"] = coverage(verdict)
        row["context_source_recall"] = source_recall(points, [p for point in points for p in point["paths"]
                                                              if p in text])
        row["contradictions"] = verdict["contradictions"]
        row["judge_error"] = verdict["error"]
        if agent:
            result = agent.ask(record["question"])
            row["_cost"] = cost_usd(result) or 0.0
            answer_verdict = judge_coverage(gemini, record["question"], points, result.answer)
            row["answer_coverage"] = coverage(answer_verdict)
            row["answer_source_recall"] = source_recall(points, result.citations)
            row["contradictions"] += answer_verdict["contradictions"]
            row["judge_error"] = row["judge_error"] or answer_verdict["error"]
            row["answer"] = result.answer[:1500]
            row["citations"] = result.citations
        rows.append(row)
        print(f"{record['id']} baglam={row['context_coverage']} cevap={row.get('answer_coverage')}  "
              f"{record['question']}", flush=True)


def run(project, golden_path, label="", use_ask=False, top_k=TOP_K, runtime=None, limit=0):
    from backend.knowledge import eval as golden
    from backend.knowledge.agent import ResearchAgent
    from backend.knowledge.runtime import Runtime
    from backend.knowledge.service import KnowledgeService
    from backend.shared.paths import PROJECT_ROOT

    service = KnowledgeService(runtime or Runtime())
    project = service.resolve_project(project)
    records = golden.load_golden_set(golden_path)
    records = records[:limit] if limit else records
    gemini = golden._gemini_client()
    agent = ResearchAgent(service, project=project) if use_ask else None
    tracing.init_tracing("rag-eval")
    service.search("isinma", project=project, top_k=1)
    rows, error = [], None
    try:
        _run_records(records, rows, service, project, top_k, gemini, agent)
    except Exception as exc:  # kredi bitmesi gibi: o ana kadarki satirlar kaybolmasin
        error = f"{type(exc).__name__}: {exc}"
        print(f"Olcum yarida kaldi: {error}", flush=True)
    agent_cost = sum(r.pop("_cost", 0.0) for r in rows)
    summary = aggregate(rows)
    summary["agent_cost_usd"] = round(agent_cost, 4)
    summary["error"] = error
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = os.path.join(PROJECT_ROOT, "artifacts", "knowledge_eval", f"{project}_overview.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "top_k": top_k,
                            "ask": use_ask, "summary": summary, "rows": rows}, ensure_ascii=False) + "\n")
    tracing.flush()
    return summary, rows


def main():
    from backend.shared.paths import PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Genel bakis sorulari olcumu")
    sub = parser.add_subparsers(dest="command", required=True)
    refs = sub.add_parser("refs", help="uzun baglam modelinden referans anahtar noktalar uret")
    refs.add_argument("project")
    refs.add_argument("questions", help="satir basina bir soru")
    runp = sub.add_parser("run", help="search (ve --ask ile ajan) kapsamasini olc")
    runp.add_argument("project")
    runp.add_argument("--label", default="")
    runp.add_argument("--ask", action="store_true")
    runp.add_argument("--top-k", type=int, default=TOP_K)
    runp.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    path = os.path.join(PROJECT_ROOT, "knowledge_eval", f"{args.project}_overview.jsonl")
    if args.command == "refs":
        with open(args.questions, encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        make_references(args.project, questions, path)
    else:
        run(args.project, path, label=args.label, use_ask=args.ask, top_k=args.top_k, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
