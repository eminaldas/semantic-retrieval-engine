"""Altin set bakimi: mevcut kayitlarin denetimi ve zor soru adaylari uretimi.

    python -m backend.knowledge.goldgen audit factprobe            # dar kayitlari bul
    python -m backend.knowledge.goldgen generate factprobe --docs 12 --pairs 8

audit: her kaydin expected_text'ini tum belgelerde arar. Ayni bilgi listede olmayan bir belgede de
geciyorsa, arama o belgeyi getirdiginde 'kacti' sayilir; bu kayitlar raporlanir (kayit degistirilmez).

generate: Gemini'ye bir belge (ya da ayni klasorden iki belge) verip zor soru adaylari ister. Her
aday, belgeden birebir alinti ile gelir; alinti belgede gecmiyorsa aday atilir (uydurma soru kalmaz).
Alintinin gectigi diger belgeler de expected_paths'e eklenir. Adaylar
knowledge_eval/<proje>_candidates.jsonl'a origin=synthetic, reviewed=null olarak yazilir; altin sete
ancak gozden gecirildikten sonra (reviewed tarihiyle) tasinir. Sentetik sorular insan sorularindan
ayri raporlanir (eval.aggregate hit@3_by_origin)."""

import argparse
import json
import os
import random
from datetime import date

from backend.knowledge import eval as golden
from backend.knowledge import store

GEN_MODEL = "gemini-flash-latest"
CALL_SITE = "knowledge_goldgen"
MAX_DOC_CHARS = 12000
MIN_QUOTE_CHARS = 8

TYPES = {
    "parafraz": "belgedeki kelimeleri kullanmadan, esanlamlilarla sorulan olgu sorusu",
    "sayi": "belgedeki bir sayi, esik, oran ya da tarih",
    "neden": "bir kararin gerekcesi",
    "zaman": "ayni konuda belgenin verdigi en guncel durum ya da hangi asamada ne yapildigi",
    "cok_adim": "iki belgedeki bilgiyi birlestirmeden cevaplanamayan soru (sadece iki belge verildiginde)",
    "yakin_cevapsiz": "belgenin konusuna yakin ama cevabi belgede OLMAYAN, gercekci bir soru",
}

PROMPT = """Bir RAG sisteminin altin test setine Turkce soru adaylari yaziyorsun. Sorular, projeyi
bilen bir gelistiricinin gercekten soracagi gibi olmali; belgeyi gormeyen biri de anlamali
("bu belgede" gibi ifadeler yok).

Belge(ler):
{docs}

En fazla {n} soru uret. Kullanilabilecek turler:
{types}

Kurallar:
- Her cevaplanabilir soru icin quotes: cevabi kanitlayan, belgeden BIREBIR kopyalanmis kisa alinti(lar)
  (8-120 karakter), hangi belgeden oldugu path ile. Alintiyi degistirme, kisaltma isareti koyma.
- cok_adim sorusunda her iki belgeden de en az bir alinti olmali.
- yakin_cevapsiz sorusunda quotes bos, answer "YOK" olmali.
- answer: tek cumlelik referans cevap.
- Kolay, belgedeki basligi tekrar eden sorular yazma."""

SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "type": {"type": "string", "enum": list(TYPES)},
            "answer": {"type": "string"},
            "quotes": {"type": "array", "items": {"type": "object", "properties": {
                "path": {"type": "string"}, "text": {"type": "string"}}, "required": ["path", "text"]}},
        },
        "required": ["question", "type", "answer", "quotes"],
    }}},
    "required": ["questions"],
}


def load_documents(client, project):
    """path -> tam metin (parca sirasiyla). Sadece belgeler; notlar degisken oldugu icin disarida."""
    rows = store.scroll_all(client, store.project_filter(project=project, doc_type="doc"), ("path", "chunk_id", "text"))
    docs = {}
    for row in sorted(rows, key=lambda r: r["chunk_id"]):
        docs.setdefault(row["path"], []).append(row["text"].strip())
    return {path: "\n".join(parts) for path, parts in docs.items()}


def paths_containing(docs, text):
    needle = golden._norm_text(text)
    return sorted(p for p, body in docs.items() if needle in golden._norm_text(body))


def audit(records, docs):
    """Beklenen ifadenin listede olmayan belgelerde de gectigi kayitlar."""
    findings = []
    for record in records:
        expected = record.get("expected_text")
        if not record["expected_paths"] or not expected:
            continue
        options = [expected] if isinstance(expected, str) else expected
        listed = {golden._norm_path(p) for p in record["expected_paths"]}
        extra = sorted({p for option in options for p in paths_containing(docs, option)
                        if golden._norm_path(p) not in listed})
        missing = [option for option in options if not paths_containing(docs, option)]
        if extra or missing:
            findings.append({"id": record["id"], "question": record["question"], "also_in": extra,
                             "text_not_found": missing})
    return findings


def to_record(candidate, docs, index):
    """Model adayini dogrulanmis altin set kaydina cevirir; dogrulanamazsa None."""
    if candidate["type"] == "yakin_cevapsiz":
        return {"id": f"g{index:03d}", "type": "cevapsiz", "question": candidate["question"],
                "expected_paths": [], "expected_text": None, "origin": "synthetic", "reviewed": None,
                "generated_as": "yakin_cevapsiz"}
    quotes = [q for q in candidate["quotes"] if len(q["text"].strip()) >= MIN_QUOTE_CHARS
              and q["path"] in docs and paths_containing({q["path"]: docs[q["path"]]}, q["text"])]
    if not quotes or len(quotes) != len(candidate["quotes"]):
        return None  # tek bir alinti bile belgede yoksa aday guvenilmez
    record = {"id": f"g{index:03d}", "type": candidate["type"], "question": candidate["question"],
              "reference_answer": candidate["answer"], "origin": "synthetic", "reviewed": None}
    if candidate["type"] == "cok_adim":
        hops = {}
        for q in quotes:
            hops.setdefault(q["path"], []).append(q["text"])
        if len(hops) < 2:
            return None
        record["hops"] = [{"paths": paths_containing(docs, texts[0]), "text": texts} for texts in hops.values()]
        record["expected_paths"] = sorted({p for h in record["hops"] for p in h["paths"]})
        record["expected_text"] = [t for texts in hops.values() for t in texts]
    else:
        record["expected_paths"] = sorted({p for q in quotes for p in paths_containing(docs, q["text"])})
        record["expected_text"] = [q["text"] for q in quotes]
    return record


def pick_batches(docs, n_docs, n_pairs, seed=0):
    rng = random.Random(seed)
    paths = sorted(p for p, body in docs.items() if len(body) > 400)
    singles = [[p] for p in rng.sample(paths, min(n_docs, len(paths)))]
    by_dir = {}
    for p in paths:
        by_dir.setdefault(p.rsplit("/", 1)[0] if "/" in p else "", []).append(p)
    pairs = [sorted(rng.sample(group, 2)) for group in by_dir.values() if len(group) >= 2]
    rng.shuffle(pairs)
    return singles + pairs[:n_pairs]


def generate(gemini, docs, batches, per_batch=4, model=GEN_MODEL):
    from google.genai import types

    from backend.shared import gemini_retry, gemini_usage

    config = types.GenerateContentConfig(temperature=0.7, response_mime_type="application/json",
                                         response_schema=SCHEMA)
    type_lines = "\n".join(f"- {k}: {v}" for k, v in TYPES.items())
    candidates = []
    for batch in batches:
        text = "\n\n".join(f"=== {p} ===\n{docs[p][:MAX_DOC_CHARS]}" for p in batch)
        prompt = PROMPT.format(docs=text, n=per_batch + (1 if len(batch) > 1 else 0), types=type_lines)

        def _call():
            response = gemini.models.generate_content(model=model, contents=prompt, config=config)
            gemini_usage.log_usage(CALL_SITE, model, response)
            return response

        response = gemini_retry.call_with_retry(_call, pacing_key=CALL_SITE)
        try:
            found = json.loads(response.text or "{}").get("questions", [])
        except ValueError:
            found = []
        for c in found:
            c["source_docs"] = batch
        candidates.extend(found)
    return candidates


def main():
    from dotenv import load_dotenv
    from google import genai

    from backend.knowledge.runtime import Runtime
    from backend.knowledge.service import KnowledgeService
    from backend.shared.paths import PROJECT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "generate"])
    parser.add_argument("project")
    parser.add_argument("--golden-set", default=None)
    parser.add_argument("--docs", type=int, default=12, help="tek belgeden soru uretilecek belge sayisi")
    parser.add_argument("--pairs", type=int, default=8, help="cok adimli soru icin belge cifti sayisi")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runtime = Runtime()
    project = KnowledgeService(runtime).resolve_project(args.project)
    docs = load_documents(runtime.client(), project)
    eval_dir = os.path.join(PROJECT_ROOT, "knowledge_eval")

    if args.command == "audit":
        records = golden.load_golden_set(args.golden_set or os.path.join(eval_dir, f"{project}.jsonl"))
        for f in audit(records, docs):
            print(json.dumps(f, ensure_ascii=False))
        return

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    candidates = generate(gemini, docs, pick_batches(docs, args.docs, args.pairs, seed=args.seed))
    records, dropped = [], 0
    for c in candidates:
        record = to_record(c, docs, len(records) + 1)
        if record is None:
            dropped += 1
            continue
        record["generated_on"] = date.today().isoformat()
        record["source_docs"] = c["source_docs"]
        records.append(record)
    out = os.path.join(eval_dir, f"{project}_candidates.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(records)} aday yazildi, {dropped} aday alintisi belgede bulunamadigi icin atildi: {out}")


if __name__ == "__main__":
    main()
