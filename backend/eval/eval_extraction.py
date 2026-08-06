"""Extraction precision/recall degerlendirme araci - bagimsiz elle etiketlenmis bir
gold-set ile step2_extractor.py'nin gercek Gemini cikisini olcer.
docs/superpowers/specs/2026-07-24-extraction-eval-design.md'deki tasarima gore.

Not: bu bir 'stepN_' pipeline modulu degil - istendiginde calistirilan bir
kalite-olcum araci (eval_retrieval.py/dashboard.py gibi ayri duruyor).
Retrieval fazinin aksine gercek Gemini API cagrisi yapar (GEMINI_API_KEY gerekir)."""

import json
from backend.pipeline import step3_graph_writer as gw


def load_golden_set(path):
    """eval/golden_extraction.jsonl'u okur, bos satirlari atlar."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_extraction_result(result):
    """extract_chunk'in ExtractionResult ciktisini entity_id kumesine ve
    (source_id, relation_type, target_id) tuple kumesine cevirir - grafa yazarken
    kullanilan AYNI normalize_entity_id ile (yeniden yazilmiyor)."""
    entity_ids = set()
    for entity in result.entities:
        entity_id, _ = gw.normalize_entity_id(entity.name, entity.type)
        entity_ids.add(entity_id)

    relation_tuples = set()
    for relation in result.relations:
        source_id, _ = gw.normalize_entity_id(relation.source.name, relation.source.type)
        target_id, _ = gw.normalize_entity_id(relation.target.name, relation.target.type)
        relation_tuples.add((source_id, relation.relation_type, target_id))

    return entity_ids, relation_tuples


def compute_extraction_metrics(gold_entities, gold_relations, predicted_entities, predicted_relations):
    """Tek bir chunk icin ham sayimlari dondurur (oran degil - toplu ozette
    micro-average hesaplanacagi icin burada sadece dogru/tahmin/gold sayilari lazim)."""
    gold_entity_set = set(gold_entities)
    gold_relation_set = {(r["source"], r["relation_type"], r["target"]) for r in gold_relations}

    return {
        "entity_correct": len(gold_entity_set & predicted_entities),
        "entity_predicted": len(predicted_entities),
        "entity_gold": len(gold_entity_set),
        "relation_correct": len(gold_relation_set & predicted_relations),
        "relation_predicted": len(predicted_relations),
        "relation_gold": len(gold_relation_set),
    }


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def aggregate_extraction_results(chunk_results):
    """Chunk-bazli ham sayimlardan micro-averaged precision/recall/F1 hesaplar
    (toplam dogru / toplam tahmin, toplam dogru / toplam gold - once toplanir sonra bolunur,
    kucuk/buyuk chunk'lar esit agirlikta sayilmasin diye per-chunk ortalama alinmiyor)."""
    total_entity_correct = sum(c["entity_correct"] for c in chunk_results)
    total_entity_predicted = sum(c["entity_predicted"] for c in chunk_results)
    total_entity_gold = sum(c["entity_gold"] for c in chunk_results)

    total_relation_correct = sum(c["relation_correct"] for c in chunk_results)
    total_relation_predicted = sum(c["relation_predicted"] for c in chunk_results)
    total_relation_gold = sum(c["relation_gold"] for c in chunk_results)

    entity_precision = _safe_divide(total_entity_correct, total_entity_predicted)
    entity_recall = _safe_divide(total_entity_correct, total_entity_gold)
    relation_precision = _safe_divide(total_relation_correct, total_relation_predicted)
    relation_recall = _safe_divide(total_relation_correct, total_relation_gold)

    return {
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": _f1(entity_precision, entity_recall),
        "relation_precision": relation_precision,
        "relation_recall": relation_recall,
        "relation_f1": _f1(relation_precision, relation_recall),
    }


import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai

from backend.pipeline import step2_extractor as extractor  # extract_chunk, load_prompt_template, DEFAULT_MODEL reuse
from backend.query import step5_query_engine as qe  # load_chunks_by_id reuse - chunk text lookup by chunk_id
from backend.shared.paths import PROJECT_ROOT


def evaluate_chunk(record, client, model, prompt_template, chunks_by_id):
    """Tek bir golden kayit icin gercek extract_chunk'i calistirir, sonra
    normalize_extraction_result + compute_extraction_metrics'e devreder."""
    chunk_text = chunks_by_id[record["chunk_id"]]["text"]
    result = extractor.extract_chunk(client, model, prompt_template, chunk_text)
    predicted_entities, predicted_relations = normalize_extraction_result(result)
    counts = compute_extraction_metrics(
        record["gold_entities"], record["gold_relations"], predicted_entities, predicted_relations
    )
    counts["chunk_id"] = record["chunk_id"]
    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extraction precision/recall degerlendirme araci")
    parser.add_argument("--golden-set", default=None, help="golden set jsonl yolu (varsayilan: eval/golden_extraction.jsonl)")
    args = parser.parse_args()

    BASE_DIR = PROJECT_ROOT
    GOLDEN_SET_FILE = args.golden_set or os.path.join(BASE_DIR, "eval", "golden_extraction.jsonl")
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "01_extraction_prompt.md")
    LOG_FILE = os.path.join(BASE_DIR, "artifacts", "eval_extraction_log.jsonl")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("HATA: GEMINI_API_KEY tanimli degil. .env dosyasina bir anahtar ekleyin (bkz. .env.example).")
        raise SystemExit(1)

    golden_set = load_golden_set(GOLDEN_SET_FILE)
    chunks_by_id = qe.load_chunks_by_id(CHUNKS_FILE)
    prompt_template = extractor.load_prompt_template(PROMPT_FILE)
    client = genai.Client(api_key=api_key)
    model = os.environ.get("GEMINI_EXTRACTOR_MODEL", extractor.DEFAULT_MODEL)

    chunk_results = []
    for record in golden_set:
        counts = evaluate_chunk(record, client, model, prompt_template, chunks_by_id)
        chunk_results.append(counts)
        print(
            f"  {counts['chunk_id']}: entity {counts['entity_correct']}/{counts['entity_predicted']} "
            f"(gold={counts['entity_gold']})  relation {counts['relation_correct']}/{counts['relation_predicted']} "
            f"(gold={counts['relation_gold']})"
        )

    aggregate = aggregate_extraction_results(chunk_results)
    print(f"\n=== Ozet ({len(chunk_results)} chunk) ===")
    print(f"  entity_precision:   {aggregate['entity_precision']:.2f}")
    print(f"  entity_recall:      {aggregate['entity_recall']:.2f}")
    print(f"  entity_f1:          {aggregate['entity_f1']:.2f}")
    print(f"  relation_precision: {aggregate['relation_precision']:.2f}")
    print(f"  relation_recall:    {aggregate['relation_recall']:.2f}")
    print(f"  relation_f1:        {aggregate['relation_f1']:.2f}")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate": aggregate,
        "per_chunk": chunk_results,
    }
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(f"\nBASARILI! Log: {LOG_FILE}")
