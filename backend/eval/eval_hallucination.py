"""Cevap halusinasyon orani degerlendirme araci - retrieval fazinin ayni 5 golden
sorusunu (eval/golden_retrieval.jsonl) gercek tam pipeline'dan (path+vektor+rerank+
Answerer) gecirip, ayri bir 'hakem' LLM cagrisiyla cevabin kanita dayanip
dayanmadigini olcer. docs/superpowers/specs/2026-07-25-hallucination-eval-design.md.

Not: bu bir 'stepN_' pipeline modulu degil - eval_retrieval.py/eval_extraction.py
ile ayni kategoride bir kalite-olcum araci. Uc eval aracinin en 'agiri' - hem
Neo4j+Qdrant hem iki ayri gercek LLM cagrisi (Answerer + Judge) gerektirir."""

import json
from typing import List

from google.genai import types
from pydantic import BaseModel

from backend.shared import gemini_retry
from backend.shared import gemini_usage


def load_golden_set(path):
    """eval/golden_retrieval.jsonl'u okur (yeniden kullanilir, yeni dosya yok),
    bos satirlari atlar. eval_retrieval.py/eval_extraction.py ile ayni desen -
    kasitli olarak onlardan import edilmiyor, kendi kucuk kopyasi burada duruyor."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


DEFAULT_JUDGE_MODEL = "gemini-pro-latest"  # Answerer'dan (gemini-flash-latest) kasitli olarak farkli/daha guclu


class HallucinationJudgment(BaseModel):
    grounded: bool
    unsupported_claims: List[str] = []
    reasoning: str


def call_judge(client, model, prompt_template, evidence_bundle, answer_text):
    """Cevabin kanita dayanip dayanmadigini degerlendiren hakem cagrisi.
    Retry/backoff artik gemini_retry.py'de ortaklastirildi (devam 38) - call_answerer
    da artik retry kazandi, bu ikisi arasindaki eski 'kasitli fark' artik gecerli degil."""
    prompt = (
        prompt_template
        .replace("{EVIDENCE_BUNDLE}", evidence_bundle)
        .replace("{ANSWER_TEXT}", answer_text)
    )

    def _call():
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HallucinationJudgment,
            ),
        )
        gemini_usage.log_usage("eval_hallucination_judge", model, response)
        return response.parsed or HallucinationJudgment.model_validate_json(response.text)

    return gemini_retry.call_with_retry(_call, pacing_key="eval_hallucination_judge")


def cross_check_structural_vs_semantic(citation_issues, judgment):
    """validate_citations (yapisal - atif indeksleri gecerli mi) ile
    HallucinationJudgment (anlamsal - icerik kanitla ortusuyor mu) sonuclarini
    karsilastirir. 'diverges=True' ozellikle ilginc: biri temiz digeri degilse,
    yapisal dogrulamanin tek basina yeterli olmadigini gosterir."""
    structurally_clean = len(citation_issues) == 0
    semantically_grounded = judgment.grounded
    return {
        "structurally_clean": structurally_clean,
        "semantically_grounded": semantically_grounded,
        "diverges": structurally_clean != semantically_grounded,
    }


def aggregate_hallucination_results(results):
    """Soru-bazli grounded sonuclarindan groundedness_rate hesaplar. Bos liste
    (orn. hicbir soru icin kanit bulunamadiysa) icin sifira bolme yerine 0.0 doner."""
    n = len(results)
    if n == 0:
        return {"groundedness_rate": 0.0}
    return {"groundedness_rate": sum(r["grounded"] for r in results) / n}


import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai

from backend.query import query_engine as qe
from backend.shared.paths import PROJECT_ROOT


def evaluate_question(record, session, qdrant_client, embedding_model, cross_encoder, chunks_by_id,
                       client, answerer_model, answerer_prompt, judge_model, judge_prompt,
                       max_hops=2, seed_topk=20, top_n=5):
    """Tek bir golden soru icin tam pipeline'i (path+vektor+rerank+Answerer) calistirir,
    sonra hakem cagrisiyla cevabi kanitla kiyaslar. Kanit havuzu bossa None doner
    (Answerer/Judge hic cagrilmaz, query_engine.py'nin kendi __main__'indeki
    'Kanit bulunamadi, Answerer atlaniyor' davranisiyla tutarli)."""
    candidate_paths = qe.find_candidate_paths(session, record["cause_entity_id"], record["target_entity_id"], max_hops)
    seed_chunk_ids = qe.vector_seed_search(qdrant_client, embedding_model, record["question"], seed_topk)
    evidence_pool = qe.build_evidence_pool(candidate_paths, seed_chunk_ids, chunks_by_id)
    ranked_evidence = qe.rerank_evidence(cross_encoder, record["question"], evidence_pool, top_n)

    if not ranked_evidence:
        return None

    best_path = candidate_paths[0] if candidate_paths else None
    context_pack = qe.build_context_pack(record["question"], best_path, ranked_evidence)
    answer = qe.call_answerer(client, answerer_model, answerer_prompt, context_pack)

    available_sids = [f"S{i}" for i in range(1, len(ranked_evidence) + 1)]
    citation_issues = qe.validate_citations(answer, available_sids)

    judgment = call_judge(client, judge_model, judge_prompt, context_pack["evidence_bundle"], answer.answer_text)
    cross_check = cross_check_structural_vs_semantic(citation_issues, judgment)

    return {
        "question": record["question"],
        "grounded": judgment.grounded,
        "unsupported_claims": judgment.unsupported_claims,
        "citation_issues": citation_issues,
        "diverges": cross_check["diverges"],
    }


if __name__ == "__main__":
    parser = __import__("argparse").ArgumentParser(description="Cevap halusinasyon orani degerlendirme araci")
    parser.add_argument("--golden-set", default=None, help="golden set jsonl yolu (varsayilan: eval/golden_retrieval.jsonl)")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--seed-topk", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    BASE_DIR = PROJECT_ROOT
    GOLDEN_SET_FILE = args.golden_set or os.path.join(BASE_DIR, "eval", "golden_retrieval.jsonl")
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    ANSWERER_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "02_answerer_prompt.md")
    JUDGE_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "08_hallucination_judge_prompt.md")
    LOG_FILE = os.path.join(BASE_DIR, "artifacts", "eval_hallucination_log.jsonl")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("HATA: GEMINI_API_KEY tanimli degil. .env dosyasina bir anahtar ekleyin (bkz. .env.example).")
        raise SystemExit(1)

    golden_set = load_golden_set(GOLDEN_SET_FILE)
    chunks_by_id = qe.load_chunks_by_id(CHUNKS_FILE)
    answerer_prompt = qe.load_prompt_template(ANSWERER_PROMPT_FILE)
    judge_prompt = qe.load_prompt_template(JUDGE_PROMPT_FILE)

    neo4j_driver = qe.GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD")),
    )
    qdrant_client = qe.QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

    print("Modeller yukleniyor...")
    embedding_model = qe.SentenceTransformer(qe.EMBEDDING_MODEL_NAME)
    cross_encoder = qe.CrossEncoder(qe.CROSS_ENCODER_MODEL_NAME)
    client = genai.Client(api_key=api_key)
    answerer_model = os.environ.get("GEMINI_ANSWERER_MODEL", qe.DEFAULT_ANSWERER_MODEL)
    judge_model = os.environ.get("GEMINI_HALLUCINATION_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    results = []
    with neo4j_driver.session() as session:
        for record in golden_set:
            result = evaluate_question(
                record, session, qdrant_client, embedding_model, cross_encoder, chunks_by_id,
                client, answerer_model, answerer_prompt, judge_model, judge_prompt,
                args.max_hops, args.seed_topk, args.top_n,
            )
            if result is None:
                print(f"  (atlandi - kanit bulunamadi) {record['question']}")
                continue
            results.append(result)
            grounded_mark = "OK" if result["grounded"] else "FAIL"
            diverges_mark = " [DIVERGES]" if result["diverges"] else ""
            print(f"  [grounded={grounded_mark}]{diverges_mark} unsupported={len(result['unsupported_claims'])} citation_issues={len(result['citation_issues'])}  {result['question']}")
    neo4j_driver.close()

    aggregate = aggregate_hallucination_results(results)
    print(f"\n=== Ozet ({len(results)} soru) ===")
    print(f"  groundedness_rate: {aggregate['groundedness_rate']:.2f}")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate": aggregate,
        "per_question": results,
    }
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(f"\nBASARILI! Log: {LOG_FILE}")
