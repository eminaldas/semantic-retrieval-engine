import argparse
import json
import os
import re
import sys
from typing import List, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from neo4j import GraphDatabase
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

import step5b_entity_linker
import step5c_risk_ranking as risk_ranking
import step5d_conversation_memory as conv
import gemini_retry
import gemini_usage

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # step1/step4 ile ayni model
CROSS_ENCODER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # cok dilli (Turkce dahil)
VECTOR_COLLECTION = "chunks"
DEFAULT_ANSWERER_MODEL = "gemini-flash-latest"

# docs/01_architecture_v0.1.md #7.5: 1-hop genislemede izinli/onerilen relation seti.
TRAVERSAL_RELATION_TYPES = [
    "TRIGGERS", "CAUSES", "LEADS_TO", "RISKS", "IMPACTS", "DEGRADES", "IMPROVES",
]

# docs/01_architecture_v0.1.md #8.1: "v0.1 pratik" agirliklar (yuksek/orta/orta-dusuk).
# DEGRADES/IMPROVES dokumanda acikca agirliklandirilmamis; IMPACTS/RISKS ile ayni
# "orta" katmana alindi (bunlar da dogrudan bir etkiyi tanimliyor).
RELATION_WEIGHTS = {
    "CAUSES": 3, "LEADS_TO": 3, "TRIGGERS": 3,
    "RISKS": 2, "IMPACTS": 2, "DEGRADES": 2, "IMPROVES": 2,
    "DEPENDS_ON": 1, "USES": 1, "MITIGATES": 1, "PREVENTS": 1,
}


def resolve_mode(mode_arg, cause, target):
    """--mode acikca verilmisse oldugu gibi doner. Verilmemisse (None): --cause/--target'tan
    biri verilmisse 'pair' (mevcut MODE_B kisayolu bozulmadan calisir), hicbiri verilmemisse
    'general' (v0.3 madde 4 - yeni varsayilan, once nedensellik degil genel cevap dener)."""
    if mode_arg is not None:
        return mode_arg
    return "pair" if (cause or target) else "general"


class ModeDetection(BaseModel):
    """detect_mode_from_question'in LLM'den beklediği yapilandirilmis cikti."""
    mode: Literal["general", "pair", "open"]


def detect_mode_from_question(question, gemini_client, model_name, prompt_template):
    """Soru metninden general/pair/open secer (docs/prompts/11_mode_detection_prompt.md).
    gemini_client None ise (API anahtari yok) ya da cagri kalici olarak basarisiz olursa
    None doner - cagiran taraf bunu resolve_mode(None, None, None) -> 'general' olarak yorumlar."""
    if gemini_client is None:
        return None
    prompt = prompt_template.replace("{QUESTION}", question)

    def _call():
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ModeDetection,
            ),
        )
        gemini_usage.log_usage("mode_detection", model_name, response)
        parsed = response.parsed or ModeDetection.model_validate_json(response.text)
        return parsed.mode

    try:
        return gemini_retry.call_with_retry(_call, pacing_key="mode_detection")
    except RuntimeError:
        return None


def load_chunks_by_id(chunks_file):
    chunks = {}
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                chunks[chunk["chunk_id"]] = chunk
    return chunks


def score_path(relationships):
    """docs/01_architecture_v0.1.md #8.1: relation agirligi + kisa path bonusu +
    yuksek confidence bonusu + farkli kaynakli evidence bonusu.

    Relation agirligi ORTALAMA alinir (toplam degil): path'in hop basina nedensel
    "kalitesini" temsil eder. Toplarsak, ayni gucte relation'lari zincirleyen uzun
    bir path sirf hop biriktirdigi icin kisa bir path'i gecebilir - bu da ayri bir
    faktor olan "kisa path bonusu"nu anlamsizlastirir.
    """
    avg_relation_weight = sum(RELATION_WEIGHTS.get(r["type"], 1) for r in relationships) / len(relationships)
    length_bonus = max(0, 4 - len(relationships))
    avg_confidence = sum(r["confidence"] for r in relationships) / len(relationships)
    confidence_bonus = avg_confidence * 2

    distinct_evidence = {cid for r in relationships for cid in r["evidence_chunk_ids"]}
    diversity_bonus = (len(distinct_evidence) - 1) * 0.5 if len(distinct_evidence) > 1 else 0

    return avg_relation_weight + length_bonus + confidence_bonus + diversity_bonus


def find_candidate_paths(session, cause_id, target_id, max_hops):
    rel_pattern = "|".join(TRAVERSAL_RELATION_TYPES)
    query = (
        f"MATCH path = (a:Entity {{entity_id: $cause_id}})"
        f"-[:{rel_pattern}*1..{max_hops}]->"
        f"(c:Entity {{entity_id: $target_id}}) "
        f"WHERE ALL(r IN relationships(path) WHERE coalesce(r.status, 'ACTIVE') <> 'SUPERSEDED') "
        f"RETURN path"
    )
    result = session.run(query, cause_id=cause_id, target_id=target_id)

    paths = []
    for record in result:
        path = record["path"]
        relationships = [
            {"type": r.type, "confidence": r["confidence"], "evidence_chunk_ids": r["evidence_chunk_ids"]}
            for r in path.relationships
        ]
        names = [n["name"] for n in path.nodes]
        paths.append({"names": names, "relationships": relationships, "score": score_path(relationships)})

    paths.sort(key=lambda p: p["score"], reverse=True)
    return paths


def format_path(path):
    parts = [f"[{path['names'][0]}]"]
    for name, rel in zip(path["names"][1:], path["relationships"]):
        parts.append(f"--({rel['type']})-->")
        parts.append(f"[{name}]")
    return " ".join(parts)


def vector_seed_search(qdrant_client, embedding_model, question, top_k):
    """docs/01_architecture_v0.1.md #2: topK=20 vektor seed search."""
    vector = embedding_model.encode(question).tolist()
    hits = qdrant_client.query_points(collection_name=VECTOR_COLLECTION, query=vector, limit=top_k).points
    return [hit.payload["chunk_id"] for hit in hits]


def build_evidence_pool(candidate_paths, seed_chunk_ids, chunks_by_id, top_paths=5):
    """Path'lerin evidence_chunk_ids'i + vector seed chunk'lari birlesip tekillestirilir."""
    chunk_ids = set(seed_chunk_ids)
    for path in candidate_paths[:top_paths]:
        for rel in path["relationships"]:
            chunk_ids.update(rel["evidence_chunk_ids"])
    return [chunks_by_id[cid] for cid in chunk_ids if cid in chunks_by_id]


def rerank_evidence(cross_encoder, question, evidence_pool, top_n=5):
    """docs/01_architecture_v0.1.md #4: EvidenceSelector - evidence_pool -> S1..S5."""
    if not evidence_pool:
        return []
    pairs = [(question, chunk["text"]) for chunk in evidence_pool]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(evidence_pool, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_n]


class AnswererOutput(BaseModel):
    """docs/01_architecture_v0.1.md #3 (CIKTI-1)"""
    answer_text: str
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    used_sources: List[str]
    assumptions: List[str] = []


def load_prompt_template(prompt_path):
    """docs/prompts/*.md icindeki fenced code block'u okur (step2_extractor.py ile ayni sozlesme)."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.index("```\n") + len("```\n")
    end = content.rindex("```")
    return content[start:end].strip()


def build_context_pack(question, best_path, ranked_evidence, path_summary_override=None):
    """docs/01_architecture_v0.1.md #9: question + reasoning_path_summary + evidence_bundle.

    path_summary_override: verilirse best_path'ten turetme atlanir, bu string
    dogrudan kullanilir - MODE_A'nin (step5c_risk_ranking.py) risk siralamasi
    metnini enjekte etmesi icin (MODE_B'nin best_path -> format_path akisi degismedi)."""
    if path_summary_override is not None:
        path_summary = path_summary_override
    else:
        path_summary = format_path(best_path) if best_path else "(Graf uzerinde dogrudan bir path bulunamadi.)"
    evidence_lines = [
        f"[S{i}] (chunk_id={chunk['chunk_id']}, source={chunk['source_id']}, location={chunk['location']}): {chunk['text']}"
        for i, (chunk, _score) in enumerate(ranked_evidence, start=1)
    ]
    return {
        "question": question,
        "path_summary": path_summary,
        "evidence_bundle": "\n".join(evidence_lines) if evidence_lines else "(Kanit bulunamadi.)",
    }


def call_answerer(client, model_name, prompt_template, context_pack):
    prompt = (
        prompt_template
        .replace("{QUESTION}", context_pack["question"])
        .replace("{PATH_SUMMARY}", context_pack["path_summary"])
        .replace("{EVIDENCE_BUNDLE}", context_pack["evidence_bundle"])
    )

    def _call():
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnswererOutput,
            ),
        )
        gemini_usage.log_usage("step5_query_engine_answerer", model_name, response)
        return response.parsed or AnswererOutput.model_validate_json(response.text)

    return gemini_retry.call_with_retry(_call, pacing_key="step5_query_engine_answerer")


def validate_citations(answer, available_sids):
    """docs/01_architecture_v0.1.md #3 kalite kurali - LLM'in talimata uydugunu VARSAYMAK
    yerine kodda dogrular. Ihlal varsa kullaniciya gosterilecek uyari metinleri dondurur."""
    cited_sids = {f"S{n}" for n in re.findall(r"\[S(\d+)\]", answer.answer_text)}
    used_sids = set(answer.used_sources)
    available = set(available_sids)

    issues = []
    if not cited_sids.issubset(used_sids):
        issues.append(f"answer_text'te gecen ama used_sources'ta olmayan referanslar: {sorted(cited_sids - used_sids)}")
    if not used_sids.issubset(available):
        issues.append(f"used_sources, saglanan kanitlarin ({sorted(available)}) disina cikiyor: {sorted(used_sids - available)}")
    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QueryEngine: MODE_B (A->C) / MODE_A (acik uclu risk siralamasi) / MODE_GENERAL (ontolojiden bagimsiz genel soru-cevap)")
    parser.add_argument("--mode", choices=["pair", "open", "general"], default=None, help="verilmezse: --cause/--target verilmisse pair (MODE_B), verilmemisse general (MODE_GENERAL, v0.3 madde 4 varsayilani). pair=MODE_B (A->C), open=MODE_A (acik uclu risk siralamasi), general=ontolojiden bagimsiz genel soru-cevap (Neo4j kullanilmaz) - open/general icin --cause/--target verilmez")
    parser.add_argument("--cause", required=False, default=None, help="cause_entity_id, orn. SERVICE::auth_service (verilmezse --question'dan otomatik cikarilir; --mode open ile birlikte verilemez)")
    parser.add_argument("--target", required=False, default=None, help="target_entity_id, orn. DATABASE::payment_database (verilmezse --question'dan otomatik cikarilir; --mode open ile birlikte verilemez)")
    parser.add_argument("--question", required=False, default=None, help="soru metni (--clear-conversation verilmedikce zorunlu)")
    parser.add_argument("--max-hops", type=int, default=2, help="Neo4j traversal derinligi (varsayilan: 2, #7.5, sadece --mode pair)")
    parser.add_argument("--seed-topk", type=int, default=20, help="vector seed search topK (varsayilan: 20, #2)")
    parser.add_argument("--top-n", type=int, default=5, help="EvidenceSelector sonrasi kanit sayisi (varsayilan: 5, #4)")
    parser.add_argument("--top-risk", type=int, default=5, help="--mode open: PageRank risk siralamasinda gosterilecek entity sayisi (varsayilan: 5)")
    parser.add_argument("--no-answer", action="store_true", help="Sadece path/risk+evidence goster, Answerer (LLM) cagirma")
    parser.add_argument("--no-memory", action="store_true", help="Konusma hafizasini devre disi birak (soru yeniden yazilmaz, tur kaydedilmez)")
    parser.add_argument("--clear-conversation", action="store_true", help="Kalici konusma gecmisini/ozetini sifirlar ve cikar (--question gerekmez)")
    args = parser.parse_args()
    mode_was_auto_detected = args.mode is None and not args.cause and not args.target

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    HISTORY_FILE = os.path.join(BASE_DIR, "artifacts", "conversation_history.jsonl")
    SUMMARY_FILE = os.path.join(BASE_DIR, "artifacts", "conversation_summary.json")

    if args.clear_conversation:
        conv.clear_conversation(HISTORY_FILE, SUMMARY_FILE)
        print("Konusma gecmisi ve ozeti sifirlandi.")
        sys.exit(0)

    if not args.question:
        parser.error("--question, --clear-conversation verilmedikce zorunludur.")

    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    ANSWERER_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "02_answerer_prompt.md")
    REWRITE_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "09_conversation_rewrite_prompt.md")
    SUMMARY_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "10_conversation_summary_prompt.md")
    MODE_DETECTION_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "11_mode_detection_prompt.md")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    chunks_by_id = load_chunks_by_id(CHUNKS_FILE)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    genai_client = genai.Client(api_key=api_key) if api_key else None
    answer_model = os.environ.get("GEMINI_ANSWERER_MODEL", DEFAULT_ANSWERER_MODEL)

    print("Modeller yukleniyor...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

    effective_question = args.question
    if not args.no_memory and genai_client is not None:
        rewrite_prompt_template = conv.load_prompt_template(REWRITE_PROMPT_FILE)
        effective_question = conv.prepare_question(
            HISTORY_FILE, SUMMARY_FILE, args.question, genai_client, conv.DEFAULT_MODEL, rewrite_prompt_template,
        )
        if effective_question != args.question:
            print(f"\n(Baglamdan yeniden yazilan soru: {effective_question})")

    if mode_was_auto_detected:
        mode_detection_prompt = load_prompt_template(MODE_DETECTION_PROMPT_FILE)
        mode_detection_model = os.environ.get("GEMINI_EXTRACTOR_MODEL", step5b_entity_linker.DEFAULT_MODEL)
        detected_mode = detect_mode_from_question(effective_question, genai_client, mode_detection_model, mode_detection_prompt)
        args.mode = resolve_mode(detected_mode, args.cause, args.target)
    else:
        args.mode = resolve_mode(args.mode, args.cause, args.target)

    if args.mode != "pair" and (args.cause or args.target):
        parser.error(f"--mode {args.mode} ile --cause/--target birlikte verilemez (bu modda A/C onceden bilinmiyor).")

    neo4j_driver = None
    if args.mode != "general":
        neo4j_driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD")),
        )
    qdrant_client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

    seed_chunk_ids = vector_seed_search(qdrant_client, embedding_model, effective_question, args.seed_topk)

    def run_general_mode(question_for_memory):
        """MODE_GENERAL'i calistirir ve cikti basar, sys.exit(0) ile biter - geri donmez.
        question_for_memory: conv.record_turn'e yazilacak orijinal soru (auto-detect sonrasi
        genele dusuldugunde de args.question kullanilir, cunku kullanicinin gordugu soru budur)."""
        evidence_pool = [chunks_by_id[cid] for cid in seed_chunk_ids if cid in chunks_by_id]
        ranked_evidence = rerank_evidence(cross_encoder, effective_question, evidence_pool, args.top_n)

        print("\n=== MODE_GENERAL: Genel Soru-Cevap ===")
        print("(vektor arama + rerank + Answerer - grafa bagli degil, Neo4j kullanilmadi)")

        general_summary = "(Bu genel bir bilgi sorgusu - nedensellik/risk analizi degil, sadece belge kanitina dayanarak cevaplaniyor.)"

        answer = None
        if not args.no_answer:
            if genai_client is None:
                print("\nUYARI: GEMINI_API_KEY tanimli degil, Answerer atlaniyor (--no-answer ile sessizce atlayabilirsin).")
            elif not ranked_evidence:
                print("\nUYARI: Kanit bulunamadi, Answerer atlaniyor.")
            else:
                context_pack = build_context_pack(effective_question, None, ranked_evidence, path_summary_override=general_summary)
                prompt_template = load_prompt_template(ANSWERER_PROMPT_FILE)
                try:
                    answer = call_answerer(genai_client, answer_model, prompt_template, context_pack)
                except RuntimeError as exc:
                    print(f"\nUYARI: Answerer basarisiz oldu: {exc}")

        if answer is not None:
            available_sids = [f"S{i}" for i in range(1, len(ranked_evidence) + 1)]
            citation_issues = validate_citations(answer, available_sids)

            print("\n=== ÇIKTI-1: Answer ===")
            print(f"  answer_text: {answer.answer_text}")
            print(f"  risk_level: {answer.risk_level}")
            print(f"  used_sources: {answer.used_sources}")
            print(f"  assumptions: {answer.assumptions}")
            if citation_issues:
                print("  UYARI (kalite kurali ihlali - docs/01_architecture_v0.1.md #3):")
                for issue in citation_issues:
                    print(f"    - {issue}")

            if not args.no_memory:
                summary_prompt_template = conv.load_prompt_template(SUMMARY_PROMPT_FILE)
                conv.record_turn(
                    HISTORY_FILE, SUMMARY_FILE,
                    {"mode": "general", "question": question_for_memory, "rewritten_question": effective_question,
                     "answer_text": answer.answer_text, "response": None, "source": "cli"},
                    genai_client, conv.DEFAULT_MODEL, summary_prompt_template,
                )

        print(f"\n=== ÇIKTI-3: Evidence Report (S1..S{len(ranked_evidence)}) ===")
        for i, (chunk, score) in enumerate(ranked_evidence, start=1):
            print(f"  S{i} score={score:.2f} chunk_id={chunk['chunk_id']} source_id={chunk['source_id']} location={chunk['location']}")
            print(f"      quote: {chunk['quote']}")

        sys.exit(0)

    if args.mode == "general":
        run_general_mode(args.question)

    if args.mode == "open":
        with neo4j_driver.session() as session:
            ranked_nodes = risk_ranking.compute_risk_ranking(session, RELATION_WEIGHTS, top_k=args.top_risk)
        neo4j_driver.close()

        evidence_pool = risk_ranking.build_risk_evidence_pool(ranked_nodes, seed_chunk_ids, chunks_by_id)
        ranked_evidence = rerank_evidence(cross_encoder, effective_question, evidence_pool, args.top_n)

        print(f"\n=== MODE_A: Risk Ranking (top {args.top_risk}, PageRank) ===")
        if not ranked_nodes:
            print("  (yok - grafta skorlanacak entity bulunamadi)")
        for node in ranked_nodes:
            print(f"  score={node['score']:.2f}  {node['entity_id']}")

        risk_summary = risk_ranking.format_risk_summary(ranked_nodes)

        answer = None
        if not args.no_answer:
            if genai_client is None:
                print("\nUYARI: GEMINI_API_KEY tanimli degil, Answerer atlaniyor (--no-answer ile sessizce atlayabilirsin).")
            elif not ranked_evidence:
                print("\nUYARI: Kanit bulunamadi, Answerer atlaniyor.")
            else:
                context_pack = build_context_pack(effective_question, None, ranked_evidence, path_summary_override=risk_summary)
                prompt_template = load_prompt_template(ANSWERER_PROMPT_FILE)
                try:
                    answer = call_answerer(genai_client, answer_model, prompt_template, context_pack)
                except RuntimeError as exc:
                    print(f"\nUYARI: Answerer basarisiz oldu: {exc}")

        if answer is not None:
            available_sids = [f"S{i}" for i in range(1, len(ranked_evidence) + 1)]
            citation_issues = validate_citations(answer, available_sids)

            print("\n=== ÇIKTI-1: Answer ===")
            print(f"  answer_text: {answer.answer_text}")
            print(f"  risk_level: {answer.risk_level}")
            print(f"  used_sources: {answer.used_sources}")
            print(f"  assumptions: {answer.assumptions}")
            if citation_issues:
                print("  UYARI (kalite kurali ihlali - docs/01_architecture_v0.1.md #3):")
                for issue in citation_issues:
                    print(f"    - {issue}")

            if not args.no_memory:
                summary_prompt_template = conv.load_prompt_template(SUMMARY_PROMPT_FILE)
                conv.record_turn(
                    HISTORY_FILE, SUMMARY_FILE,
                    {"mode": args.mode, "question": args.question, "rewritten_question": effective_question,
                     "answer_text": answer.answer_text, "response": None, "source": "cli"},
                    genai_client, conv.DEFAULT_MODEL, summary_prompt_template,
                )

        print(f"\n=== ÇIKTI-3: Evidence Report (S1..S{len(ranked_evidence)}) ===")
        for i, (chunk, score) in enumerate(ranked_evidence, start=1):
            print(f"  S{i} score={score:.2f} chunk_id={chunk['chunk_id']} source_id={chunk['source_id']} location={chunk['location']}")
            print(f"      quote: {chunk['quote']}")

        sys.exit(0)

    # --- MODE_B (--mode pair) ---
    cause_id, target_id = args.cause, args.target
    if cause_id is None or target_id is None:
        if genai_client is None:
            print("\nHATA: Otomatik varlik eslestirme icin GEMINI_API_KEY gerekli.")
            print("Ya .env dosyasina bir anahtar ekleyin, ya da --cause/--target'i elle verin.")
            sys.exit(1)

        linker_prompt_file = os.path.join(BASE_DIR, "docs", "prompts", "07_entity_linker_prompt.md")
        linker_prompt = step5b_entity_linker.load_prompt_template(linker_prompt_file)
        linker_model = os.environ.get("GEMINI_EXTRACTOR_MODEL", step5b_entity_linker.DEFAULT_MODEL)

        phrases = step5b_entity_linker.call_entity_linker(genai_client, linker_model, linker_prompt, effective_question)

        failures = []
        if cause_id is None:
            resolved_id, score, candidates = step5b_entity_linker.resolve_phrase_to_entity(
                phrases.cause_phrase, qdrant_client, embedding_model
            )
            if resolved_id is None:
                failures.append({"role": "cause", "phrase": phrases.cause_phrase, "candidates": candidates})
            else:
                cause_id = resolved_id
                print(f"\nOtomatik eslestirme: '{phrases.cause_phrase}' -> {cause_id} (skor={score:.2f})")

        if target_id is None:
            resolved_id, score, candidates = step5b_entity_linker.resolve_phrase_to_entity(
                phrases.target_phrase, qdrant_client, embedding_model
            )
            if resolved_id is None:
                failures.append({"role": "target", "phrase": phrases.target_phrase, "candidates": candidates})
            else:
                target_id = resolved_id
                print(f"Otomatik eslestirme: '{phrases.target_phrase}' -> {target_id} (skor={score:.2f})")

        if failures:
            if mode_was_auto_detected:
                print("\n(Otomatik algilanan nedensel mod icin varlik eslestirme basarisiz oldu, genel moda dusuluyor.)")
                if neo4j_driver is not None:
                    neo4j_driver.close()
                run_general_mode(args.question)
            print("\nHATA: Otomatik varlik eslestirme basarisiz.")
            for failure in failures:
                print(f"  '{failure['phrase']}' ({failure['role']}) icin guvenilir bir eslesme bulunamadi. En yakin adaylar:")
                for entity_id, name, entity_type, cand_score in failure["candidates"]:
                    print(f"    - {entity_id} ({name}, skor={cand_score:.2f})")
            print("  --cause/--target'i elle vererek devam edebilirsiniz.")
            sys.exit(1)

    with neo4j_driver.session() as session:
        candidate_paths = find_candidate_paths(session, cause_id, target_id, args.max_hops)
    neo4j_driver.close()

    evidence_pool = build_evidence_pool(candidate_paths, seed_chunk_ids, chunks_by_id)
    ranked_evidence = rerank_evidence(cross_encoder, effective_question, evidence_pool, args.top_n)

    print(f"\n=== Candidate Paths ({len(candidate_paths)}, max_hops={args.max_hops}) ===")
    if not candidate_paths:
        print(f"  (yok - {cause_id} -> {target_id} arasinda {args.max_hops} hop icinde path bulunamadi)")
    for path in candidate_paths:
        print(f"  score={path['score']:.2f}  {format_path(path)}")

    best_path = candidate_paths[0] if candidate_paths else None

    # ÇIKTI-1: Final Answer (Modul F - Answerer)
    answer = None
    if not args.no_answer:
        if genai_client is None:
            print("\nUYARI: GEMINI_API_KEY tanimli degil, Answerer atlaniyor (--no-answer ile sessizce atlayabilirsin).")
        elif not ranked_evidence:
            print("\nUYARI: Kanit bulunamadi, Answerer atlaniyor.")
        else:
            context_pack = build_context_pack(effective_question, best_path, ranked_evidence)
            prompt_template = load_prompt_template(ANSWERER_PROMPT_FILE)
            try:
                answer = call_answerer(genai_client, answer_model, prompt_template, context_pack)
            except RuntimeError as exc:
                print(f"\nUYARI: Answerer basarisiz oldu: {exc}")

    if answer is not None:
        available_sids = [f"S{i}" for i in range(1, len(ranked_evidence) + 1)]
        citation_issues = validate_citations(answer, available_sids)

        print("\n=== ÇIKTI-1: Answer ===")
        print(f"  answer_text: {answer.answer_text}")
        print(f"  risk_level: {answer.risk_level}")
        print(f"  used_sources: {answer.used_sources}")
        print(f"  assumptions: {answer.assumptions}")
        if citation_issues:
            print("  UYARI (kalite kurali ihlali - docs/01_architecture_v0.1.md #3):")
            for issue in citation_issues:
                print(f"    - {issue}")

        if not args.no_memory:
            summary_prompt_template = conv.load_prompt_template(SUMMARY_PROMPT_FILE)
            conv.record_turn(
                HISTORY_FILE, SUMMARY_FILE,
                {"mode": args.mode, "question": args.question, "rewritten_question": effective_question,
                 "answer_text": answer.answer_text, "response": None, "source": "cli"},
                genai_client, conv.DEFAULT_MODEL, summary_prompt_template,
            )

    print("\n=== ÇIKTI-2: Reasoning Path ===")
    print(f"  {format_path(best_path) if best_path else '(yok)'}")

    print(f"\n=== ÇIKTI-3: Evidence Report (S1..S{len(ranked_evidence)}) ===")
    for i, (chunk, score) in enumerate(ranked_evidence, start=1):
        print(f"  S{i} score={score:.2f} chunk_id={chunk['chunk_id']} source_id={chunk['source_id']} location={chunk['location']}")
        print(f"      quote: {chunk['quote']}")
