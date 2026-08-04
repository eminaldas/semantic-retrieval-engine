import json
import os
import sys

print("Modüller yükleniyor (ilk çalıştırmada birkaç saniye sürebilir)...", flush=True)

from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
from neo4j import GraphDatabase
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Literal

from backend.pipeline.step3_graph_writer import RELATION_TYPES
from backend.shared import gemini_retry
from backend.shared import gemini_usage
from backend.shared.paths import PROJECT_ROOT

STAGE1_MODEL = "gemini-flash-lite-latest"
DEBATE_MODEL = "gemini-flash-latest"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # step1/step4/step5b ile ayni
NEGATION_SIMILARITY_THRESHOLD = 0.5  # devam 39'da canli arastirmayla kalibre edilecek, bkz. Global Constraints


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _make_embed_fn(embedding_model):
    def embed_fn(text):
        return embedding_model.encode(text)
    return embed_fn


def build_conflict_candidates(rows, new_chunk_ids, chunks_by_id, embed_fn, similarity_threshold):
    """docs/superpowers/specs/2026-08-02-self-correction-negation-blindspot-design.md:
    artik (kaynak, hedef) cifti sabit degil - her SATIR bir kaynak entity'nin TUM
    giden kenarlarini (herhangi bir hedef) iceriyor. Her yeni kenar icin, kanit
    metni anlamsal olarak benzer olan eski kenarlar pair_similar_edges ile
    bulunuyor; benzer eski kenari olmayan yeni kenarlar (alakasiz, birlikte var
    olan iliskiler) hic aday olusturmuyor - LLM cagri hacmini sinirlayan ucuz
    on-filtre budur."""
    candidates = []
    for row in rows:
        edges = [e for e in row["edges"] if e.get("status", "ACTIVE") == "ACTIVE"]

        old_edges = []
        new_edges = []
        for edge in edges:
            old_ids = [cid for cid in edge["evidence_chunk_ids"] if cid not in new_chunk_ids]
            new_ids = [cid for cid in edge["evidence_chunk_ids"] if cid in new_chunk_ids]
            if old_ids:
                old_edges.append({**edge, "evidence_chunk_ids": old_ids})
            if new_ids:
                new_edges.append({**edge, "evidence_chunk_ids": new_ids})

        if not old_edges or not new_edges:
            continue

        for new_edge in new_edges:
            similar_old_edges = pair_similar_edges(old_edges, new_edge, chunks_by_id, embed_fn, similarity_threshold)
            if similar_old_edges:
                candidates.append({
                    "source_id": row["source_id"], "source_name": row["source_name"],
                    "new_edge": new_edge, "old_edges": similar_old_edges,
                })
    return candidates


def _edge_evidence_text(edge, chunks_by_id):
    """Bir kenarin kanit chunk metinlerini birlestirir (embedding karsilastirmasi icin)."""
    texts = [chunks_by_id[cid]["text"] for cid in edge["evidence_chunk_ids"] if cid in chunks_by_id]
    return " ".join(texts)


def _cosine_similarity(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def pair_similar_edges(old_edges, new_edge, chunks_by_id, embed_fn, threshold):
    """new_edge'in kanit metniyle anlamsal olarak yeterince benzer olan eski
    kenarlari dondurur (kosinus benzerligi >= threshold). Bos liste donerse bu
    yeni kenar icin hicbir aday olusturulmaz - pahali Stage1 LLM cagrisi hic
    tetiklenmez."""
    new_text = _edge_evidence_text(new_edge, chunks_by_id)
    if not new_text:
        return []
    new_vec = embed_fn(new_text)

    similar = []
    for old_edge in old_edges:
        old_text = _edge_evidence_text(old_edge, chunks_by_id)
        if not old_text:
            continue
        old_vec = embed_fn(old_text)
        if _cosine_similarity(new_vec, old_vec) >= threshold:
            similar.append(old_edge)
    return similar


def build_conflict_candidate_query():
    """docs/superpowers/specs/2026-08-02-self-correction-negation-blindspot-design.md:
    artik (a,b) ciftini degil, sadece kaynak entity a'yi grupluyor - a'nin TUM
    giden kenarlari (herhangi bir hedef) tek satirda toplaniyor, her kenar kendi
    target_id/target_name'ini de tasiyor. Bu, 'ayni kaynak, farkli hedef'
    negasyonlarinin (orn. recovery.conf -> recovery yerine recovery.conf ->
    Database Server) artik ayni satirda gorulebilmesini sagliyor - eskiden (a,b)
    sabit oldugu icin bu iki kenar hic kesismiyordu (devam 20)."""
    return (
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "WITH a, collect({type: type(r), confidence: r.confidence, "
        "evidence_chunk_ids: r.evidence_chunk_ids, status: coalesce(r.status, 'ACTIVE'), "
        "target_id: b.entity_id, target_name: b.name}) AS edges "
        "WHERE ANY(e IN edges WHERE ANY(cid IN e.evidence_chunk_ids WHERE cid IN $new_chunk_ids)) "
        "RETURN a.entity_id AS source_id, a.name AS source_name, edges"
    )


def format_relation_types(edges):
    return ", ".join(sorted({e["type"] for e in edges}))


def format_evidence_text(edges, chunks_by_id):
    lines = []
    for edge in edges:
        for chunk_id in edge["evidence_chunk_ids"]:
            chunk = chunks_by_id.get(chunk_id)
            if chunk:
                lines.append(f"[{edge['type']} -> {edge['target_name']}, {chunk_id}]: {chunk['text']}")
    return "\n".join(lines)


def build_prompt_vars(source_name, old_edges, new_edge, chunks_by_id):
    return {
        "{SOURCE_NAME}": source_name,
        "{OLD_RELATION_TYPES}": format_relation_types(old_edges),
        "{NEW_RELATION_TYPE}": new_edge["type"],
        "{NEW_TARGET_NAME}": new_edge["target_name"],
        "{OLD_EVIDENCE_TEXT}": format_evidence_text(old_edges, chunks_by_id),
        "{NEW_EVIDENCE_TEXT}": format_evidence_text([new_edge], chunks_by_id),
    }


def fill_prompt(template, variables):
    for placeholder, value in variables.items():
        template = template.replace(placeholder, value)
    return template


def load_prompt_template(prompt_path):
    """docs/prompts/*.md icindeki fenced code block'u okur (step2/step5 ile ayni sozlesme)."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.index("```\n") + len("```\n")
    end = content.rindex("```")
    return content[start:end].strip()


class ConflictClassification(BaseModel):
    classification: Literal["CONSISTENT", "CONTRADICTS", "COEXISTS"]
    reasoning: str


class ThesisArgument(BaseModel):
    argument: str


class AntithesisArgument(BaseModel):
    argument: str


class JudgeVerdict(BaseModel):
    verdict: Literal["KEEP_OLD", "REPLACE_WITH_NEW", "BOTH_VALID", "UNCERTAIN"]
    reasoning: str


def call_llm(client, model, prompt_template, prompt_vars, response_model):
    prompt = fill_prompt(prompt_template, prompt_vars)

    def _call():
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        gemini_usage.log_usage("step6_self_correction", model, response)
        return response.parsed or response_model.model_validate_json(response.text)

    return gemini_retry.call_with_retry(_call, pacing_key="step6_self_correction")


def supersede_old_edges(tx, source_id, old_edges):
    """Hakem REPLACE_WITH_NEW derse eski edge'leri status=SUPERSEDED yapar (hard-delete yok).
    Her old_edge kendi target_id'sini tasir - artik hepsi ayni hedefe isaret etmek zorunda degil."""
    for edge in old_edges:
        relation_type = edge["type"]
        if relation_type not in RELATION_TYPES:
            raise ValueError(f"Ontoloji disi relation_type: {relation_type}")
        query = f"""
            MATCH (a:Entity {{entity_id: $source_id}})-[r:{relation_type}]->(b:Entity {{entity_id: $target_id}})
            SET r.status = 'SUPERSEDED'
        """
        tx.run(query, source_id=source_id, target_id=edge["target_id"])


if __name__ == "__main__":
    BASE_DIR = PROJECT_ROOT
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    EXTRACTION_FILE = os.path.join(BASE_DIR, "artifacts", "extraction.jsonl")
    NEW_CHUNK_IDS_FILE = os.path.join(BASE_DIR, "artifacts", "new_chunk_ids.json")
    LOG_FILE = os.path.join(BASE_DIR, "artifacts", "self_correction_log.jsonl")
    PROMPTS_DIR = os.path.join(BASE_DIR, "docs", "prompts")

    load_dotenv(os.path.join(BASE_DIR, ".env"))

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("HATA: GEMINI_API_KEY (veya GOOGLE_API_KEY) tanimli degil.")
        sys.exit(1)

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        print("HATA: NEO4J_PASSWORD tanimli degil.")
        sys.exit(1)

    for required_file in (CHUNKS_FILE, EXTRACTION_FILE, NEW_CHUNK_IDS_FILE):
        if not os.path.exists(required_file):
            print(f"HATA: {required_file} bulunamadi. Once step1-3 calistirilmali (guncel step3_graph_writer.py, new_chunk_ids.json'i o yazar).")
            sys.exit(1)

    chunks_by_id = {c["chunk_id"]: c for c in load_jsonl(CHUNKS_FILE)}
    with open(NEW_CHUNK_IDS_FILE, "r", encoding="utf-8") as f:
        new_chunk_ids = set(json.load(f))

    classifier_template = load_prompt_template(os.path.join(PROMPTS_DIR, "03_conflict_classifier_prompt.md"))
    thesis_template = load_prompt_template(os.path.join(PROMPTS_DIR, "04_thesis_prompt.md"))
    antithesis_template = load_prompt_template(os.path.join(PROMPTS_DIR, "05_antithesis_prompt.md"))
    judge_template = load_prompt_template(os.path.join(PROMPTS_DIR, "06_judge_prompt.md"))

    client = genai.Client(api_key=api_key)
    classifier_model = os.environ.get("GEMINI_CONFLICT_CLASSIFIER_MODEL", STAGE1_MODEL)
    debate_model = os.environ.get("GEMINI_DEBATE_MODEL", DEBATE_MODEL)

    print("Embedding modeli yukleniyor...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embed_fn = _make_embed_fn(embedding_model)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            rows = [dict(record) for record in session.run(
                build_conflict_candidate_query(), new_chunk_ids=list(new_chunk_ids)
            )]
            candidates = build_conflict_candidates(
                rows, new_chunk_ids, chunks_by_id, embed_fn, NEGATION_SIMILARITY_THRESHOLD
            )
            print(f"{len(candidates)} celiski adayi bulundu.")

            with open(LOG_FILE, "w", encoding="utf-8") as log:
                for i, candidate in enumerate(candidates, start=1):
                    prompt_vars = build_prompt_vars(
                        candidate["source_name"], candidate["old_edges"], candidate["new_edge"], chunks_by_id,
                    )
                    try:
                        classification = call_llm(
                            client, classifier_model, classifier_template, prompt_vars, ConflictClassification
                        )

                        log_entry = {
                            "source_id": candidate["source_id"],
                            "new_target_id": candidate["new_edge"]["target_id"],
                            "new_relation_type": candidate["new_edge"]["type"],
                            "old_target_ids": sorted({e["target_id"] for e in candidate["old_edges"]}),
                            "old_relation_types": sorted({e["type"] for e in candidate["old_edges"]}),
                            "stage1_classification": classification.classification,
                            "stage1_reasoning": classification.reasoning,
                        }

                        print(
                            f"  [{i}/{len(candidates)}] {candidate['source_name']} -> "
                            f"{candidate['new_edge']['target_name']}: {classification.classification}"
                        )

                        if classification.classification == "CONTRADICTS":
                            thesis = call_llm(client, debate_model, thesis_template, prompt_vars, ThesisArgument)
                            antithesis = call_llm(client, debate_model, antithesis_template, prompt_vars, AntithesisArgument)

                            judge_vars = dict(prompt_vars)
                            judge_vars["{THESIS_ARGUMENT}"] = thesis.argument
                            judge_vars["{ANTITHESIS_ARGUMENT}"] = antithesis.argument
                            verdict = call_llm(client, debate_model, judge_template, judge_vars, JudgeVerdict)

                            log_entry["thesis"] = thesis.argument
                            log_entry["antithesis"] = antithesis.argument
                            log_entry["verdict"] = verdict.verdict
                            log_entry["verdict_reasoning"] = verdict.reasoning

                            print(f"      Hakem karari: {verdict.verdict}")

                            if verdict.verdict == "REPLACE_WITH_NEW":
                                session.execute_write(
                                    supersede_old_edges, candidate["source_id"], candidate["old_edges"]
                                )
                                print("      Eski edge(lar) SUPERSEDED olarak isaretlendi.")
                    except RuntimeError as exc:
                        print(
                            f"  [{i}/{len(candidates)}] {candidate['source_name']} -> "
                            f"{candidate['new_edge']['target_name']}: ATLANDI - {exc}"
                        )
                        continue

                    log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            print(f"\nBASARILI! Log: {LOG_FILE}")
    finally:
        driver.close()
