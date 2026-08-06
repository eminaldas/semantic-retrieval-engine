"""Dogal dil sorusundan A (cause) ve C (target) entity_id'lerini otomatik cozer.
docs/superpowers/specs/2026-07-24-entity-linking-design.md'deki tasarima gore."""

from google.genai import types
from pydantic import BaseModel

from backend.shared import gemini_retry
from backend.shared import gemini_usage

DEFAULT_MODEL = "gemini-flash-lite-latest"  # extractor.py ile ayni varsayilan

ENTITY_COLLECTION_NAME = "entities"  # graph_writer.py::ENTITY_COLLECTION_NAME ile ayni
ENTITY_SIMILARITY_THRESHOLD = 0.85  # graph_writer.py::ENTITY_SIMILARITY_THRESHOLD ile ayni baslangic degeri


def load_prompt_template(prompt_path):
    """docs/prompts/*.md icindeki fenced code block'u okur (step2/step5/step6 ile ayni sozlesme)."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.index("```\n") + len("```\n")
    end = content.rindex("```")
    return content[start:end].strip()


def resolve_phrase_to_entity(phrase, qdrant_client, embedding_model, threshold=ENTITY_SIMILARITY_THRESHOLD, top_k=3):
    """Bir soru-ifadesini (orn. 'Auth servisi') mevcut entity_id'lerden birine cozer.
    Tip filtresi yok - LLM ifadeyi cikarirken entity tipini bilmiyor."""
    vector = embedding_model.encode(phrase)
    vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    hits = qdrant_client.query_points(
        collection_name=ENTITY_COLLECTION_NAME,
        query=vector,
        limit=top_k,
    ).points

    candidates = [(hit.payload["entity_id"], hit.payload["name"], hit.payload["type"], hit.score) for hit in hits]

    if candidates and candidates[0][3] >= threshold:
        entity_id, _, _, score = candidates[0]
        return entity_id, score, candidates

    return None, None, candidates


class EntityLinkPhrases(BaseModel):
    """LLM'in cikarttigi A (cause) ve C (target) ifadeleri."""
    cause_phrase: str
    target_phrase: str


class EntityLinkResult:
    """Basarilı entity linking sonucu."""
    def __init__(self, cause_entity_id, cause_phrase, cause_score, target_entity_id, target_phrase, target_score):
        self.cause_entity_id = cause_entity_id
        self.cause_phrase = cause_phrase
        self.cause_score = cause_score
        self.target_entity_id = target_entity_id
        self.target_phrase = target_phrase
        self.target_score = target_score


class EntityLinkError(Exception):
    """Bir veya her iki tarafin entity linking'i basarisiz oldugunda atilir."""
    def __init__(self, failures):
        self.failures = failures
        super().__init__("Entity linking basarisiz: " + ", ".join(f["role"] for f in failures))


def call_entity_linker(client, model, prompt_template, question):
    """Canli servise bagli - unit test yok, bkz. docs/superpowers/plans/2026-07-24-entity-linking.md
    Global Constraints. Retry/backoff artik gemini_retry.py'de ortaklastirildi (devam 38)."""
    prompt = prompt_template.replace("{QUESTION}", question)

    def _call():
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EntityLinkPhrases,
            ),
        )
        gemini_usage.log_usage("step5b_entity_linker", model, response)
        return response.parsed or EntityLinkPhrases.model_validate_json(response.text)

    return gemini_retry.call_with_retry(_call, pacing_key="step5b_entity_linker")


def link_entities(question, qdrant_client, embedding_model, gemini_client, model_name, prompt_template, threshold=ENTITY_SIMILARITY_THRESHOLD):
    """Dogal dil sorusundan A ve C entity'lerini bulur. Basarisiz olursa EntityLinkError atar.

    Args:
        question: Dogal dil sorusu (orn. "Auth servisi kokerse odeme veritabanini nasil etkiler?")
        qdrant_client: Qdrant vector DB istemcisi
        embedding_model: Sentence Transformer embedding modeli
        gemini_client: Google Gemini islemcisi
        model_name: Gemini model adı (orn. "gemini-flash-lite-latest")
        prompt_template: LLM prompt template'i (orn. docs/prompts/07_entity_linker_prompt.md'den)
        threshold: Entity matching threshold'i (varsayilan ENTITY_SIMILARITY_THRESHOLD)

    Returns:
        EntityLinkResult: cause_entity_id, target_entity_id ve skorlari

    Raises:
        EntityLinkError: A veya C entity'si bulmak basarisiz olursa, failures listesi ile
    """
    phrases = call_entity_linker(gemini_client, model_name, prompt_template, question)

    cause_id, cause_score, cause_candidates = resolve_phrase_to_entity(
        phrases.cause_phrase, qdrant_client, embedding_model, threshold
    )
    target_id, target_score, target_candidates = resolve_phrase_to_entity(
        phrases.target_phrase, qdrant_client, embedding_model, threshold
    )

    failures = []
    if cause_id is None:
        failures.append({"role": "cause", "phrase": phrases.cause_phrase, "candidates": cause_candidates})
    if target_id is None:
        failures.append({"role": "target", "phrase": phrases.target_phrase, "candidates": target_candidates})
    if failures:
        raise EntityLinkError(failures)

    return EntityLinkResult(cause_id, phrases.cause_phrase, cause_score, target_id, phrases.target_phrase, target_score)
