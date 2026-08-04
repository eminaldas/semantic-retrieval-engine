import json
import os
import sys
from typing import List, Literal

# google-genai paketi __init__.py'sinde kullanmadigimiz buyuk bir alt modulu
# (_gaos, "Enterprise Agent Platform") de eager import ediyor; ilk calistirmada
# bos ekranda birkac saniye sessiz kalinabiliyor - donmus degil, sadece sessiz.
print("Modüller yükleniyor (ilk çalıştırmada birkaç saniye sürebilir)...", flush=True)

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from backend.shared import gemini_retry
from backend.shared import gemini_usage

# Ontology v0.1 (docs/01_architecture_v0.1.md #4) - bu listeler dışına cikilamaz.
EntityType = Literal[
    "SERVICE", "COMPONENT", "DATABASE", "QUEUE", "MECHANISM", "FAILURE_MODE", "METRIC"
]
RelationType = Literal[
    "TRIGGERS", "CAUSES", "LEADS_TO", "IMPACTS", "RISKS", "DEPENDS_ON",
    "USES", "DEGRADES", "IMPROVES", "MITIGATES", "PREVENTS",
]

MIN_CONFIDENCE = 0.55  # docs/01_architecture_v0.1.md #5
DEFAULT_MODEL = "gemini-flash-lite-latest"


class EntityRef(BaseModel):
    name: str
    type: EntityType


class ExtractedEntity(EntityRef):
    aliases: List[str] = []
    confidence: float


class ExtractedRelation(BaseModel):
    source: EntityRef
    relation_type: RelationType
    target: EntityRef
    confidence: float


class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]


def load_prompt_template(prompt_path):
    """docs/prompts/01_extraction_prompt.md icindeki fenced code block'u okur."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.index("```\n") + len("```\n")
    end = content.rindex("```")
    return content[start:end].strip()


def load_chunks(chunks_file):
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def already_extracted_chunk_ids(extraction_file):
    """extraction_file'da (varsa) hangi chunk_id'lerin zaten islendigini dondurur."""
    if not os.path.exists(extraction_file):
        return set()
    ids = set()
    with open(extraction_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["chunk_id"])
    return ids


def filter_by_confidence(result: ExtractionResult) -> ExtractionResult:
    """Confidence filtresi: docs/01_architecture_v0.1.md #5 - 0.55 altı graph'a yazilmaz."""
    return ExtractionResult(
        entities=[e for e in result.entities if e.confidence >= MIN_CONFIDENCE],
        relations=[r for r in result.relations if r.confidence >= MIN_CONFIDENCE],
    )


def extract_chunk(client, model, prompt_template, chunk_text, min_interval_seconds=0.0):
    prompt = prompt_template.replace("{CHUNK_TEXT}", chunk_text)

    def _call():
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
            ),
        )
        gemini_usage.log_usage("step2_extractor", model, response)
        result = response.parsed or ExtractionResult.model_validate_json(response.text)
        return filter_by_confidence(result)

    return gemini_retry.call_with_retry(_call, min_interval_seconds=min_interval_seconds, pacing_key="step2_extractor")


def save_extraction(chunk_id, result: ExtractionResult, output_handle):
    """artifacts/extraction.jsonl semasi: docs/01_architecture_v0.1.md #10.2"""
    record = {
        "chunk_id": chunk_id,
        "entities": [e.model_dump() for e in result.entities],
        "relations": [r.model_dump() for r in result.relations],
    }
    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
    OUTPUT_FILE = os.path.join(BASE_DIR, "artifacts", "extraction.jsonl")
    PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "01_extraction_prompt.md")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("HATA: GEMINI_API_KEY (veya GOOGLE_API_KEY) tanimli degil.")
        print(f"Proje kokunde bir .env dosyasi olustur (bkz. .env.example) veya ortam degiskeni ata.")
        sys.exit(1)

    if not os.path.exists(CHUNKS_FILE):
        print(f"HATA: {CHUNKS_FILE} bulunamadi. Once step1_chunker.py calistirilmali.")
        sys.exit(1)

    model_name = os.environ.get("GEMINI_EXTRACTOR_MODEL", DEFAULT_MODEL)
    min_interval = float(os.environ.get("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "1.0"))
    client = genai.Client(api_key=api_key)
    prompt_template = load_prompt_template(PROMPT_FILE)
    chunks = list(load_chunks(CHUNKS_FILE))

    already_done = already_extracted_chunk_ids(OUTPUT_FILE)
    new_chunks = [c for c in chunks if c["chunk_id"] not in already_done]

    if not new_chunks:
        print(f"Islenecek yeni chunk yok - {len(chunks)} chunk'in hepsi zaten extract edilmis.")
        sys.exit(0)

    print(
        f"{len(new_chunks)} yeni chunk, model={model_name} ile islenecek "
        f"(toplam {len(chunks)}, {len(already_done)} zaten islenmis)..."
    )

    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
    with open(OUTPUT_FILE, mode, encoding="utf-8") as out:
        for i, chunk in enumerate(new_chunks, start=1):
            chunk_id = chunk["chunk_id"]
            try:
                result = extract_chunk(client, model_name, prompt_template, chunk["text"], min_interval_seconds=min_interval)
            except RuntimeError as exc:
                print(f"  [{i}/{len(new_chunks)}] {chunk_id}: ATLANDI - {exc}")
                continue

            save_extraction(chunk_id, result, out)
            print(
                f"  [{i}/{len(new_chunks)}] {chunk_id}: "
                f"{len(result.entities)} entity, {len(result.relations)} relation"
            )

    print(f"\nBASARILI! Cikti: {OUTPUT_FILE}")
