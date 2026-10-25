"""Soru parcalama (query decomposition): iki ayri bilgiyi birlestiren soruyu, belgelerde ayri ayri
aranacak kisa sorgulara boler. Tek sorguyla aranan cok adimli sorularda ikinci belge aday havuzuna
hic girmiyordu (factprobe zor set h026, h031; 2026-10-25); belge basina sinir bunu cozmedi, cunku
eksik olan siralama degil getirmeydi. Her parca ayri getirilip ayri rerank edilir (search.subqueries).

Tek bilgiyi soran soruda model bos liste dondurur; arama eskisi gibi tek sorguyla calisir.
Gemini yoksa ya da hata verirse de bos liste doner: arama parcalamasiz devam eder."""

import json

from backend.shared import tracing

DEFAULT_MODEL = "gemini-flash-lite-latest"
CALL_SITE = "knowledge_decompose"
MAX_PARTS = 3

PROMPT = """Bir proje belgeleri arama sistemine gelen soruyu, belgelerde ayri ayri aranmasi gereken
parcalara bol.
- Soru tek bir bilgiyi soruyorsa bolme: queries bos liste olsun.
- Soru birden fazla ayri bilgiyi birlestiriyorsa (iki belge, iki tarih ya da asama, bir karsilastirma,
  "X'te ne denmis, Y'de nasil degismis" gibi) her bilgi icin kisa ve tek basina anlasilir bir arama
  sorgusu yaz, en fazla {max_parts}.
- Sorudaki ozel adlari, sayilari, tarihleri ve dosya adlarini aynen koru. Soruda olmayan bilgi ekleme.

Soru: {question}"""

SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}


def parse_queries(raw, question, max_parts=MAX_PARTS):
    """Model ciktisini temizler: bos, tekrar eden ve sorunun kendisiyle ayni sorgular atilir.
    Tek parca kaldiysa bolme anlamsizdir, bos liste doner."""
    try:
        items = json.loads(raw).get("queries", [])
    except (ValueError, AttributeError):
        return []
    seen, queries = {question.strip().casefold()}, []
    for item in items if isinstance(items, list) else []:
        text = str(item).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            queries.append(text)
    return queries[:max_parts] if len(queries) > 1 else []


def decompose(client, question, model=DEFAULT_MODEL, max_parts=MAX_PARTS):
    """Soruyu arama sorgularina boler; bolunmeyecekse ya da hata olursa []."""
    if client is None:
        return []
    from google.genai import types

    from backend.shared import gemini_retry, gemini_usage

    config = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                         response_schema=SCHEMA)
    prompt = PROMPT.format(question=question, max_parts=max_parts)

    def _call():
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        gemini_usage.log_usage(CALL_SITE, model, response)
        return response

    with tracing.span("decompose", tracing.LLM, **{"llm.model_name": model,
                                                   "input.value": tracing.clip(question)}) as span:
        try:
            response = gemini_retry.call_with_retry(_call, pacing_key=CALL_SITE)
        except Exception as exc:  # parcalama bir iyilestirme; arama onsuz da calismali
            span.set("rag.error", f"{type(exc).__name__}: {exc}")
            return []
        queries = parse_queries(response.text or "", question, max_parts)
        span.update({"output.value": queries or None, "rag.subquery_count": len(queries)})
    return queries
