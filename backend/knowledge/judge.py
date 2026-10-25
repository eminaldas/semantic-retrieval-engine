"""Ajan cevabinin kalitesi icin LLM yargic: cevap getirilen kanita dayaniyor mu (grounded),
altin setteki referansla uyusuyor mu (correct).

Etiketler ikili (gecti/kaldi); 1-5 puanlar yargiclar arasinda tutarsiz ve neyin duzeltilecegini
soylemiyor. Yargic kanit olarak sadece ajanin araclardan aldigi metni gorur, yani 'grounded'
genel bilgiyle dogru olan ama belgelerde olmayan iddiayi da yakalar. Yargic da yanilabilir:
etiketleri bir kez elle kontrol edilmis kayitlarla karsilastirip uyum orani raporlanmali
(docs/olcum notlari)."""

import json

from backend.shared import tracing

JUDGE_MODEL = "gemini-flash-latest"
CALL_SITE = "knowledge_judge"
MAX_EVIDENCE_CHARS = 24000  # ~6k token; ajanin tum arac ciktilari genelde bunun altinda

PROMPT = """Bir proje belgeleri asistaninin cevabini degerlendiriyorsun. Sadece asagidaki KANIT
metnini bilgi kaynagi say; kendi genel bilgini kullanma.

SORU:
{question}

KANIT (asistanin araclardan aldigi belge parcalari):
{evidence}

CEVAP:
{answer}

{reference_block}Karar ver:
- grounded: Cevaptaki her olgusal iddia KANIT'ta acikca yer aliyor mu? Kaynak etiketleri, baglac
  ve ozet cumleleri iddia sayilmaz. Tek bir desteksiz iddia varsa false.
- correct: {correct_rule}
- unsupported: KANIT'ta olmayan iddialar (kisa, en fazla 3; yoksa bos liste).
- reason: tek cumle gerekce.
JSON dondur."""

_REFERENCE = "BEKLENEN BILGI (altin set):\n{reference}\n\n"
_CORRECT_WITH_REF = "Cevap BEKLENEN BILGI'yi iceriyor ya da onunla ayni sonucu soyluyor mu? Celisiyorsa ya da eksikse false."
_CORRECT_NO_REF = "Cevap soruyu dogrudan yanitliyor mu? Konu disina kaciyorsa ya da soruyu yanitlamiyorsa false."

SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "correct": {"type": "boolean"},
        "unsupported": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["grounded", "correct", "unsupported", "reason"],
}


def build_prompt(question, evidence, answer, reference=None):
    text = "\n\n---\n\n".join(evidence)
    if len(text) > MAX_EVIDENCE_CHARS:
        text = text[:MAX_EVIDENCE_CHARS] + "\n[... kesildi]"
    if isinstance(reference, list):
        reference = " / ".join(reference)
    return PROMPT.format(
        question=question, evidence=text or "(asistan hicbir arac kullanmadi)", answer=answer,
        reference_block=_REFERENCE.format(reference=reference) if reference else "",
        correct_rule=_CORRECT_WITH_REF if reference else _CORRECT_NO_REF)


def parse_verdict(raw):
    """Model ciktisini dogrular; bozuksa yargic 'karar veremedi' doner, olcume kaldi diye girmez."""
    try:
        data = json.loads(raw)
        return {"grounded": bool(data["grounded"]), "correct": bool(data["correct"]),
                "unsupported": [str(x) for x in data.get("unsupported", [])][:3],
                "reason": str(data.get("reason", "")), "error": None}
    except (ValueError, KeyError, TypeError) as exc:
        return {"grounded": None, "correct": None, "unsupported": [], "reason": "", "error": f"{type(exc).__name__}: {exc}"}


def judge_answer(client, question, evidence, answer, reference=None, model=JUDGE_MODEL):
    from google.genai import types

    from backend.shared import gemini_retry, gemini_usage

    prompt = build_prompt(question, evidence, answer, reference)
    config = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                         response_schema=SCHEMA)

    def _call():
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        gemini_usage.log_usage(CALL_SITE, model, response)
        return response

    with tracing.span("judge", tracing.LLM, **{"llm.model_name": model, "input.value": tracing.clip(question)}) as span:
        response = gemini_retry.call_with_retry(_call, pacing_key=CALL_SITE)
        verdict = parse_verdict(response.text or "")
        usage = response.usage_metadata
        verdict["prompt_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
        verdict["output_tokens"] = getattr(usage, "candidates_token_count", 0) or 0
        span.update({"rag.grounded": verdict["grounded"], "rag.correct": verdict["correct"],
                     "output.value": verdict["reason"], "llm.token_count.prompt": verdict["prompt_tokens"],
                     "llm.token_count.completion": verdict["output_tokens"]})
    return verdict
