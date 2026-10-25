"""Uzun baglam taban cizgisi: arama yok, projenin tum belgeleri tek seferde modele verilir.

factprobe (~78 belge) Gemini'nin baglamina sigiyor. Ayni altin seti, ayni puanlama ve ayni yargicla
ajanin (RAG) karsisina konur: RAG kaliteden ne kaybettiriyor, maliyet ve surede ne kazandiriyor.

    python -m backend.knowledge.agent --eval --long-context [--judge] [--limit 3] [--golden-set ...]

Belgeler Qdrant'taki parcalardan birlestirilir (aramanin gordugu metnin aynisi, notlar haric).
Belge metni Gemini'de bir saatlik onbellege (cached content) konur; her soru sadece soruyu ve
cevabi oder, onbellekten okunan token ~%10 fiyatla sayilir. Onbellek kurulamazsa her soruda tum
metin gider (pahali, ama olcum yine calisir).

Yargica giden kanit, cevapta kaynak gosterilen belgelerin metnidir (tum korpus degil)."""

import time

from backend.knowledge.agent import AgentResult, DEFAULT_MODEL, SYSTEM_PROMPT
from backend.shared import tracing

CALL_SITE = "knowledge_longctx"
CACHE_TTL = "3600s"
MAX_CORPUS_TOKENS = 900_000  # Flash'in 1M baglaminda soru ve cevaba yer kalsin
# Yargica tam belgeler gider; ajanin 24k sinirinda cok kaynakli cevabin son belgesi kesilip dogru iddialar
# "desteksiz" sayiliyordu (smoke testte k01: KARARLAR_OZET.md GEREKSINIMLER.md'den sonra kesildi).
JUDGE_EVIDENCE_CHARS = 100_000

# Arama araclari yok; cevap kurallari ajaninkiyle ayni ki puanlama ayni sekilde calissin.
INSTRUCTION = ("Sen bir proje belgeleri asistanisin. Projenin TUM belgeleri asagida <belge> etiketleri icinde. "
               "Sadece bu belgelere dayanarak Turkce cevap ver.\n\n"
               + SYSTEM_PROMPT[SYSTEM_PROMPT.index("Cevap kurallari:"):]
               .replace("Tum sonuclar UYARI ya da 'zayif eslesme' ise bu da bulunamadi demektir.", ""))


def build_corpus(docs):
    return "\n\n".join(f'<belge yol="{path}">\n{text}\n</belge>' for path, text in sorted(docs.items()))


def cited_evidence(docs, citations):
    """Yargic icin: cevapta gosterilen belgelerin metni (yol eslesmesi buyuk/kucuk harf ve / \\ duyarsiz)."""
    by_key = {p.replace("\\", "/").lower(): p for p in docs}
    return [f"[{by_key[k]}]\n{docs[by_key[k]]}" for k in
            dict.fromkeys(c.replace("\\", "/").lower() for c in citations) if k in by_key]


class LongContextReader:
    """ResearchAgent ile ayni arayuz (ask -> AgentResult), boylece agent._evaluate ikisini de olcer."""

    judge_evidence_chars = JUDGE_EVIDENCE_CHARS

    def __init__(self, docs, model=DEFAULT_MODEL, client=None):
        self.docs, self.model, self._client = docs, model, client
        self.corpus = build_corpus(docs)
        self._cache = None

    def _gemini(self):
        if self._client is None:
            import os

            from dotenv import load_dotenv
            from google import genai

            from backend.shared.paths import PROJECT_ROOT

            load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
            self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        return self._client

    def corpus_tokens(self):
        return self._gemini().models.count_tokens(model=self.model, contents=INSTRUCTION + self.corpus).total_tokens

    def _cached_content(self):
        """Belge metnini bir kez onbellege koyar; olmazsa None (her soruda tum metin gider)."""
        if self._cache is None:
            from google.genai import types

            try:
                self._cache = self._gemini().caches.create(model=self.model, config=types.CreateCachedContentConfig(
                    system_instruction=INSTRUCTION, contents=[self.corpus], ttl=CACHE_TTL)).name
            except Exception as exc:
                print(f"Onbellek kurulamadi, her soruda tum metin gidecek: {type(exc).__name__}: {exc}")
                self._cache = ""
        return self._cache or None

    def close(self):
        if self._cache:
            try:
                self._gemini().caches.delete(name=self._cache)
            except Exception:
                pass  # TTL dolunca kendisi silinir

    def ask(self, question):
        from google.genai import types

        from backend.shared import gemini_retry, gemini_usage

        started = time.monotonic()
        cache = self._cached_content()
        if cache:
            config = types.GenerateContentConfig(temperature=0.0, cached_content=cache)
            contents = question
        else:
            config = types.GenerateContentConfig(temperature=0.0, system_instruction=INSTRUCTION)
            contents = f"{self.corpus}\n\nSoru: {question}"

        def _call():
            response = self._gemini().models.generate_content(model=self.model, contents=contents, config=config)
            gemini_usage.log_usage(CALL_SITE, self.model, response)
            return response

        with tracing.span("longctx.ask", tracing.LLM, **{"llm.model_name": self.model,
                                                         "input.value": tracing.clip(question),
                                                         "rag.cached": bool(cache)}) as span:
            response = gemini_retry.call_with_retry(_call, pacing_key=CALL_SITE)
            usage = response.usage_metadata
            result = AgentResult(answer=(response.text or "").strip(),
                                 prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                                 output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                                 cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0)
            result.evidence = cited_evidence(self.docs, result.citations)
            result.seconds = time.monotonic() - started
            span.update({"output.value": tracing.clip(result.answer), "rag.citations": result.citations,
                         "rag.says_not_found": result.says_not_found,
                         "llm.token_count.prompt": result.prompt_tokens,
                         "llm.token_count.completion": result.output_tokens,
                         "rag.cached_tokens": result.cached_tokens})
        return result
