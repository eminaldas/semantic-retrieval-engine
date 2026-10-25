"""Contextual retrieval (Anthropic, 2024): her parcaya "bu parca belgenin neresi, neyi anlatiyor"
diye kisa bir cumle eklenir; dense ve BM25 vektorleri bu cumleyle birlikte uretilir. Kucuk
parcalar ("K4.24 [KARAR] kaldirildi.") tek basina hangi konuya ait oldugunu soylemiyor.

Maliyeti dusuk tutmak icin parca basina degil belge basina tek cagri yapilir: belge bir kez
gider, numarali parcalarin hepsinin cumlesi tek JSON'da doner. Uretilen cumleler (belge yolu +
parca metni) hash'iyle diskte saklanir; degismeyen parca yeniden gonderilmez.

Gemini yoksa ya da hata verirse bos doner: indeksleme bagimsiz calismaya devam eder."""

import hashlib
import json
import os
import sys
import threading

from pydantic import BaseModel

from backend.knowledge import projects

DEFAULT_MODEL = "gemini-flash-lite-latest"
CALL_SITE = "knowledge_contextual"
MAX_CHUNKS_PER_CALL = 120  # cikti JSON'u buyudukce model parca atlamaya basliyor
MAX_DOC_CHARS = 400_000  # Flash-Lite baglami genis; bunun ustu kesilir (bugun en buyuk belge ~110k)

PROMPT = """Asagida bir proje belgesi ve o belgeden alinmis numarali parcalar var.
Her parca icin, aramada bulunabilmesi icin parcanin belgede nerede durdugunu ve neyi anlattigini
soyleyen TEK kisa Turkce cumle yaz (en fazla 25 kelime).
- Belgeye ozgu adlari kullan: karar numaralari (K4.24 gibi), faz, modul, dosya, kisi, tarih.
- Parcanin metnini tekrar etme; parcada olmayan ama belgeden anlasilan baglami ekle.
- Bilgi uydurma; belgede yoksa yazma.
Her parca icin bir kayit dondur: i = parca numarasi, c = cumle.

<belge yol="{path}">
{document}
</belge>

<parcalar>
{chunks}
</parcalar>"""


class ChunkContext(BaseModel):
    i: int
    c: str


class ContextBatch(BaseModel):
    contexts: list[ChunkContext]


def cache_key(relpath, chunk_text):
    return hashlib.sha256(f"{relpath}\n{chunk_text}".encode("utf-8")).hexdigest()[:24]


def cache_path(project, manifest_dir=projects.MANIFEST_DIR):
    return os.path.join(manifest_dir, f"{project}_contexts.json")


def build_prompt(relpath, document, chunk_texts, offset=0):
    numbered = "\n\n".join(f"[{offset + i + 1}]\n{text.strip()}" for i, text in enumerate(chunk_texts))
    return PROMPT.format(path=relpath, document=document[:MAX_DOC_CHARS], chunks=numbered)


def parse_batch(batch, count, offset=0):
    """Model ciktisini parca sirasina dizer; eksik ya da fazladan numaralar bos kalir."""
    result = [""] * count
    for item in batch.contexts:
        index = item.i - 1 - offset
        if 0 <= index < count:
            result[index] = " ".join(item.c.split())
    return result


class GeminiContextualizer:
    """describe(project, relpath, document, chunk_texts) -> parca basina cumle listesi."""

    def __init__(self, model=DEFAULT_MODEL, manifest_dir=projects.MANIFEST_DIR, client=None,
                 log=lambda msg: print(msg, file=sys.stderr, flush=True)):
        self.model = model
        self.manifest_dir = manifest_dir
        self._client = client
        self.log = log
        self._lock = threading.Lock()
        self._caches = {}

    def _gemini(self):
        if self._client is None:
            from dotenv import load_dotenv
            from google import genai

            from backend.shared.paths import PROJECT_ROOT

            load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY tanimli degil")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _cache(self, project):
        if project not in self._caches:
            path = cache_path(project, self.manifest_dir)
            self._caches[project] = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        return self._caches[project]

    def _save(self, project):
        os.makedirs(self.manifest_dir, exist_ok=True)
        with open(cache_path(project, self.manifest_dir), "w", encoding="utf-8") as f:
            json.dump(self._caches[project], f, ensure_ascii=False, indent=0, sort_keys=True)

    def _call(self, prompt):
        from google.genai import types

        from backend.shared import gemini_retry, gemini_usage

        def _do():
            response = self._gemini().models.generate_content(
                model=self.model, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=ContextBatch,
                                                   temperature=0.0),
            )
            gemini_usage.log_usage(CALL_SITE, self.model, response)
            return response.parsed or ContextBatch.model_validate_json(response.text)

        return gemini_retry.call_with_retry(_do, pacing_key=CALL_SITE)

    def describe(self, project, relpath, document, chunk_texts):
        with self._lock:
            cache = self._cache(project)
            keys = [cache_key(relpath, t) for t in chunk_texts]
            missing = [i for i, k in enumerate(keys) if k not in cache]
            for start in range(0, len(missing), MAX_CHUNKS_PER_CALL):
                group = missing[start:start + MAX_CHUNKS_PER_CALL]
                texts = [chunk_texts[i] for i in group]
                try:
                    batch = self._call(build_prompt(relpath, document, texts))
                except Exception as exc:  # baglamsiz indekslemeye dus, sync'i durdurma
                    self.log(f"baglam uretilemedi ({relpath}): {exc}")
                    break
                for i, sentence in zip(group, parse_batch(batch, len(texts))):
                    if sentence:
                        cache[keys[i]] = sentence
            if missing:
                self._save(project)
            return [cache.get(k, "") for k in keys]
