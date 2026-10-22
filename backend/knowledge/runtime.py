"""Modellerin ve Qdrant istemcisinin tembel (ilk kullanimda) yuklenmesi - MCP sunucusu
acilirken birkac saniyelik model yuklemesini ilk aramaya erteler.

Model secimi factprobe altin setiyle yapildi (2026-10-22, artifacts/knowledge_eval/ab_models.json):
MiniLM+mMiniLM Hit@1 0.76 / MRR 0.85, BGE-M3+bge-reranker-v2-m3 Hit@1 0.81 / MRR 0.87.
Asil kazanc alakasiz soruyu ayirmada: bge-reranker'da alakasiz sorularin en iyi parcasi
<= 0.02, cevapli sorularinki >= 0.08 (search.RELEVANCE_THRESHOLD)."""

import os
import threading

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # 1024 boyut, 8192 token; 512'de kesiyoruz (parcalar kisa)
EMBEDDING_MAX_TOKENS = 512
# Parca sinirlarini belirleyen anlamsal chunker eski modelle kaliyor: parcalar ve
# altin set olcumleri model degisiminden bagimsiz kalsin.
CHUNKER_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
CROSS_ENCODER_MAX_TOKENS = 512


def _on_gpu():
    import torch

    return torch.cuda.is_available()


def load_embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    model.max_seq_length = EMBEDDING_MAX_TOKENS
    if _on_gpu():
        model.half()
    return model


def load_chunker_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(CHUNKER_MODEL_NAME)


def load_reranker():
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, max_length=CROSS_ENCODER_MAX_TOKENS)
    if _on_gpu():
        model.model.half()
    return model


class Runtime:
    def __init__(self):
        self._lock = threading.Lock()
        self._client = self._embedder = self._reranker = self._chunker_model = self._contextualizer = None

    def client(self):
        with self._lock:
            if self._client is None:
                from dotenv import load_dotenv
                from qdrant_client import QdrantClient

                from backend.shared.paths import PROJECT_ROOT

                load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
                self._client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
            return self._client

    def embedder(self):
        with self._lock:
            if self._embedder is None:
                self._embedder = load_embedder()
            return self._embedder

    def chunker_model(self):
        with self._lock:
            if self._chunker_model is None:
                self._chunker_model = load_chunker_model()
            return self._chunker_model

    def reranker(self):
        with self._lock:
            if self._reranker is None:
                self._reranker = load_reranker()
            return self._reranker

    def contextualizer(self):
        """Gemini baglam ureticisi; sadece "contextual": true olan projelerde cagrilir."""
        with self._lock:
            if self._contextualizer is None:
                from backend.knowledge.contextual import GeminiContextualizer

                self._contextualizer = GeminiContextualizer()
            return self._contextualizer
