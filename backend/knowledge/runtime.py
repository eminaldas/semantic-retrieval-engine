"""Modellerin ve Qdrant istemcisinin tembel (ilk kullanimda) yuklenmesi - MCP sunucusu
acilirken birkac saniyelik model yuklemesini ilk aramaya erteler."""

import os
import threading

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # chunker/vector_writer ile ayni
CROSS_ENCODER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # query_engine ile ayni


class Runtime:
    def __init__(self):
        self._lock = threading.Lock()
        self._client = self._embedder = self._reranker = None

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
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            return self._embedder

    def reranker(self):
        with self._lock:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
            return self._reranker
