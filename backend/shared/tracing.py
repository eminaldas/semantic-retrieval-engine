"""Arama, MCP araclari ve ajan icin OpenTelemetry izleri (OpenInference semasi).

Izler yerel Arize Phoenix'e (docker compose, port 6006) OTLP/HTTP ile gider; Phoenix
retriever/reranker/LLM span turlerini bu semadan tanir. init_tracing cagrilmazsa, paket
kurulu degilse ya da RAG_TRACING=0 ise her span no-op'tur: arama yolu hicbir sey odemez.
Phoenix kapaliysa exporter arka planda hata loglar, istek akisi bozulmaz.

Ortam degiskenleri (.env):
  RAG_TRACING=0            izlemeyi kapatir
  RAG_TRACE_ENDPOINT       varsayilan http://localhost:6006/v1/traces
  PHOENIX_API_KEY          Phoenix'te kimlik dogrulama aciksa (system key) zorunlu
  RAG_TRACE_CONTENT=0      parca metinlerini ve sorgulari span'e yazmaz (sadece id/skor)"""

import json
import os
from contextlib import contextmanager

try:
    from opentelemetry import trace as _otel
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # paket yoksa izleme sessizce kapali
    _otel = None

DEFAULT_ENDPOINT = "http://localhost:6006/v1/traces"
MAX_TEXT_CHARS = 1000  # span basina metin siniri; Phoenix'i belge deposuna cevirmeyelim
TRACER_NAME = "semantic-retrieval-engine"

# OpenInference span turleri
CHAIN, RETRIEVER, RERANKER, EMBEDDING, LLM, TOOL, AGENT = (
    "CHAIN", "RETRIEVER", "RERANKER", "EMBEDDING", "LLM", "TOOL", "AGENT")

_state = {"enabled": False, "content": True, "tracer": None, "provider": None}


def init_tracing(project_name, endpoint=None):
    """Izlemeyi acar; Phoenix'te span'ler project_name projesinde toplanir. Acildiysa True."""
    if _otel is None or os.environ.get("RAG_TRACING", "1") == "0":
        return False
    if _state["enabled"]:
        return True
    _load_env()
    if os.environ.get("RAG_TRACING", "1") == "0":
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False
    headers = {}
    if os.environ.get("PHOENIX_API_KEY"):
        headers["authorization"] = f"Bearer {os.environ['PHOENIX_API_KEY']}"
    exporter = OTLPSpanExporter(endpoint=endpoint or os.environ.get("RAG_TRACE_ENDPOINT", DEFAULT_ENDPOINT),
                                headers=headers)
    provider = TracerProvider(resource=Resource.create({
        "service.name": project_name, "openinference.project.name": project_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    install(provider, content=os.environ.get("RAG_TRACE_CONTENT", "1") != "0")
    return True


def install(provider, content=True):
    """Span'leri verilen TracerProvider'a yonlendirir (init_tracing ve testler kullanir)."""
    _state.update(enabled=True, content=content, tracer=provider.get_tracer(TRACER_NAME), provider=provider)


def uninstall():
    _state.update(enabled=False, content=True, tracer=None, provider=None)


def _load_env():
    try:
        from dotenv import load_dotenv

        from backend.shared.paths import PROJECT_ROOT
    except ImportError:
        return
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def enabled():
    return _state["enabled"]


def clip(text):
    """Span'e yazilacak metin; RAG_TRACE_CONTENT=0 ise None (yazilmaz)."""
    if not _state["content"] or text is None:
        return None
    text = str(text)
    return text if len(text) <= MAX_TEXT_CHARS else text[:MAX_TEXT_CHARS] + "..."


def _primitive_list(values):
    """OTel sadece tek tipli str/int/float/bool listelerini dogrudan saklar; gerisi JSON olur."""
    return bool(values) and len({type(v) for v in values}) == 1 and isinstance(values[0], (str, int, float, bool))


class Span:
    """OTel span'ini saran kucuk arayuz; None degerleri atlar, dict/list'i JSON'a cevirir."""

    def __init__(self, span=None):
        self._span = span

    def set(self, key, value):
        if self._span is None or value is None:
            return self
        if isinstance(value, dict) or (isinstance(value, (list, tuple)) and not _primitive_list(value)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        self._span.set_attribute(key, value)
        return self

    def update(self, attributes):
        for key, value in attributes.items():
            self.set(key, value)
        return self

    def error(self, exc):
        if self._span is not None:
            self._span.record_exception(exc)
            self._span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
        return self


@contextmanager
def span(name, kind=CHAIN, **attributes):
    """with span("search", RETRIEVER, **{"input.value": q}) as s: ... s.set(...)
    Izleme kapaliysa bos bir Span verir. Govdedeki istisna span'e yazilip yeniden atilir."""
    if not _state["enabled"]:
        yield Span()
        return
    with _state["tracer"].start_as_current_span(name, record_exception=False, set_status_on_exception=False) as raw:
        s = Span(raw).set("openinference.span.kind", kind).update(attributes)
        try:
            yield s
        except Exception as exc:
            s.error(exc)
            raise


def document_attributes(prefix, docs, score_key, extra=None):
    """OpenInference belge listesi: <prefix>.<i>.document.{id,score,content,metadata}.
    extra: chunk_id -> metadata'ya eklenecek alanlar (orn. dense/bm25 sirasi)."""
    extra = extra or {}
    attrs = {}
    for i, doc in enumerate(docs):
        base = f"{prefix}.{i}.document"
        attrs[f"{base}.id"] = doc.get("chunk_id")
        if doc.get(score_key) is not None:
            attrs[f"{base}.score"] = float(doc[score_key])
        attrs[f"{base}.content"] = clip(doc.get("text"))
        attrs[f"{base}.metadata"] = {k: doc.get(k) for k in ("path", "location", "doc_type", "modified_at", "superseded_by")
                                     if doc.get(k) is not None} | extra.get(doc.get("chunk_id"), {})
    return attrs


def event(name, **attributes):
    """Acik span'e (orn. MCP arac span'i) olay ekler; izleme kapaliysa bir sey yapmaz."""
    if _state["enabled"]:
        _otel.get_current_span().add_event(name, {k: v for k, v in attributes.items() if v is not None})


def flush():
    """Kisa omurlu komutlar (eval, ajan CLI) cikmadan once bekleyen span'leri gonderir."""
    if _state["enabled"]:
        _state["provider"].force_flush()
