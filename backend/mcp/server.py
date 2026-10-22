"""Model Context Protocol (MCP) sunucusu - stdio uzerinden, SDK kullanmadan.

Protokol: her satir bir JSON-RPC 2.0 mesaji. Istemci (Claude Code / Desktop)
'initialize' -> 'notifications/initialized' -> 'tools/list' -> 'tools/call' sirasiyla
konusur. stdout SADECE protokol mesajlari icindir; model yukleme loglari gibi her
sey stderr'e yonlendirilir, yoksa istemci bozuk JSON gorur.

Calistirma: python -m backend.mcp.server"""

import json
import sys
import traceback

SERVER_INFO = {"name": "semantic-retrieval-engine", "version": "0.1.0"}
SUPPORTED_PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]

INSTRUCTIONS = (
    "Proje belgeleri (spec, plan, rapor, karar notlari) icin hafiza katmani. "
    "Projeyi tanimiyorsaniz once 'overview'. Uzun belgeleri okumadan once 'search' ile ilgili "
    "parcalari bulun; sonuclar kaynak, satir araligi ve degisiklik tarihiyle gelir, cevapta "
    "bunlari kaynak olarak gosterin ve ayni konuda celisen sonuclarda yeni tarihliyi esas alin. "
    "Tum belge gerekirse 'get_document'. Bir is ya da karar bittiginde 'add_note' ile ozetini "
    "kaydedin ki sonraki konusmalar hatirlasin."
)

_PROJECT = {"type": "string", "description": "Proje adi (orn. nehaber). Tek proje varsa bos birakilabilir."}

TOOLS = [
    {
        "name": "search",
        "description": (
            "Proje belgelerinde hibrit (anlamsal + anahtar kelime) arama yapar, rerank sonrasi en ilgili parcalari "
            "kaynak ve konumla dondurur. Proje hakkindaki cevabi yalnizca donen parcalara dayandir ve kullandigin "
            "belgenin yolunu belirt. 'zayif eslesme' isaretli parcalari kanit sayma; sonuc UYARI ile basliyorsa "
            "kullaniciya belgelerde bulunamadigini soyle, kendi genel bilgini proje bilgisi gibi sunma. "
            "'guncel degil' isaretli sonucta dosya indeksten sonra degismistir: bunu kullaniciya soyle ya da "
            "sync_project calistirip yeniden ara. 'eski not' isaretli notun yerini yenisi almistir."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Dogal dilde soru ya da anahtar kelimeler"},
                "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 4,
                            "description": "Birbirinden bagimsiz 2-4 soruyu tek cagrida aramak icin; sonuclar soru basina ayri basliklarda"},
                "project": _PROJECT,
                "doc_type": {"type": "string", "enum": ["doc", "note"], "description": "Sadece belgeler ya da sadece notlar"},
                "path_prefix": {"type": "string", "description": "Sadece bu klasor (orn. 'graf/') ya da dosya (orn. 'KARARLAR_OZET.md')"},
                "modified_after": {"type": "string", "description": "YYYY-AA-GG; bu tarihte ya da sonra degisen belgeler"},
                "modified_before": {"type": "string", "description": "YYYY-AA-GG; bu tarihte ya da once degisen belgeler"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 15, "default": 5},
            },
        },
    },
    {
        "name": "overview",
        "description": ("Projenin haritasi: klasorler, son degisen belgeler, son notlar. Projeyi ilk kez "
                        "soruyorsan ya da hangi belgenin nerede oldugunu bilmiyorsan search'ten once cagir."),
        "inputSchema": {"type": "object", "properties": {"project": _PROJECT}},
    },
    {
        "name": "list_documents",
        "description": "Indekslenmis belgeleri (yol, tur, parca sayisi, son degisiklik) listeler.",
        "inputSchema": {"type": "object", "properties": {"project": _PROJECT}},
    },
    {
        "name": "get_document",
        "description": "Bir belgenin tamamini (parca parca, konumlariyla) getirir. Once search denenmeli.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "list_documents'taki goreli yol"},
                "project": _PROJECT,
                "max_chars": {"type": "integer", "default": 20000},
            },
            "required": ["path"],
        },
    },
    {
        "name": "add_note",
        "description": ("Bir karar, ilerleme ya da konusma ozetini not olarak kaydeder ve hemen indeksler. "
                        "Yeni not eski bir notun yerine geciyorsa supersedes ile eskisini ver; eski not aramada geri plana duser."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string", "description": "Markdown icerik"},
                "project": _PROJECT,
                "supersedes": {"type": "string", "description": "Yerine gecilen notun yolu (orn. notes/2026-10-20-1200-esik.md)"},
            },
            "required": ["title", "body"],
        },
    },
    {
        "name": "add_document",
        "description": ("Tam bir belgeyi (plan, rapor, ozet) motorun belge klasorune yazar ve indeksler; proje reposuna "
                        "dokunmaz. Kisa karar/ilerleme kaydi icin add_note daha uygun."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Belge adi; dosya adi bundan uretilir (notes/belgeler/<ad>.md)"},
                "content": {"type": "string", "description": "Markdown icerik"},
                "project": _PROJECT,
                "overwrite": {"type": "boolean", "default": False, "description": "Ayni adli belge varsa guncelle"},
            },
            "required": ["name", "content"],
        },
    },
    {
        "name": "index_status",
        "description": ("Indeksin guncel olup olmadigini gosterir: diskte yeni, degismis ya da silinmis dosyalar, "
                        "son senkron ve parca sayisi. Sonuclarda 'guncel degil' goruyorsan bak."),
        "inputSchema": {"type": "object", "properties": {"project": _PROJECT}},
    },
    {
        "name": "sync_project",
        "description": "Proje klasorundeki yeni/degisen belgeleri indeksler, silinenleri kaldirir.",
        "inputSchema": {
            "type": "object",
            "properties": {"project": _PROJECT, "full": {"type": "boolean", "default": False}},
        },
    },
    {
        "name": "list_projects",
        "description": "Motora bagli projeleri listeler.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

_ALLOWED_ARGS = {t["name"]: set(t["inputSchema"].get("properties", {})) for t in TOOLS}


class MCPServer:
    def __init__(self, service):
        self.service = service

    def handle(self, message):
        """Tek bir JSON-RPC mesajini isler. Bildirimler (id'siz) icin None doner."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, -32600, "Invalid Request")
        method = message.get("method")
        msg_id = message.get("id")
        if msg_id is None:
            return None  # notifications/initialized, notifications/cancelled ...
        params = message.get("params") or {}
        if method == "initialize":
            return _result(msg_id, self._initialize(params))
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(msg_id, {"tools": TOOLS})
        if method == "tools/call":
            return _result(msg_id, self._call_tool(params))
        return _error(msg_id, -32601, f"Method not found: {method}")

    def _initialize(self, params):
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        }

    def _call_tool(self, params):
        name = params.get("name")
        args = params.get("arguments") or {}
        if name not in _ALLOWED_ARGS:
            return _tool_text(f"Bilinmeyen arac: {name}", is_error=True)
        unknown = set(args) - _ALLOWED_ARGS[name]
        if unknown:
            return _tool_text(f"Bilinmeyen parametre(ler): {', '.join(sorted(unknown))}", is_error=True)
        try:
            return _tool_text(getattr(self.service, name)(**args))
        except (ValueError, KeyError, TypeError) as exc:
            return _tool_text(f"Hata: {exc}", is_error=True)
        except Exception as exc:  # Qdrant kapali vb. - istemciye anlamli mesaj, ayrinti stderr'e
            traceback.print_exc(file=sys.stderr)
            return _tool_text(f"Beklenmeyen hata ({type(exc).__name__}): {exc}. Qdrant ayakta mi? (docker compose up -d)", is_error=True)


def _result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _tool_text(text, is_error=False):
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def serve(server, stdin, stdout):
    """stdin'den satir satir okur, her yaniti tek satir JSON olarak yazar."""
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = _error(None, -32700, "Parse error")
        else:
            response = server.handle(message)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()


def main():
    import io

    from backend.knowledge.runtime import Runtime
    from backend.knowledge.service import KnowledgeService

    protocol_out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    protocol_in = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    sys.stdout = sys.stderr  # kutuphanelerin print'leri protokolu bozmasin
    serve(MCPServer(KnowledgeService(Runtime())), protocol_in, protocol_out)


if __name__ == "__main__":
    main()
