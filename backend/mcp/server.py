"""Model Context Protocol (MCP) sunucusu - stdio uzerinden, SDK kullanmadan.

Protokol: her satir bir JSON-RPC 2.0 mesaji. Istemci (Claude Code / Desktop)
'initialize' -> 'notifications/initialized' -> 'tools/list' -> 'tools/call' sirasiyla
konusur. stdout SADECE protokol mesajlari icindir; model yukleme loglari gibi her
sey stderr'e yonlendirilir, yoksa istemci bozuk JSON gorur.

Calistirma: python -m backend.mcp.server"""

import json
import sys
import time
import traceback

from backend.shared import tracing

SERVER_INFO = {"name": "semantic-retrieval-engine", "version": "0.1.0"}
SUPPORTED_PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]

INSTRUCTIONS = (
    "Proje belgeleri (spec, plan, rapor, karar notlari) icin hafiza katmani. "
    "Sohbetin ilk sorusu projeyle ilgiliyse once 'brief' cagirin: son notlar ve degisen belgelerle "
    "nerede kalindigini gosterir. Projeyi tanimiyorsaniz 'overview'. Uzun belgeleri okumadan once 'search' ile ilgili "
    "parcalari bulun; sonuclar kaynak, satir araligi ve degisiklik tarihiyle gelir, cevapta "
    "bunlari kaynak olarak gosterin ve ayni konuda celisen sonuclarda yeni tarihliyi esas alin. "
    "Tum belge gerekirse 'get_document'. Bir is ya da karar bittiginde 'checkpoint' ile oturumu "
    "(ozet, kararlar, acik isler) kaydedin ki sonraki konusmalar hatirlasin; tek bir kayit icin 'add_note', "
    "kapanan is icin 'close_task'."
)

_PROJECT = {"type": "string", "description": "Proje adi (orn. nehaber). Tek proje varsa bos birakilabilir."}

TOOLS = [
    {
        "name": "brief",
        "description": ("Yeni sohbette ilk cagrilacak 'nerede kalmistik' ozeti: projeler, son notlar (kararlar, "
                        "ilerleme) ve son degisen belgeler, tarihleriyle. Kisa (~1.500 karakter); ayrinti icin search."),
        "inputSchema": {"type": "object", "properties": {"project": _PROJECT}},
    },
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
                "kind": {"type": "string", "enum": ["note", "decision", "task", "fact", "preference", "focus", "session"],
                         "description": "Sadece bu turdeki kayitlar (orn. 'decision' kararlar, 'task' isler)"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 15, "default": 5},
            },
        },
    },
    {
        "name": "ask",
        "description": ("Soruyu ayri bir arastirma ajani (Gemini) belgelerde arayip okuyarak cevaplar ve kaynak gosterir; "
                        "belgelerde yoksa 'Belgelerde bulamadim' der. Birden cok belgeyi birlestiren ya da uzun okuma "
                        "gerektiren sorularda, baglamini parcalarla doldurmak istemediginde kullan. ~10 sn surer ve "
                        "Gemini ucreti vardir; tek bir olguyu bulmak icin search daha hizli. Cevaptaki kaynaklari "
                        "kullaniciya aynen aktar."),
        "inputSchema": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "Dogal dilde soru"}, "project": _PROJECT},
            "required": ["question"],
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
                        "Tur icin kind ver (karar, is, olgu...); her kayit kisa bir kimlik alir (orn. K-0042). "
                        "Yeni not eski bir kaydin yerine geciyorsa supersedes ile eskisinin kimligini ya da yolunu "
                        "ver; eski kayit aramada geri plana duser."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string", "description": "Markdown icerik"},
                "project": _PROJECT,
                "kind": {"type": "string", "enum": ["note", "decision", "task", "fact", "preference", "focus"],
                         "default": "note", "description": "Kaydin turu; 'task' acik is olarak brifingde gorunur"},
                "supersedes": {"type": "string", "description": "Yerine gecilen kaydin kimligi (orn. K-0042) ya da yolu"},
            },
            "required": ["title", "body"],
        },
    },
    {
        "name": "checkpoint",
        "description": ("Oturumu tek cagrida kaydeder: ozet bir oturum kaydi, her karar ve her acik is ayri kayit "
                        "olur. Kullanici 'kaydet' dediginde ya da buyuk bir is bittiginde cagir; sonraki sohbet "
                        "brifingde bunlari gorur."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Oturumda ne yapildi (Markdown)"},
                "decisions": {"type": "array", "items": {"type": "string"}, "description": "Alinan kararlar, her biri bir cumle"},
                "open_tasks": {"type": "array", "items": {"type": "string"}, "description": "Devam eden ya da bekleyen isler"},
                "project": _PROJECT,
            },
            "required": ["summary"],
        },
    },
    {
        "name": "close_task",
        "description": "Acik bir isi kapatir; brifingdeki 'Acik isler' listesinden duser.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Isin kisa kimligi (orn. I-0007) ya da not yolu"},
                "note": {"type": "string", "description": "Nasil kapandi (istege bagli)"},
                "project": _PROJECT,
            },
            "required": ["task"],
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
        "name": "propose_project",
        "description": ("Yeni bir repo/klasoru proje olarak onerir (orn. kullanici yeni bir repoya gecti). Klasor, izin "
                        "verilen ana klasore gore goreli verilir. Proje hemen eklenmez: kullanici bilgisayarda onaylar, "
                        "sonra project='<ad>' ile aranabilir. Yalnizca kullanici istediginde kullanin."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Proje adi: kucuk harf, rakam, '-' (orn. 'yeni-repo')"},
                "folder": {"type": "string", "description": "Ana klasore gore goreli klasor (orn. 'yeni-repo' ya da 'isler/yeni-repo')"},
                "description": {"type": "string", "description": "Projenin kisa aciklamasi"},
            },
            "required": ["name", "folder"],
        },
    },
    {
        "name": "list_projects",
        "description": "Motora bagli projeleri listeler.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

# claude.ai'de "+" menusunden sohbete eklenen hazir mesajlar (MCP prompts).
PROMPTS = [
    {
        "name": "oturumu-kaydet",
        "title": "Oturumu kaydet",
        "description": "Bu sohbette yapilanlari, kararlari ve acik isleri checkpoint ile kaydetmeyi ister.",
        "arguments": [{"name": "project", "description": "Proje adi (bos: varsayilan)", "required": False}],
    },
    {
        "name": "devam-et",
        "title": "Devam et",
        "description": "Brifingi sohbete ekler ve kalinan yerden devam etmeyi ister.",
        "arguments": [{"name": "project", "description": "Proje adi (bos: varsayilan)", "required": False}],
    },
]
SAVE_TEXT = ("Bu sohbette yapilanlari kaydet: checkpoint cagir; summary'ye ne yapildigini, decisions'a "
             "verilen kararlari, open_tasks'a devam eden isleri yaz. Kapanan bir is varsa close_task ile kapat. "
             "Uydurma, yalnizca bu sohbette gecenleri yaz.")
CONTINUE_TEXT = ("Asagidaki brifing bu bilgi tabaninda nerede kaldigimizi gosteriyor. Ona gore kaldigimiz yerden "
                 "devam edelim; eksik bir ayrinti gerekirse search ile bul, uydurma.\n\n")

_ALLOWED_ARGS = {t["name"]: set(t["inputSchema"].get("properties", {})) for t in TOOLS}
WRITE_TOOLS = {"add_note", "add_document", "sync_project", "checkpoint", "close_task"}


class MCPServer:
    def __init__(self, service, instructions=None, workspace=None, audit=None):
        """instructions: workspace'e ozel talimat (config); genel talimatin sonuna eklenir.
        audit: AuditLog (audit.py); her arac ve prompt cagrisi icerik olmadan kaydedilir."""
        self.service = service
        self.workspace = workspace
        self.audit = audit
        header = f"Bu baglanti '{workspace}' workspace'ine aittir; yalnizca onun projelerini gorur. " if workspace else ""
        self.instructions = INSTRUCTIONS + (f"\n\n{header}{instructions or ''}".rstrip() if header or instructions else "")
        scope = getattr(service, "scope", None)
        # Salt okunur workspace'te yazma araclari listede bile gorunmez (model denemeye kalkmasin).
        hidden = WRITE_TOOLS if scope is not None and scope.read_only else set()
        if scope is not None and not scope.external_llm:
            hidden = hidden | {"ask"}  # dis LLM kapali: belge metni Gemini'ye gitmez
        if scope is None or not scope.project_base:
            hidden = hidden | {"propose_project"}  # yalnizca project_base tanimli workspace'lerde
        self.tools = [t for t in TOOLS if t["name"] not in hidden]
        self._allowed_args = {name: args for name, args in _ALLOWED_ARGS.items() if name not in hidden}

    def handle(self, message, client=None):
        """Tek bir JSON-RPC mesajini isler. Bildirimler (id'siz) icin None doner.
        client: istemci etiketi (orn. 'oauth:1a2b3c4d'); izlere ve denetim gunlugune yazilir."""
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
            return _result(msg_id, {"tools": self.tools})
        if method == "tools/call":
            return _result(msg_id, self._call_tool(params, client))
        if method == "prompts/list":
            return _result(msg_id, {"prompts": PROMPTS})
        if method == "prompts/get":
            return self._get_prompt(msg_id, params, client)
        return _error(msg_id, -32601, f"Method not found: {method}")

    def _initialize(self, params):
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}, "prompts": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": self.instructions,
        }

    def _labels(self, client):
        return {"rag.workspace": self.workspace or "*", "rag.client": client}

    def _record(self, kind, name, project, client, started, ok, chars=0, error_kind=None):
        if self.audit is not None:
            self.audit.write(workspace=self.workspace or "*", client=client, kind=kind, name=name,
                             project=project[:64] if isinstance(project, str) else None, ok=ok, chars=chars,
                             ms=round((time.monotonic() - started) * 1000), error=error_kind)

    def _get_prompt(self, msg_id, params, client=None):
        name = params.get("name")
        if name not in {p["name"] for p in PROMPTS}:
            return _error(msg_id, -32602, f"Bilinmeyen prompt: {name}")
        project = (params.get("arguments") or {}).get("project") or None
        started = time.monotonic()
        if name == "oturumu-kaydet":
            self._record("prompt", name, project, client, started, ok=True, chars=len(SAVE_TEXT))
            return _result(msg_id, {"description": "Oturumu kaydet",
                                    "messages": [{"role": "user", "content": {"type": "text", "text": SAVE_TEXT}}]})
        with tracing.span("prompt.devam-et", tracing.TOOL, **{"input.value": project or ""},
                          **self._labels(client)) as span:
            try:
                text = CONTINUE_TEXT + self.service.brief(project=project)
            except (ValueError, KeyError) as exc:
                span.set("rag.error_kind", "input").error(exc)
                self._record("prompt", name, project, client, started, ok=False, error_kind="input")
                return _error(msg_id, -32602, str(exc))
        self._record("prompt", name, project, client, started, ok=True, chars=len(text))
        return _result(msg_id, {"description": "Kalinan yerden devam",
                                "messages": [{"role": "user", "content": {"type": "text", "text": text}}]})

    def _call_tool(self, params, client=None):
        name = params.get("name")
        args = params.get("arguments") or {}
        started = time.monotonic()
        if name not in self._allowed_args:
            # gizli araclar (salt okunur workspace'te yazma, dis LLM kapaliyken ask) denemesi de kayda gecer
            self._record("tool", str(name)[:40], args.get("project") if isinstance(args, dict) else None, client,
                         started, ok=False, error_kind="unknown_tool")
            return _tool_text(f"Bilinmeyen arac: {name}", is_error=True)
        unknown = set(args) - self._allowed_args[name]
        if unknown:
            self._record("tool", name, args.get("project"), client, started, ok=False, error_kind="unknown_args")
            return _tool_text(f"Bilinmeyen parametre(ler): {', '.join(sorted(unknown))}", is_error=True)
        params = json.dumps(args, ensure_ascii=False)
        error_kind = None
        with tracing.span(f"tool.{name}", tracing.TOOL, **{"tool.name": name, "tool.parameters": params,
                                                           "input.value": tracing.clip(params)},
                          **self._labels(client)) as span:
            try:
                text, is_error = getattr(self.service, name)(**args), False
            except (ValueError, KeyError, TypeError) as exc:
                text, is_error, error_kind = f"Hata: {exc}", True, "input"
                span.set("rag.error_kind", "input").error(exc)
            except Exception as exc:  # Qdrant kapali vb. - istemciye anlamli mesaj, ayrinti stderr'e
                traceback.print_exc(file=sys.stderr)
                text, is_error = f"Beklenmeyen hata ({type(exc).__name__}): {exc}. Qdrant ayakta mi? (docker compose up -d)", True
                error_kind = "internal"
                span.set("rag.error_kind", "internal").error(exc)
            span.set("output.value", tracing.clip(text)).set("rag.output_chars", len(text))
        self._record("tool", name, args.get("project"), client, started, ok=not is_error, chars=len(text),
                     error_kind=error_kind)
        return _tool_text(text, is_error=is_error)


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
    tracing.init_tracing("rag-mcp")
    serve(MCPServer(KnowledgeService(Runtime())), protocol_in, protocol_out)


if __name__ == "__main__":
    main()
