"""MCP'nin 'Streamable HTTP' tasimasi - claude.ai custom connector'larinin baglanabilmesi icin.

Ayni MCPServer.handle() mantigi; sadece tasima stdio yerine HTTP:
- POST /mcp, govde tek bir JSON-RPC mesaji -> yanit application/json olarak doner.
  (Spec akis icin SSE'ye de izin veriyor; araclarimiz tek seferlik cevap dondugu icin
  duz JSON yeterli.)
- Bildirimler (id'siz mesajlar) -> 202 Accepted, govde yok.
- GET /mcp -> 405 (sunucudan istemciye akis acmiyoruz).
- Her istek 'Authorization: Bearer <...>' tasimali. Iki tur token gecerli:
  .env'deki sabit MCP_HTTP_TOKEN (Claude Code / curl icin) ve OAuth akisiyla
  verilen kisa omurlu access token'lar (claude.ai icin, bkz. backend/mcp/oauth.py).
  Token tanimli degilse sunucu hic acilmaz - internete acik bir hafiza katmani
  korumasiz calismamali.

Sadece 127.0.0.1'e baglanir; disariya tunel (cloudflared) acar. Disaridan gorunen
adres (OAuth issuer) .env'deki MCP_PUBLIC_URL'den okunur. Varsayilan port, makinedeki
diger gelistirme sunuculariyla cakismasin diye nadir kullanilan bir porttur.
Calistirma: python -m backend.mcp.http_server [--port 47821]"""

import base64
import hmac
import html
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlencode, urlparse

from backend.mcp.oauth import OAuthError
from backend.mcp.server import MCPServer, _error

MCP_PATH = "/mcp"
DEFAULT_PORT = 47821
MAX_BODY_BYTES = 1_000_000
RESOURCE_METADATA_PATH = "/.well-known/oauth-protected-resource"
SERVER_METADATA_PATH = "/.well-known/oauth-authorization-server"

LOGIN_PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Retrieval Engine - Giris</title>
<style>body{{font-family:system-ui,sans-serif;max-width:420px;margin:12vh auto;padding:0 16px;color:#222}}
input[type=password]{{width:100%;padding:10px;font-size:16px;box-sizing:border-box}}
button{{margin-top:12px;padding:10px 18px;font-size:16px}}.err{{color:#b00020}}</style></head>
<body><h2>Semantic Retrieval Engine</h2>
<p><b>{client}</b> bilgi tabanina erisim istiyor. Onaylamak icin sahibin parolasini gir.</p>
{error}
<form method="post" action="/authorize">{hidden}
<input type="password" name="password" autofocus autocomplete="current-password" required>
<button type="submit">Izin ver</button></form></body></html>"""

AUTHORIZE_FIELDS = ("client_id", "redirect_uri", "response_type", "code_challenge",
                    "code_challenge_method", "state", "scope", "resource")


def _first_values(query):
    return {k: v[0] for k, v in parse_qs(query, keep_blank_values=True).items()}


def _basic_credentials(header):
    if not header.startswith("Basic "):
        return None
    try:
        decoded = base64.b64decode(header[len("Basic "):].strip()).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    client_id, sep, secret = decoded.partition(":")
    if not sep:
        return None
    return unquote(client_id), unquote(secret)


def make_handler(mcp_server, token, oauth=None, public_url=None):
    lock = threading.Lock()  # modeller ve Qdrant istemcisi tek istek tek seferde

    class Handler(BaseHTTPRequestHandler):
        server_version = "semantic-retrieval-engine"
        _auth_note = ""
        _invalid_token = False

        def parse_request(self):
            # handler keep-alive baglantida tekrar kullanilir: istek basina durum sifirlanir
            self._auth_note, self._invalid_token = "", False
            return super().parse_request()

        def log_message(self, fmt, *args):
            headers = getattr(self, "headers", None)
            agent = headers.get("User-Agent", "-") if headers else "-"
            note = self._auth_note
            sys.stderr.write("%s - %s - %s %s\n" % (self.address_string(), fmt % args, agent[:80], note))

        def _route(self):
            return urlparse(self.path).path

        def _issuer(self):
            if public_url:
                return public_url.rstrip("/")
            return "http://" + self.headers.get("Host", "127.0.0.1")

        def _authorized(self):
            # claude.ai'de "No sign-in" + ozel istek basligi yolu icin sabit token X-API-Key'de de kabul edilir
            # _auth_note hangi yolun kullanildigini loglar (token'in kendisi asla loglanmaz)
            api_key = self.headers.get("X-API-Key", "").strip()
            if api_key and hmac.compare_digest(api_key.encode(), token.encode()):
                self._auth_note = "auth=apikey"
                return True
            header = self.headers.get("Authorization", "")
            scheme, _, presented = header.partition(" ")
            presented = presented.strip()
            if scheme.lower() != "bearer" or not presented:
                self._auth_note = f"auth=yok sema={scheme[:10] or '-'} apikey={'yanlis' if api_key else 'yok'}"
                return False
            if hmac.compare_digest(presented.encode(), token.encode()):
                self._auth_note = "auth=sabit"
                return True
            if oauth is not None and oauth.validate_access_token(presented):
                self._auth_note = "auth=oauth"
                return True
            self._auth_note = f"auth=gecersiz uzunluk={len(presented)}"
            self._invalid_token = True
            return False

        def _send(self, status, body=None, headers=None, raw=None, content_type=None):
            if raw is not None:
                payload = raw
            else:
                payload = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
                if body is not None:
                    content_type = "application/json; charset=utf-8"
            self.send_response(status)
            if content_type:
                self.send_header("Content-Type", content_type)
            if status == 401 and self._route().startswith(MCP_PATH):
                params = []
                if self._invalid_token:  # RFC 6750: istemci refresh token'a gecsin
                    params.append('error="invalid_token"')
                if oauth is not None:
                    params.append(f'resource_metadata="{self._issuer()}{RESOURCE_METADATA_PATH}"')
                self.send_header("WWW-Authenticate", " ".join(["Bearer", ", ".join(params)]).strip())
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                return None
            return self.rfile.read(length)

        def _login_page(self, params, client_name, error="", status=200):
            hidden = "".join(
                f'<input type="hidden" name="{name}" value="{html.escape(params[name], quote=True)}">'
                for name in AUTHORIZE_FIELDS if name in params
            )
            page = LOGIN_PAGE.format(
                client=html.escape(client_name),
                error=f'<p class="err">{html.escape(error)}</p>' if error else "",
                hidden=hidden,
            )
            self._send(status, raw=page.encode("utf-8"), content_type="text/html; charset=utf-8",
                       headers={"X-Frame-Options": "DENY",
                                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'"})

        # --- GET ---------------------------------------------------------

        def do_GET(self):
            route = self._route()
            if route == "/health":
                return self._send(200, {"status": "ok"})
            if oauth is not None and route.startswith(RESOURCE_METADATA_PATH):
                return self._send(200, oauth.resource_metadata(self._issuer(), MCP_PATH))
            if oauth is not None and route.startswith(SERVER_METADATA_PATH):
                return self._send(200, oauth.server_metadata(self._issuer()))
            if oauth is not None and route == "/authorize":
                params = _first_values(urlparse(self.path).query)
                try:
                    client_name = oauth.check_authorize_request(params)
                except OAuthError as exc:
                    return self._send(exc.status, exc.body())
                return self._login_page(params, client_name)
            if route != MCP_PATH:
                return self._send(404, {"error": "not found"})
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            self._send(405, {"error": "SSE akisi desteklenmiyor; POST kullanin"})

        def do_DELETE(self):
            self._send(405, {"error": "oturum yok"})

        # --- POST --------------------------------------------------------

        def do_POST(self):
            route = self._route()
            if oauth is not None and route in ("/register", "/authorize", "/token"):
                return self._oauth_post(route)
            if route != MCP_PATH:
                return self._send(404, {"error": "not found"})
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._send(413 if length > MAX_BODY_BYTES else 400, _error(None, -32600, "Invalid Request"))
            try:
                message = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._send(400, _error(None, -32700, "Parse error"))
            if isinstance(message, list):
                return self._send(400, _error(None, -32600, "Batch istekleri desteklenmiyor"))
            if isinstance(message, dict):
                self._auth_note = f"{self._auth_note} rpc={str(message.get('method'))[:40]}".strip()
            with lock:
                response = mcp_server.handle(message)
            if response is None:
                return self._send(202)
            self._send(200, response)

        def _oauth_post(self, route):
            raw = self._read_body()
            if raw is None:
                return self._send(400, {"error": "invalid_request", "error_description": "govde bos ya da cok buyuk"})
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return self._send(400, {"error": "invalid_request", "error_description": "UTF-8 degil"})
            if route == "/register":
                try:
                    return self._send(201, oauth.register(json.loads(text)))
                except json.JSONDecodeError:
                    return self._send(400, {"error": "invalid_client_metadata", "error_description": "JSON degil"})
                except OAuthError as exc:
                    return self._send(exc.status, exc.body())
            form = _first_values(text)
            if route == "/token":
                basic = _basic_credentials(self.headers.get("Authorization", ""))
                if basic:  # client_secret_basic
                    form["client_id"], form["client_secret"] = basic
                self._auth_note = f"grant={form.get('grant_type', '-')[:20]}"
                try:
                    return self._send(200, oauth.exchange(form))
                except OAuthError as exc:
                    self._auth_note += f" hata={exc.error}"
                    return self._send(exc.status, exc.body())
            # /authorize: giris formu gonderildi
            password = form.pop("password", "")
            try:
                client_name = oauth.check_authorize_request(form)
            except OAuthError as exc:
                return self._send(exc.status, exc.body())
            try:
                code = oauth.approve(form, password)
            except OAuthError as exc:
                return self._login_page(form, client_name, error=exc.description, status=exc.status)
            query = {"code": code}
            if form.get("state"):
                query["state"] = form["state"]
            separator = "&" if "?" in form["redirect_uri"] else "?"
            self._send(302, headers={"Location": form["redirect_uri"] + separator + urlencode(query)})

    return Handler


def oauth_password(env, token):
    """OAuth giris sayfasinin parolasi. MCP_OAUTH_PASSWORD verilmisse o; verilmemisse eski davranis
    (parola = MCP_HTTP_TOKEN) uyariyla surer. Ayri olmali: giris sayfasina yazilan parola sizarsa
    tam yetkili API anahtari da sizmis olmasin. Doner: (parola, uyari ya da None)."""
    password = env.get("MCP_OAUTH_PASSWORD", "")
    if not password:
        return token, "UYARI: MCP_OAUTH_PASSWORD tanimli degil; OAuth giris parolasi olarak MCP_HTTP_TOKEN kullaniliyor."
    if hmac.compare_digest(password.encode(), token.encode()):
        raise ValueError("MCP_OAUTH_PASSWORD, MCP_HTTP_TOKEN ile ayni olmamali")
    return password, None


def build_server(mcp_server, token, host="127.0.0.1", port=DEFAULT_PORT, oauth=None, public_url=None):
    if not token or len(token) < 24:
        raise ValueError("MCP_HTTP_TOKEN en az 24 karakter olmali (.env). Ornek: python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
    return ThreadingHTTPServer((host, port), make_handler(mcp_server, token, oauth=oauth, public_url=public_url))


def main():
    import argparse
    import os

    from dotenv import load_dotenv

    from backend.knowledge.projects import load_config
    from backend.knowledge.runtime import Runtime
    from backend.knowledge.service import KnowledgeService
    from backend.knowledge.workspaces import load_workspaces
    from backend.mcp.oauth import OAuthProvider
    from backend.shared import tracing
    from backend.shared.paths import PROJECT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    if tracing.init_tracing("rag-mcp"):
        print("Izleme acik: Phoenix (rag-mcp projesi).", file=sys.stderr, flush=True)
    token = os.environ.get("MCP_HTTP_TOKEN", "")
    try:
        password, warning = oauth_password(os.environ, token)
        if warning:
            print(warning, file=sys.stderr, flush=True)
        oauth = OAuthProvider(password, store_path=os.path.join(PROJECT_ROOT, "artifacts", "oauth", "store.json"))
        workspaces = load_workspaces(load_config())  # hatali _workspaces sunucuyu actirmaz
        runtime = Runtime()
        service = KnowledgeService(runtime)
        httpd = build_server(MCPServer(service), token, port=args.port,
                             oauth=oauth, public_url=os.environ.get("MCP_PUBLIC_URL"))
    except ValueError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"MCP HTTP sunucusu: http://127.0.0.1:{args.port}{MCP_PATH}", file=sys.stderr, flush=True)
    if workspaces:  # adresleri Asama 3'te acilacak; simdilik yalnizca dogrulanir
        print(f"Workspace'ler (henuz adressiz): {', '.join(sorted(workspaces))}", file=sys.stderr, flush=True)
    threading.Thread(target=_warm_up, args=(runtime, service), daemon=True).start()
    httpd.serve_forever()


def _warm_up(runtime, service=None):
    """Modelleri acilista yukler (yoksa yeniden baslatma sonrasi ilk arama ~30 sn bekliyor),
    sonra klasor izleyicisini baslatir. RAG_WATCH=0 izleyiciyi kapatir."""
    import os

    from backend.knowledge.watcher import ProjectWatcher

    try:
        runtime.client()
        runtime.embedder().encode("isinma")
        runtime.reranker().predict([("isinma", "isinma")])
        print("Modeller hazir.", file=sys.stderr, flush=True)
    except Exception as exc:  # isinma basarisizsa ilk arama yine tembel yukler
        print(f"Isinma basarisiz: {exc}", file=sys.stderr, flush=True)
    if service is not None and os.environ.get("RAG_WATCH", "1") != "0":
        ProjectWatcher(service).start()
        print("Klasor izleyici calisiyor.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
