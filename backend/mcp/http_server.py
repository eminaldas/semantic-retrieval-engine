"""MCP'nin 'Streamable HTTP' tasimasi - claude.ai custom connector'larinin baglanabilmesi icin.

Ayni MCPServer.handle() mantigi; sadece tasima stdio yerine HTTP:
- POST /mcp, govde tek bir JSON-RPC mesaji -> yanit application/json olarak doner.
  (Spec akis icin SSE'ye de izin veriyor; araclarimiz tek seferlik cevap dondugu icin
  duz JSON yeterli.)
- Bildirimler (id'siz mesajlar) -> 202 Accepted, govde yok.
- GET /mcp -> 405 (sunucudan istemciye akis acmiyoruz).
- Her istek 'Authorization: Bearer <MCP_HTTP_TOKEN>' tasimali; token .env'den okunur.
  Token tanimli degilse sunucu hic acilmaz - internete acik bir hafiza katmani
  korumasiz calismamali.

Sadece 127.0.0.1'e baglanir; disariya tunel (cloudflared/ngrok) acar.
Calistirma: python -m backend.mcp.http_server [--port 8765]"""

import hmac
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from backend.mcp.server import MCPServer, _error

MCP_PATH = "/mcp"
MAX_BODY_BYTES = 1_000_000


def make_handler(mcp_server, token):
    lock = threading.Lock()  # modeller ve Qdrant istemcisi tek istek tek seferde

    class Handler(BaseHTTPRequestHandler):
        server_version = "semantic-retrieval-engine"

        def log_message(self, fmt, *args):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _authorized(self):
            header = self.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return False
            return hmac.compare_digest(header[len("Bearer "):].strip().encode(), token.encode())

        def _send(self, status, body=None):
            payload = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            if body is not None:
                self.send_header("Content-Type", "application/json; charset=utf-8")
            if status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.split("?")[0] == "/health":
                return self._send(200, {"status": "ok"})
            if self.path.split("?")[0] != MCP_PATH:
                return self._send(404, {"error": "not found"})
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            self._send(405, {"error": "SSE akisi desteklenmiyor; POST kullanin"})

        def do_DELETE(self):
            self._send(405, {"error": "oturum yok"})

        def do_POST(self):
            if self.path.split("?")[0] != MCP_PATH:
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
            with lock:
                response = mcp_server.handle(message)
            if response is None:
                return self._send(202)
            self._send(200, response)

    return Handler


def build_server(mcp_server, token, host="127.0.0.1", port=8765):
    if not token or len(token) < 24:
        raise ValueError("MCP_HTTP_TOKEN en az 24 karakter olmali (.env). Ornek: python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
    return ThreadingHTTPServer((host, port), make_handler(mcp_server, token))


def main():
    import argparse
    import os

    from dotenv import load_dotenv

    from backend.knowledge.runtime import Runtime
    from backend.knowledge.service import KnowledgeService
    from backend.shared.paths import PROJECT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    try:
        httpd = build_server(MCPServer(KnowledgeService(Runtime())), os.environ.get("MCP_HTTP_TOKEN", ""), port=args.port)
    except ValueError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"MCP HTTP sunucusu: http://127.0.0.1:{args.port}{MCP_PATH}", file=sys.stderr, flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
