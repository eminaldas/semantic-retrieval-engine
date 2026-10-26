import json
import threading
import urllib.error
import urllib.request

import pytest

from backend.mcp import http_server
from backend.mcp.server import MCPServer

TOKEN = "t" * 32


class FakeService:
    def search(self, query, project=None, doc_type=None, top_k=6):
        return f"sonuc: {query}"


@pytest.fixture
def base_url():
    httpd = http_server.build_server(MCPServer(FakeService()), TOKEN, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(url, body, token=TOKEN):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        return exc.code, None


def test_oauth_password_is_separate_from_api_token():
    token = "t" * 32
    assert http_server.oauth_password({"MCP_OAUTH_PASSWORD": "o" * 32}, token) == ("o" * 32, None)
    password, warning = http_server.oauth_password({}, token)
    assert password == token and warning
    with pytest.raises(ValueError):
        http_server.oauth_password({"MCP_OAUTH_PASSWORD": token}, token)


def test_rejects_short_token():
    with pytest.raises(ValueError):
        http_server.build_server(MCPServer(FakeService()), "kisa", port=0)


def test_missing_or_wrong_token_is_401(base_url):
    assert _post(base_url + "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "ping"}, token=None)[0] == 401
    assert _post(base_url + "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "ping"}, token="y" * 32)[0] == 401


def test_only_a_rejected_token_gets_invalid_token_challenge(base_url):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
    challenges = []
    for headers in ({"Authorization": "Bearer " + "y" * 32}, {}):
        req = urllib.request.Request(base_url + "/mcp", data=body, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(req)
        challenges.append(info.value.headers["WWW-Authenticate"])
    # suresi dolmus token istemciyi refresh'e yonlendirir; token hic yoksa sadece Bearer
    assert challenges == ['Bearer error="invalid_token"', "Bearer"]


def test_api_key_header_is_accepted(base_url):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
    for key, expected in ((TOKEN, 200), ("y" * 32, 401)):
        req = urllib.request.Request(base_url + "/mcp", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "X-API-Key": key})
        try:
            with urllib.request.urlopen(req) as resp:
                assert resp.status == expected
        except urllib.error.HTTPError as exc:
            assert exc.code == expected


def test_tool_call_round_trip(base_url):
    status, body = _post(base_url + "/mcp", {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                             "params": {"name": "search", "arguments": {"query": "odul"}}})
    assert status == 200
    assert body["id"] == 7 and body["result"]["content"][0]["text"] == "sonuc: odul"


def test_notification_is_202_without_body(base_url):
    assert _post(base_url + "/mcp", {"jsonrpc": "2.0", "method": "notifications/initialized"}) == (202, None)


def test_unknown_path_is_404(base_url):
    assert _post(base_url + "/baska", {"jsonrpc": "2.0", "id": 1, "method": "ping"})[0] == 404


def test_health_needs_no_token(base_url):
    with urllib.request.urlopen(base_url + "/health") as resp:
        assert json.loads(resp.read()) == {"status": "ok"}
