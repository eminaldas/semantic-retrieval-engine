import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from backend.mcp import http_server
from backend.mcp.oauth import ACCESS_TTL, MAX_FAILED_LOGINS, OAuthError, OAuthProvider, pkce_challenge
from backend.mcp.server import MCPServer

PASSWORD = "p" * 32
REDIRECT = "https://claude.ai/api/mcp/auth_callback"
VERIFIER = "v" * 50


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def _authorize_params(client_id, state="xyz"):
    return {"client_id": client_id, "redirect_uri": REDIRECT, "response_type": "code",
            "code_challenge": pkce_challenge(VERIFIER), "code_challenge_method": "S256", "state": state}


def _token_form(client_id, code, verifier=VERIFIER):
    return {"grant_type": "authorization_code", "code": code, "client_id": client_id,
            "redirect_uri": REDIRECT, "code_verifier": verifier}


@pytest.fixture
def provider():
    return OAuthProvider(PASSWORD, clock=Clock())


def test_register_rejects_foreign_redirect(provider):
    with pytest.raises(OAuthError):
        provider.register({"redirect_uris": ["https://evil.example/cb"]})
    with pytest.raises(OAuthError):
        provider.register({"redirect_uris": ["http://claude.ai/cb"]})


def test_full_flow_issues_working_token(provider):
    client_id = provider.register({"redirect_uris": [REDIRECT], "client_name": "Claude"})["client_id"]
    code = provider.approve(_authorize_params(client_id), PASSWORD)
    tokens = provider.exchange(_token_form(client_id, code))
    assert tokens["token_type"] == "Bearer"
    assert provider.validate_access_token(tokens["access_token"])
    assert not provider.validate_access_token("baska-bir-token")


def test_code_is_single_use_and_needs_pkce(provider):
    client_id = provider.register({"redirect_uris": [REDIRECT]})["client_id"]
    code = provider.approve(_authorize_params(client_id), PASSWORD)
    with pytest.raises(OAuthError):
        provider.exchange(_token_form(client_id, code, verifier="w" * 50))
    with pytest.raises(OAuthError):  # yanlis denemede de kod harcandi
        provider.exchange(_token_form(client_id, code))


def test_wrong_password_and_lockout(provider):
    client_id = provider.register({"redirect_uris": [REDIRECT]})["client_id"]
    for _ in range(MAX_FAILED_LOGINS):
        with pytest.raises(OAuthError) as exc:
            provider.approve(_authorize_params(client_id), "yanlis")
        assert exc.value.status == 401
    with pytest.raises(OAuthError) as exc:
        provider.approve(_authorize_params(client_id), PASSWORD)
    assert exc.value.status == 429


def test_authorize_requires_registered_redirect_and_s256(provider):
    client_id = provider.register({"redirect_uris": [REDIRECT]})["client_id"]
    params = _authorize_params(client_id)
    with pytest.raises(OAuthError):
        provider.check_authorize_request({**params, "redirect_uri": "https://claude.ai/baska"})
    with pytest.raises(OAuthError):
        provider.check_authorize_request({**params, "code_challenge_method": "plain"})


def test_access_token_expires_and_refresh_rotates():
    clock = Clock()
    provider = OAuthProvider(PASSWORD, clock=clock)
    client_id = provider.register({"redirect_uris": [REDIRECT]})["client_id"]
    tokens = provider.exchange(_token_form(client_id, provider.approve(_authorize_params(client_id), PASSWORD)))
    clock.now += ACCESS_TTL + 1
    assert not provider.validate_access_token(tokens["access_token"])
    refreshed = provider.exchange({"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"],
                                   "client_id": client_id})
    assert provider.validate_access_token(refreshed["access_token"])
    with pytest.raises(OAuthError):  # eski refresh token tekrar kullanilamaz
        provider.exchange({"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]})


def test_confidential_client_needs_its_secret(provider):
    registered = provider.register({"redirect_uris": [REDIRECT], "token_endpoint_auth_method": "client_secret_post"})
    client_id, secret = registered["client_id"], registered["client_secret"]
    code = provider.approve(_authorize_params(client_id), PASSWORD)
    with pytest.raises(OAuthError) as exc:
        provider.exchange({**_token_form(client_id, code), "client_secret": "yanlis"})
    assert exc.value.error == "invalid_client"
    code = provider.approve(_authorize_params(client_id), PASSWORD)
    tokens = provider.exchange({**_token_form(client_id, code), "client_secret": secret})
    assert provider.validate_access_token(tokens["access_token"])


def test_register_rejects_unknown_auth_method(provider):
    with pytest.raises(OAuthError):
        provider.register({"redirect_uris": [REDIRECT], "token_endpoint_auth_method": "private_key_jwt"})


def test_tokens_survive_restart_without_plaintext(tmp_path):
    store = tmp_path / "store.json"
    provider = OAuthProvider(PASSWORD, store_path=str(store))
    client_id = provider.register({"redirect_uris": [REDIRECT]})["client_id"]
    tokens = provider.exchange(_token_form(client_id, provider.approve(_authorize_params(client_id), PASSWORD)))
    assert tokens["access_token"] not in store.read_text()
    assert OAuthProvider(PASSWORD, store_path=str(store)).validate_access_token(tokens["access_token"])


# --- HTTP uzerinden uctan uca ----------------------------------------------

class FakeService:
    def search(self, query, project=None, doc_type=None, top_k=6):
        return f"sonuc: {query}"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


@pytest.fixture
def base_url():
    oauth = OAuthProvider(PASSWORD)
    httpd = http_server.build_server(MCPServer(FakeService()), "t" * 32, port=0, oauth=oauth,
                                     public_url="https://rag.example.com")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _request(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        resp = urllib.request.build_opener(NoRedirect).open(req)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()
    return resp.status, resp.headers, resp.read()


def test_401_points_to_resource_metadata(base_url):
    status, headers, _ = _request(base_url + "/mcp", data=b"{}", method="POST")
    assert status == 401
    assert 'resource_metadata="https://rag.example.com/.well-known/oauth-protected-resource"' in headers["WWW-Authenticate"]
    _, _, body = _request(base_url + "/.well-known/oauth-protected-resource")
    assert json.loads(body)["authorization_servers"] == ["https://rag.example.com"]
    _, _, body = _request(base_url + "/.well-known/oauth-authorization-server")
    assert json.loads(body)["token_endpoint"] == "https://rag.example.com/token"


def test_http_oauth_flow_end_to_end(base_url):
    _, _, body = _request(base_url + "/register", data=json.dumps({"redirect_uris": [REDIRECT]}).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
    client_id = json.loads(body)["client_id"]
    params = _authorize_params(client_id)

    status, _, page = _request(base_url + "/authorize?" + urllib.parse.urlencode(params))
    assert status == 200 and b'name="password"' in page

    form = urllib.parse.urlencode({**params, "password": PASSWORD}).encode()
    status, headers, _ = _request(base_url + "/authorize", data=form, method="POST")
    assert status == 302
    location = urllib.parse.urlparse(headers["Location"])
    query = urllib.parse.parse_qs(location.query)
    assert headers["Location"].startswith(REDIRECT) and query["state"] == ["xyz"]

    token_form = urllib.parse.urlencode(_token_form(client_id, query["code"][0])).encode()
    _, _, body = _request(base_url + "/token", data=token_form, method="POST")
    access = json.loads(body)["access_token"]

    call = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "search", "arguments": {"query": "odul"}}}
    status, _, body = _request(base_url + "/mcp", data=json.dumps(call).encode(), method="POST",
                               headers={"Content-Type": "application/json", "Authorization": f"Bearer {access}"})
    assert status == 200 and json.loads(body)["result"]["content"][0]["text"] == "sonuc: odul"


def test_wrong_password_rerenders_login(base_url):
    _, _, body = _request(base_url + "/register", data=json.dumps({"redirect_uris": [REDIRECT]}).encode(), method="POST")
    params = _authorize_params(json.loads(body)["client_id"])
    form = urllib.parse.urlencode({**params, "password": "yanlis"}).encode()
    status, _, page = _request(base_url + "/authorize", data=form, method="POST")
    assert status == 401 and "parola yanlis".encode() in page
