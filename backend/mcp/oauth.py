"""claude.ai custom connector'lari icin minimal OAuth 2.1 yetkilendirme sunucusu.

claude.ai connector'lari sabit bir 'Authorization' basligi gonderemez; ya kimliksiz
baglanir ya da MCP yetkilendirme spec'indeki OAuth akisini yurutur. Bu modul o akisin
ihtiyac duydugu parcalari saglar:
- RFC 9728 korunan kaynak metadatasi + RFC 8414 yetkilendirme sunucusu metadatasi
- RFC 7591 dinamik istemci kaydi (claude.ai kendini /register ile kaydeder)
- Authorization code + PKCE (yalnizca S256) ve donen (rotating) refresh token

Tek kullanicili: /authorize sayfasinda sahibin parolasi istenir (MCP_HTTP_TOKEN).
redirect_uri yalnizca claude.ai/claude.com (https) ya da loopback olabilir. Token'lar
diskte duz metin degil, yalnizca sha256 ozetleri olarak tutulur."""

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from urllib.parse import urlparse

ACCESS_TTL = 3600
REFRESH_TTL = 30 * 24 * 3600
CODE_TTL = 300
REMOTE_REDIRECT_HOSTS = {"claude.ai", "claude.com"}
LOOPBACK_HOSTS = {"localhost", "127.0.0.1"}
MAX_CLIENTS = 50
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 900
SCOPE = "mcp"
# claude.ai kayitta client_secret_post isteyebiliyor; gizli istemciler de desteklenir
CLIENT_AUTH_METHODS = ("none", "client_secret_post", "client_secret_basic")


class OAuthError(Exception):
    def __init__(self, error, description, status=400):
        super().__init__(description)
        self.error = error
        self.description = description
        self.status = status

    def body(self):
        return {"error": self.error, "error_description": self.description}


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pkce_challenge(verifier):
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")


def redirect_allowed(uri):
    parsed = urlparse(uri)
    host = parsed.hostname or ""
    if parsed.fragment:
        return False
    if host in LOOPBACK_HOSTS:
        return parsed.scheme in ("http", "https")
    return parsed.scheme == "https" and host in REMOTE_REDIRECT_HOSTS


class OAuthProvider:
    def __init__(self, password, store_path=None, clock=time.time):
        if not password or len(password) < 24:
            raise ValueError("OAuth giris parolasi en az 24 karakter olmali")
        self._password = password
        self._store_path = store_path
        self._clock = clock
        self._lock = threading.Lock()
        self._clients = {}   # client_id -> {"redirect_uris": [...], "client_name": str}
        self._codes = {}     # ozet(code) -> istek bilgisi; yalnizca bellekte, 5 dk
        self._access = {}    # ozet(token) -> {"client_id", "expires"}
        self._refresh = {}   # ozet(token) -> {"client_id", "expires"}
        self._failed = []    # basarisiz giris zamanlari
        self._load()

    # --- kalicilik -------------------------------------------------------

    def _load(self):
        if not self._store_path or not os.path.exists(self._store_path):
            return
        with open(self._store_path, encoding="utf-8") as f:
            data = json.load(f)
        self._clients = data.get("clients", {})
        self._access = data.get("access", {})
        self._refresh = data.get("refresh", {})

    def _save(self):
        if not self._store_path:
            return
        now = self._clock()
        for table in (self._access, self._refresh):
            for key in [k for k, v in table.items() if v["expires"] < now]:
                del table[key]
        os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
        tmp = self._store_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"clients": self._clients, "access": self._access, "refresh": self._refresh}, f)
        os.replace(tmp, self._store_path)

    # --- metadata --------------------------------------------------------

    def server_metadata(self, issuer):
        return {
            "issuer": issuer,
            "authorization_endpoint": issuer + "/authorize",
            "token_endpoint": issuer + "/token",
            "registration_endpoint": issuer + "/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": list(CLIENT_AUTH_METHODS),
        }

    def resource_metadata(self, issuer, resource_path):
        return {
            "resource": issuer + resource_path,
            "authorization_servers": [issuer],
            "bearer_methods_supported": ["header"],
        }

    # --- akis ------------------------------------------------------------

    def register(self, body):
        if not isinstance(body, dict):
            raise OAuthError("invalid_client_metadata", "govde JSON nesnesi olmali")
        uris = body.get("redirect_uris")
        if not isinstance(uris, list) or not uris or not all(isinstance(u, str) and redirect_allowed(u) for u in uris):
            raise OAuthError("invalid_redirect_uri", "redirect_uris yalnizca claude.ai/claude.com (https) ya da localhost olabilir")
        name = str(body.get("client_name", ""))[:100]
        auth_method = body.get("token_endpoint_auth_method") or "none"
        if auth_method not in CLIENT_AUTH_METHODS:
            raise OAuthError("invalid_client_metadata", f"desteklenen token_endpoint_auth_method: {', '.join(CLIENT_AUTH_METHODS)}")
        client_id = secrets.token_urlsafe(16)
        client_secret = secrets.token_urlsafe(32) if auth_method != "none" else None
        with self._lock:
            if len(self._clients) >= MAX_CLIENTS:
                raise OAuthError("invalid_client_metadata", "istemci kaydi siniri doldu")
            self._clients[client_id] = {
                "redirect_uris": uris,
                "client_name": name,
                "auth_method": auth_method,
                "secret_digest": _digest(client_secret) if client_secret else None,
            }
            self._save()
        response = {
            "client_id": client_id,
            "client_id_issued_at": int(self._clock()),
            "client_name": name,
            "redirect_uris": uris,
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": auth_method,
        }
        if client_secret:
            response["client_secret"] = client_secret
            response["client_secret_expires_at"] = 0
        return response

    def _authenticate_client(self, client_id, client_secret):
        # cagiran kilidi tutar; gizli istemcide secret zorunlu
        client = self._clients.get(client_id or "")
        if client is None:
            raise OAuthError("invalid_client", "bilinmeyen client_id", 401)
        expected = client.get("secret_digest")
        if expected and not hmac.compare_digest(_digest(client_secret or ""), expected):
            raise OAuthError("invalid_client", "client_secret yanlis", 401)

    def check_authorize_request(self, params):
        """Gecersiz istekte OAuthError; gecerliyse istemci adini doner (giris sayfasinda gosterilir)."""
        client = self._clients.get(params.get("client_id", ""))
        if client is None:
            raise OAuthError("invalid_client", "bilinmeyen client_id")
        if params.get("redirect_uri") not in client["redirect_uris"]:
            raise OAuthError("invalid_request", "redirect_uri bu istemci icin kayitli degil")
        if params.get("response_type") != "code":
            raise OAuthError("unsupported_response_type", "yalnizca response_type=code")
        if params.get("code_challenge_method") != "S256" or not params.get("code_challenge"):
            raise OAuthError("invalid_request", "PKCE (S256) zorunlu")
        return client["client_name"] or "isimsiz istemci"

    def approve(self, params, password):
        """Parola dogruysa tek kullanimlik authorization code doner."""
        self.check_authorize_request(params)
        with self._lock:
            now = self._clock()
            self._failed = [t for t in self._failed if now - t < LOCKOUT_SECONDS]
            if len(self._failed) >= MAX_FAILED_LOGINS:
                raise OAuthError("access_denied", "cok fazla hatali deneme; 15 dakika sonra tekrar dene", 429)
            if not hmac.compare_digest(password.encode("utf-8"), self._password.encode("utf-8")):
                self._failed.append(now)
                raise OAuthError("access_denied", "parola yanlis", 401)
            code = secrets.token_urlsafe(32)
            self._codes[_digest(code)] = {
                "client_id": params["client_id"],
                "redirect_uri": params["redirect_uri"],
                "code_challenge": params["code_challenge"],
                "expires": now + CODE_TTL,
            }
        return code

    def exchange(self, form):
        grant = form.get("grant_type")
        if grant == "authorization_code":
            return self._exchange_code(form)
        if grant == "refresh_token":
            return self._exchange_refresh(form)
        raise OAuthError("unsupported_grant_type", "yalnizca authorization_code ve refresh_token")

    def _exchange_code(self, form):
        with self._lock:
            entry = self._codes.pop(_digest(form.get("code", "")), None)  # tek kullanimlik
            if entry is None or entry["expires"] < self._clock():
                raise OAuthError("invalid_grant", "kod gecersiz ya da suresi dolmus")
            if form.get("client_id") != entry["client_id"] or form.get("redirect_uri") != entry["redirect_uri"]:
                raise OAuthError("invalid_grant", "client_id ya da redirect_uri eslesmiyor")
            self._authenticate_client(entry["client_id"], form.get("client_secret"))
            verifier = form.get("code_verifier", "")
            if not verifier or not hmac.compare_digest(pkce_challenge(verifier), entry["code_challenge"]):
                raise OAuthError("invalid_grant", "PKCE dogrulanamadi")
            return self._issue(entry["client_id"])

    def _exchange_refresh(self, form):
        with self._lock:
            entry = self._refresh.pop(_digest(form.get("refresh_token", "")), None)  # rotasyon
            if entry is None or entry["expires"] < self._clock():
                raise OAuthError("invalid_grant", "refresh token gecersiz ya da suresi dolmus")
            if form.get("client_id") and form["client_id"] != entry["client_id"]:
                raise OAuthError("invalid_grant", "client_id eslesmiyor")
            self._authenticate_client(entry["client_id"], form.get("client_secret"))
            return self._issue(entry["client_id"])

    def _issue(self, client_id):
        # cagiran kilidi tutar
        now = self._clock()
        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(32)
        self._access[_digest(access)] = {"client_id": client_id, "expires": now + ACCESS_TTL}
        self._refresh[_digest(refresh)] = {"client_id": client_id, "expires": now + REFRESH_TTL}
        self._save()
        return {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": ACCESS_TTL,
            "refresh_token": refresh,
            "scope": SCOPE,
        }

    def validate_access_token(self, token):
        if not token:
            return False
        with self._lock:
            entry = self._access.get(_digest(token))
            return entry is not None and entry["expires"] >= self._clock()
