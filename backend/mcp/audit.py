"""MCP denetim gunlugu: hangi baglanti hangi araci hangi projede cagirdi (JSONL, satir basina bir cagri).

Icerik yazilmaz: sorgu, belge metni ve cevap gunluge girmez; yalnizca zaman, workspace, istemci, arac,
istenen proje, sonuc ve sure. Phoenix izleri icerik tasir ve yonetici aracidir; bu gunluk ise
"kim neye dokundu" sorusunun kalici cevabidir.

Ortam degiskenleri (.env):
  RAG_AUDIT=0          gunlugu kapatir
  RAG_AUDIT_PATH       varsayilan artifacts/audit/mcp_audit.jsonl"""

import json
import os
import sys
import threading
from datetime import datetime, timezone

from backend.shared.paths import PROJECT_ROOT

DEFAULT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "audit", "mcp_audit.jsonl")


class AuditLog:
    def __init__(self, path=DEFAULT_PATH, clock=None):
        self.path = path
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, **record):
        line = json.dumps({"ts": self._clock().isoformat(timespec="seconds"), **record}, ensure_ascii=False)
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:  # gunluk yazilamazsa istek yine cevaplanir
            print(f"Denetim gunlugu yazilamadi: {exc}", file=sys.stderr, flush=True)


def from_env():
    if os.environ.get("RAG_AUDIT", "1") == "0":
        return None
    return AuditLog(os.environ.get("RAG_AUDIT_PATH") or DEFAULT_PATH)
