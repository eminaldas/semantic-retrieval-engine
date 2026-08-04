"""Projenin tek PROJECT_ROOT kaynagi. Her modulun kendi BASE_DIR'ini hesaplamasi
yerine hepsi buradan import eder - dosyalar backend/ altinda farkli derinliklerde
olsa bile dogru sonucu verir. docs/superpowers/specs/2026-08-04-backend-restructure-docker-design.md."""

import os

_MARKER_FILE = "docker-compose.yml"


def _find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(current, _MARKER_FILE)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(f"{_MARKER_FILE} bulunamadi - proje kokune ulasilamadi.")
        current = parent


PROJECT_ROOT = _find_project_root()
