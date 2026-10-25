"""Klasor izleyici: proje klasorlerini periyodik tarar, bir dosya kaydedilip birkac saniye
degismeden kalinca o projeyi artimli olarak yeniden indeksler.

Bilerek olay tabanli (watchdog) degil, yoklama: ek bagimlilik yok, ag/OneDrive
klasorlerinde de calisir ve ~300 Markdown dosyasinin mtime taramasi milisaniyeler surer.
Hash'e sadece degisiklik gorulunce bakilir (sync_project zaten hash karsilastirir).

Sunucu acilirken bir tur senkron da yapar: sunucu kapaliyken degisen dosyalar yakalanir."""

import os
import sys
import threading
import time

from backend.knowledge import projects, sync

DEFAULT_INTERVAL_SECONDS = 5
DEFAULT_SETTLE_SECONDS = 3  # editor dosyayi parca parca yaziyorsa yarim hali indekslenmesin
DEFAULT_RECONCILE_SECONDS = 120  # degisiklik gorulmese de bu aralikla artimli senkron (hash karsilastirmasi, ucuz)


def snapshot(project, settings, notes_dir=projects.NOTES_DIR):
    """{relpath: (mtime, size)} - silinen ya da okunamayan dosyalar atlanir."""
    state = {}
    for rel, path in sync.collect_files(project, settings, notes_dir).items():
        try:
            info = os.stat(path)
        except OSError:
            continue
        state[rel] = (info.st_mtime, info.st_size)
    return state


class ProjectWatcher:
    def __init__(self, service, interval=DEFAULT_INTERVAL_SECONDS, settle=DEFAULT_SETTLE_SECONDS,
                 reconcile=DEFAULT_RECONCILE_SECONDS, log=lambda msg: print(msg, file=sys.stderr, flush=True),
                 clock=time.monotonic):
        self.service = service
        self.interval = interval
        self.settle = settle
        self.reconcile = reconcile
        self.log = log
        self.clock = clock
        self.snapshots = {}
        self.pending = {}  # proje -> son degisikligin gorulme zamani
        self.reconciled = {}  # proje -> son senkron zamani
        self._stop = threading.Event()

    def _watched(self):
        raw = self.service._config()
        if raw.get("_watch") is False:
            return {}
        return {k: v for k, v in raw.items() if not k.startswith("_") and v.get("watch", True)}

    def catch_up(self):
        """Acilista: her projeyi bir kez artimli senkronlar ve mevcut durumu kaydeder."""
        config = self.service._config()
        for name in self._watched():
            settings = projects.project_settings(config, name)
            self.snapshots[name] = snapshot(name, settings, self.service.notes_dir)
            self._sync(name, reason="acilis")

    def poll_once(self):
        """Bir tarama turu. Degisiklik gorulen proje 'settle' sn sakin kalinca senkronlanir.
        Ayrica her 'reconcile' sn'de bir, degisiklik gorulmese de artimli senkron calisir: iki tarama
        arasinda olusup silinen ya da baska yoldan indekslenen dosyalar (add_document) boylece kacmaz."""
        config = self.service._config()
        now = self.clock()
        for name in self._watched():
            current = snapshot(name, projects.project_settings(config, name), self.service.notes_dir)
            if name in self.snapshots and current != self.snapshots[name]:
                self.pending[name] = now  # her yeni degisiklik bekleme suresini yeniler
            self.snapshots[name] = current
            if name in self.pending and now - self.pending[name] >= self.settle:
                del self.pending[name]
                self._sync(name, reason="degisiklik")
            elif name not in self.pending and now - self.reconciled.get(name, now) >= self.reconcile:
                self._sync(name, reason="periyodik kontrol")
            self.reconciled.setdefault(name, now)

    def _sync(self, name, reason):
        self.reconciled[name] = self.clock()
        try:
            report = self.service._sync(name)
        except Exception as exc:  # izleyici olmemeli; bir sonraki degisiklikte yeniden dener
            self.log(f"izleyici: {name} senkronu basarisiz ({reason}): {exc}")
            return None
        if report["indexed"] or report["removed"]:
            self.log(f"izleyici: {name} ({reason}): {len(report['indexed'])} dosya indekslendi, "
                     f"{len(report['removed'])} kaldirildi")
        return report

    def run(self):
        started = time.monotonic()
        try:
            self.catch_up()
            self.log(f"izleyici: acilis senkronu bitti ({time.monotonic() - started:.0f} sn)")
        except Exception as exc:
            self.log(f"izleyici: acilis senkronu basarisiz: {exc}")
        while not self._stop.wait(self.interval):
            try:
                self.poll_once()
            except Exception as exc:
                self.log(f"izleyici: tarama hatasi: {exc}")

    def start(self):
        thread = threading.Thread(target=self.run, name="project-watcher", daemon=True)
        thread.start()
        return thread

    def stop(self):
        self._stop.set()
