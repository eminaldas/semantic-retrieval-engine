"""MCP araclarinin arkasindaki is mantigi. Sunucudan (protokol) ayri tutuldu ki
araclar Qdrant/model olmadan, sahte bir runtime ile test edilebilsin."""

import os
import re
import threading
from collections import Counter
from datetime import datetime, timezone

from backend.knowledge import projects, routing, search, store, sync
from backend.shared import tracing

MAX_BATCH_QUERIES = 4
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STALE_TOLERANCE_SECONDS = 2  # dosya sistemi zaman damgasi yuvarlamasi


def _check_date(name, value):
    if value and not _DATE_RE.match(value):
        raise ValueError(f"{name} YYYY-AA-GG biciminde olmali, gelen: {value}")


def _timestamp(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _ago(seconds):
    if seconds < 90:
        return f"{int(seconds)} sn once"
    if seconds < 5400:
        return f"{int(seconds // 60)} dk once"
    if seconds < 172800:
        return f"{int(seconds // 3600)} saat once"
    return f"{int(seconds // 86400)} gun once"


class KnowledgeService:
    def __init__(self, runtime, config_loader=projects.load_config, notes_dir=projects.NOTES_DIR):
        self.runtime = runtime
        self.config_loader = config_loader
        self.notes_dir = notes_dir
        self.sync_lock = threading.Lock()  # elle sync, add_note ve izleyici ayni anda indekslemesin
        self.last_sync = {}  # proje -> (zaman, ozet); index_status gosterir
        self.agent_factory = None  # ask araci; testler sahte ajan verir
        self.project_hints = True  # proje verilmeyen aramada baska projeye uyma notu (routing.py)

    def _config(self):
        return self.config_loader()

    def _projects(self):
        return {k: v for k, v in self._config().items() if not k.startswith("_")}

    def resolve_project(self, project):
        """Proje verilmezse: config'teki _default_project, o yoksa tek proje tanimliysa o, degilse hepsi."""
        raw = self._config()
        config = {k: v for k, v in raw.items() if not k.startswith("_")}
        if project:
            if project not in config:
                raise ValueError(f"Bilinmeyen proje: {project}. Tanimli projeler: {', '.join(sorted(config)) or '(yok)'}")
            return project
        if raw.get("_default_project") in config:
            return raw["_default_project"]
        if len(config) == 1:
            return next(iter(config))
        return None  # tum projelerde ara

    def _require_project(self, project):
        project = self.resolve_project(project)
        if not project:
            raise ValueError("Birden fazla proje var; project parametresini verin.")
        return project

    def list_projects(self):
        config = self._projects()
        if not config:
            return "Tanimli proje yok. knowledge_projects.json dosyasina bir proje ekleyin."
        lines = [f"- {name}: {settings.get('description', '')} ({settings.get('root')})".rstrip() for name, settings in sorted(config.items())]
        return "Tanimli projeler:\n" + "\n".join(lines)

    # --- arama -------------------------------------------------------------

    def search(self, query=None, queries=None, project=None, doc_type=None, top_k=5, path_prefix=None,
               modified_after=None, modified_before=None):
        batch = [q for q in (queries or []) if q and q.strip()]
        if query and query.strip():
            batch.insert(0, query)
        if not batch:
            raise ValueError("query (ya da queries) bos olamaz")
        if len(batch) > MAX_BATCH_QUERIES:
            raise ValueError(f"Bir cagrida en fazla {MAX_BATCH_QUERIES} soru")
        _check_date("modified_after", modified_after)
        _check_date("modified_before", modified_before)
        top_k = max(1, min(int(top_k), 15))
        options = dict(project=self.resolve_project(project), doc_type=doc_type, path_prefix=path_prefix,
                       modified_after=modified_after, modified_before=modified_before)
        # Proje verilmediyse soru baska projeye daha iyi uyuyor mu diye bakilir (routing.py); sadece not duser.
        hint_projects = sorted(self._projects()) if not project and self.project_hints else []
        if len(batch) == 1:
            return self._with_hint(search.format_results(self._search_one(batch[0], top_k, options)),
                                   batch[0], options["project"], hint_projects)
        # Toplu sorguda butce sorulara bolunur ki toplam baglam tek sorgununkinin ~1.5 katini gecmesin.
        budget = max(1500, int(search.MAX_OUTPUT_CHARS * 1.5) // len(batch))
        sections = [f"## Soru {i}: {q}\n\n" + self._with_hint(
                        search.format_results(self._search_one(q, top_k, options), max_chars=budget),
                        q, options["project"], hint_projects)
                    for i, q in enumerate(batch, start=1)]
        return "\n\n======\n\n".join(sections)

    def _with_hint(self, text, query, current, projects):
        if len(projects) < 2:
            return text
        hint = routing.project_hint(self.runtime.client(), self.runtime.embedder(), self.runtime.reranker(),
                                    query, current, projects)
        return f"{text}\n\n{hint}" if hint else text

    def ask(self, question, project=None):
        """Arastirma ajani (Gemini) soruyu kendi arayip okuyarak cevaplar; Claude'a parcalar yerine
        kaynakli kisa bir cevap doner. Altin sette kaynak isabeti 1.00, cevapsiz soruda 'bulamadim'
        1.00 (2026-10-25); soru basina ~10 sn ve ~0.004 $."""
        from backend.knowledge.agent import ResearchAgent, cost_usd

        if not question or not question.strip():
            raise ValueError("question bos olamaz")
        if self.agent_factory is None:
            self.agent_factory = ResearchAgent
        result = self.agent_factory(self, project=project).ask(question)
        sources = ", ".join(result.citations) or "yok"
        footer = (f"\n\n[Kaynaklar: {sources} · {len(result.steps)} arama/okuma · {result.seconds:.0f} sn"
                  f" · ~${cost_usd(result) or 0:.4f}]")
        return (result.answer or "Cevap uretilemedi.") + footer

    def _search_one(self, query, top_k, options):
        results = search.search(self.runtime.client(), self.runtime.embedder(), self.runtime.reranker(), query,
                                top_k=top_k, **options)
        results = self.mark_stale(results)
        stale = [f"{r['path']}: {r['stale']}" for r in results if r.get("stale")]
        if stale:
            tracing.event("stale_results", query=tracing.clip(query), results=stale)
        return results

    def mark_stale(self, results):
        """Sonucun dosyasi indeksten sonra degismis ya da silinmisse 'stale' alanina nedenini yazar."""
        config = self._projects()
        for r in results:
            settings = config.get(r.get("project"))
            if not settings or not r.get("modified_at"):
                continue
            path = projects.source_file(r["project"], settings, r["path"], self.notes_dir)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                r["stale"] = "dosya silinmis"
                continue
            if mtime > _timestamp(r["modified_at"]) + STALE_TOLERANCE_SECONDS:
                changed = datetime.fromtimestamp(mtime, timezone.utc).strftime("%Y-%m-%d %H:%M")
                r["stale"] = f"dosya {changed} UTC'de degisti, indeks eski"
        return results

    # --- belgeler ----------------------------------------------------------

    def list_documents(self, project=None):
        project = self.resolve_project(project)
        rows = store.scroll_all(self.runtime.client(), store.project_filter(project=project),
                                ("source_id", "path", "doc_type", "modified_at"))
        sources = store.summarize_sources(rows)
        if not sources:
            return "Indekslenmis belge yok. Once sync_project calistirin."
        lines = [f"- {s['path']} ({s['doc_type']}, {s['chunk_count']} parca, degisiklik {(s['modified_at'] or '?')[:10]})" for s in sources]
        return f"{len(sources)} belge:\n" + "\n".join(lines)

    def get_document(self, path, project=None, max_chars=20000):
        project = self._require_project(project)
        rows = store.scroll_all(self.runtime.client(), store.project_filter(source_id=projects.source_id_for(project, path)),
                                ("chunk_id", "location", "text"))
        if not rows:
            return f"Belge bulunamadi: {path}. list_documents ile yollari kontrol edin."
        rows.sort(key=lambda r: r["chunk_id"])
        text = "\n\n".join(f"({r['location']})\n{r['text'].strip()}" for r in rows)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... {len(text) - max_chars} karakter kesildi; search ile ilgili bolumu arayin]"
        return f"{path}\n\n{text}"

    def overview(self, project=None):
        """Projenin haritasi: klasorler, son degisen belgeler, son notlar. Ilk soruda yon bulmak icin."""
        project = self._require_project(project)
        rows = store.scroll_all(self.runtime.client(), store.project_filter(project=project),
                                ("source_id", "path", "doc_type", "modified_at", "section", "superseded_by"))
        sources = store.summarize_sources(rows)
        if not sources:
            return f"{project}: indekslenmis belge yok. Once sync_project calistirin."
        sections = {}
        for row in sorted(rows, key=lambda r: r.get("section") or "~"):
            if row.get("section"):
                sections.setdefault(row["source_id"], row["section"].split(" > ")[0])
        docs = [s for s in sources if s["doc_type"] == "doc"]
        notes = [s for s in sources if s["doc_type"] == "note"]
        folders = Counter(s["path"].split("/")[0] + "/" if "/" in s["path"] else "(kok)" for s in docs)
        description = self._projects().get(project, {}).get("description", "")
        lines = [f"# {project}" + (f": {description}" if description else ""),
                 f"{len(docs)} belge, {len(notes)} not, {len(rows)} parca.", "", "Klasorler:"]
        lines += [f"- {folder} {count} belge" for folder, count in sorted(folders.items())]

        def latest(items, n):
            return sorted(items, key=lambda s: s["modified_at"] or "", reverse=True)[:n]

        lines += ["", "Son degisen belgeler:"]
        lines += [f"- {s['path']} ({(s['modified_at'] or '?')[:10]})"
                  + (f": {sections[s['source_id']]}" if s["source_id"] in sections else "") for s in latest(docs, 10)]
        if notes:
            replaced = {r["source_id"] for r in rows if r.get("superseded_by")}
            lines += ["", "Son notlar:"]
            lines += [f"- {s['path']} ({(s['modified_at'] or '?')[:10]})" + (" (eski)" if s["source_id"] in replaced else "")
                      for s in latest(notes, 5)]
        return "\n".join(lines)

    # --- indeks ------------------------------------------------------------

    def index_status(self, project=None):
        """Git status gibi: diskte indeksten farkli olan dosyalar + son senkron + parca sayisi."""
        names = [self.resolve_project(project)] if self.resolve_project(project) else sorted(self._projects())
        config = self._config()
        blocks = []
        for name in names:
            settings = projects.project_settings(config, name)
            disk = {rel: projects.file_hash(path) for rel, path in sync.collect_files(name, settings, self.notes_dir).items()}
            manifest = projects.load_manifest(name)
            new = sorted(p for p in disk if p not in manifest)
            changed = sorted(p for p in disk if p in manifest and manifest[p] != disk[p])
            removed = sorted(p for p in manifest if p not in disk)
            chunks = self.runtime.client().count(store.COLLECTION_NAME, count_filter=store.project_filter(project=name)).count
            lines = [f"{name}: {len(manifest)} dosya indekste, {chunks} parca."]
            if name in self.last_sync:
                when, summary = self.last_sync[name]
                lines.append(f"Son senkron: {_ago(datetime.now(timezone.utc).timestamp() - when)} ({summary}).")
            elif os.path.exists(projects.manifest_path(name)):
                age = datetime.now(timezone.utc).timestamp() - os.path.getmtime(projects.manifest_path(name))
                lines.append(f"Son senkron: {_ago(age)}.")
            if not (new or changed or removed):
                lines.append("Indeks guncel.")
            for label, items in (("yeni", new), ("degismis", changed), ("silinmis", removed)):
                if items:
                    shown = ", ".join(items[:10]) + (f" ve {len(items) - 10} dosya daha" if len(items) > 10 else "")
                    lines.append(f"{label} ({len(items)}): {shown}")
            if new or changed or removed:
                lines.append("Guncellemek icin: sync_project")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def sync_project(self, project=None, full=False):
        project = self._require_project(project)
        report = self._sync(project, full=full)
        return (f"{project}: {len(report['indexed'])} dosya indekslendi ({report['chunks']} parca), "
                f"{len(report['removed'])} dosya kaldirildi.")

    def _sync(self, project, full=False, only=None):
        with self.sync_lock:
            report = sync.sync_project(project, self.runtime.client(), self.runtime.embedder(), config=self._config(),
                                       full=full, only=only, log=lambda *_: None,
                                       chunk_model=self.runtime.chunker_model(), notes_dir=self.notes_dir,
                                       contextualizer=self.runtime.contextualizer())
        summary = f"{len(report['indexed'])} dosya indekslendi, {len(report['removed'])} kaldirildi"
        self.last_sync[project] = (datetime.now(timezone.utc).timestamp(), summary)
        return report

    # --- yazma -------------------------------------------------------------

    def add_note(self, title, body, project=None, supersedes=None):
        project = self._require_project(project)
        if not title.strip() or not body.strip():
            raise ValueError("title ve body bos olamaz")
        old_path = None
        if supersedes:
            if not supersedes.startswith(projects.NOTES_PREFIX) or ".." in supersedes:
                raise ValueError(f"supersedes bir not yolu olmali (notes/... ile baslar), gelen: {supersedes}")
            old_path = projects.source_file(project, {}, supersedes, self.notes_dir)
            if not os.path.isfile(old_path):
                raise ValueError(f"Yerine gecilecek not bulunamadi: {supersedes}")
        folder = os.path.join(self.notes_dir, project)
        os.makedirs(folder, exist_ok=True)
        filename = projects.note_filename(title)
        rel = projects.NOTES_PREFIX + filename
        with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
            f.write(projects.render_note(project, title, body, supersedes=supersedes))
        changed = [rel]
        if old_path:
            with open(old_path, "a", encoding="utf-8") as f:
                f.write("\n" + projects.SUPERSEDED_MARKER.format(path=rel) + "\n")
            changed.append(supersedes)
        self._sync(project, only=changed)
        suffix = f" Eski not ({supersedes}) geri plana alindi." if supersedes else ""
        return f"Not kaydedildi ve indekslendi: {rel}.{suffix}"

    def add_document(self, name, content, project=None, overwrite=False):
        """Claude'un yazdigi tam bir belgeyi (plan, ozet, rapor) motorun not klasorune
        'notes/belgeler/' altina yazar ve indeksler. Proje reposuna dokunmaz."""
        project = self._require_project(project)
        if not name.strip() or not content.strip():
            raise ValueError("name ve content bos olamaz")
        rel = projects.document_relpath(name)
        path = projects.source_file(project, {}, rel, self.notes_dir)
        if os.path.exists(path) and not overwrite:
            raise ValueError(f"{rel} zaten var. Guncellemek icin overwrite=true verin.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        report = self._sync(project, only=[rel])
        return f"Belge kaydedildi ve indekslendi: {rel} ({report['chunks']} parca)."
