"""MCP araclarinin arkasindaki is mantigi. Sunucudan (protokol) ayri tutuldu ki
araclar Qdrant/model olmadan, sahte bir runtime ile test edilebilsin."""

import copy
import os
import re
import threading
from collections import Counter
from datetime import datetime, timezone

from backend.knowledge import projects, routing, search, store, summaries, sync
from backend.knowledge.scope import Scope
from backend.shared import tracing

MAX_BATCH_QUERIES = 4
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STALE_TOLERANCE_SECONDS = 2  # dosya sistemi zaman damgasi yuvarlamasi
# Brifing her sohbetin basinda baglama girer; kisa kalmali (RAG_GERI_BILDIRIM: baglami sisirme).
BRIEF_MAX_CHARS = 1500
BRIEF_NOTES = 5
BRIEF_DOCS = 8
BRIEF_TASKS = 5
NOTE_HEAD_CHARS = 4000  # kunye icin notun basi yeter


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
    def __init__(self, runtime, config_loader=projects.load_config, notes_dir=projects.NOTES_DIR, scope=None):
        self.runtime = runtime
        self.scope = scope or Scope.admin()  # bu baglantinin gorebildigi projeler (scope.py)
        self.config_loader = config_loader
        self.notes_dir = notes_dir
        self.sync_lock = threading.Lock()  # elle sync, add_note ve izleyici ayni anda indekslemesin
        self.last_sync = {}  # proje -> (zaman, ozet); index_status gosterir
        self.agent_factory = None  # ask araci; testler sahte ajan verir
        self.ask_spend = {}  # (workspace, gun) -> ask harcamasi ($); scoped kopyalar paylasir
        self.today = lambda: datetime.now().date().isoformat()
        self.project_hints = True  # proje verilmeyen aramada baska projeye uyma notu (routing.py)

    def scoped(self, scope):
        """Ayni runtime, kilit ve senkron durumunu paylasan, yalnizca scope'un projelerini goren kopya.
        Workspace baglantilari bunu kullanir; ask ajani da cagrildigi kopyanin kapsamini devralir."""
        view = copy.copy(self)
        view.scope = scope
        return view

    def _config(self):
        return self.config_loader()

    def _projects(self):
        """Kapsamin gorebildigi projeler; disaridakiler hic yokmus gibi davranilir."""
        return {k: v for k, v in self._config().items() if not k.startswith("_") and self.scope.can_read(k)}

    def resolve_project(self, project):
        """Proje verilmezse: kapsamin varsayilani, config'teki _default_project, tek proje varsa o,
        degilse None (kapsamdaki tum projeler; bkz. _project_filter)."""
        raw = self._config()
        config = self._projects()
        if project:
            if project not in config:
                if not self.scope.is_admin:  # baska workspace'teki projenin varligi belli edilmez
                    raise ValueError(f"Bu workspace'te '{project}' adli proje yok. Erisilebilir projeler: "
                                     f"{', '.join(sorted(config)) or '(yok)'}")
                raise ValueError(f"Bilinmeyen proje: {project}. Tanimli projeler: {', '.join(sorted(config)) or '(yok)'}")
            return project
        if self.scope.default in config:
            return self.scope.default
        if raw.get("_default_project") in config:
            return raw["_default_project"]
        if len(config) == 1:
            return next(iter(config))
        return None  # kapsamdaki tum projelerde ara

    def _project_filter(self, project):
        """resolve_project sonucunu Qdrant filtresine cevirir: None, yoneticide 'tum projeler',
        workspace'te 'kapsamdaki projelerden biri' olur. Filtresiz arama yalnizca yoneticide var."""
        if project or self.scope.is_admin:
            return project
        return tuple(sorted(self._projects()))

    def _require_project(self, project):
        project = self.resolve_project(project)
        if not project:
            raise ValueError("Birden fazla proje var; project parametresini verin.")
        return project

    def _require_writable(self, project):
        project = self._require_project(project)
        if not self.scope.can_write(project):
            raise ValueError(f"'{project}' bu workspace'te salt okunur; not, belge ve senkron yazilamaz.")
        return project

    def list_projects(self):
        config = self._projects()
        if not config:
            return "Tanimli proje yok. knowledge_projects.json dosyasina bir proje ekleyin."

        def line(name, settings):
            if not self.scope.is_admin:  # diskteki kok yolu workspace baglantisina gosterilmez
                mode = "" if self.scope.can_write(name) else " [salt okunur]"
                return f"- {name}: {settings.get('description', '')}".rstrip(": ") + mode
            return f"- {name}: {settings.get('description', '')} ({settings.get('root')})".rstrip()

        return "Tanimli projeler:\n" + "\n".join(line(name, settings) for name, settings in sorted(config.items()))

    # --- arama -------------------------------------------------------------

    def search(self, query=None, queries=None, project=None, doc_type=None, top_k=5, path_prefix=None,
               modified_after=None, modified_before=None, kind=None):
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
        if kind is not None and kind not in projects.KINDS:
            raise ValueError(f"kind su degerlerden biri olmali: {', '.join(projects.KINDS)}")
        options = dict(project=self._project_filter(self.resolve_project(project)), doc_type=doc_type, path_prefix=path_prefix,
                       modified_after=modified_after, modified_before=modified_before, kind=kind)
        # Proje verilmediyse soru baska projeye daha iyi uyuyor mu diye bakilir (routing.py); sadece not duser.
        hint_projects = sorted(self._projects()) if not project and self.project_hints else []
        if len(batch) == 1:
            results = self._search_one(batch[0], top_k, options)
            return self._with_hint(search.format_results(results), batch[0], options["project"], hint_projects,
                                   results)
        # Toplu sorguda butce sorulara bolunur ki toplam baglam tek sorgununkinin ~1.5 katini gecmesin.
        budget = max(1500, int(search.MAX_OUTPUT_CHARS * 1.5) // len(batch))
        sections = []
        for i, q in enumerate(batch, start=1):
            results = self._search_one(q, top_k, options)
            text = self._with_hint(search.format_results(results, max_chars=budget), q, options["project"],
                                   hint_projects, results)
            sections.append(f"## Soru {i}: {q}\n\n{text}")
        return "\n\n======\n\n".join(sections)

    def _with_hint(self, text, query, current, projects, results):
        if len(projects) < 2 or isinstance(current, tuple):  # kapsamdaki tum projelerde arandiysa ipucu anlamsiz
            return text
        best = max((r["rerank_score"] for r in results), default=0.0)
        hint = routing.project_hint(self.runtime.client(), self.runtime.embedder(), self.runtime.reranker(),
                                    query, current, projects, current_score=best)
        return f"{text}\n\n{hint}" if hint else text

    def ask(self, question, project=None):
        """Arastirma ajani (Gemini) soruyu kendi arayip okuyarak cevaplar; Claude'a parcalar yerine
        kaynakli kisa bir cevap doner. Altin sette kaynak isabeti 1.00, cevapsiz soruda 'bulamadim'
        1.00 (2026-10-25); soru basina ~10 sn ve ~0.004 $."""
        from backend.knowledge.agent import ResearchAgent, cost_usd

        if not question or not question.strip():
            raise ValueError("question bos olamaz")
        if not self.scope.external_llm:
            raise ValueError("Bu workspace'te dis LLM kapali; ask kullanilamaz. search ve get_document ile cevaplayin.")
        key = (self.scope.name, self.today())
        budget = self.scope.ask_daily_usd
        if budget is not None and self.ask_spend.get(key, 0.0) >= budget:
            raise ValueError(f"Bu workspace'in gunluk ask siniri (${budget:g}) doldu; yarina kadar search kullanin.")
        if self.agent_factory is None:
            self.agent_factory = ResearchAgent
        result = self.agent_factory(self, project=project).ask(question)
        self.ask_spend[key] = self.ask_spend.get(key, 0.0) + (cost_usd(result) or 0.0)
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
        project = self._project_filter(self.resolve_project(project))
        rows = store.scroll_all(self.runtime.client(), store.project_filter(project=project),
                                ("source_id", "path", "doc_type", "modified_at"))
        sources = [s for s in store.summarize_sources(rows) if s["doc_type"] != summaries.DOC_TYPE]
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
        sources = [s for s in store.summarize_sources(rows) if s["doc_type"] != summaries.DOC_TYPE]
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
        summary_rows = {r["path"]: r for r in rows if r.get("doc_type") == summaries.DOC_TYPE}
        lines = [f"# {project}" + (f": {description}" if description else ""),
                 f"{len(docs)} belge, {len(notes)} not, {len(rows) - len(summary_rows)} parca."]
        project_summary = summary_rows.get(summaries.PREFIX)
        if project_summary:
            text = store.scroll_all(self.runtime.client(), store.project_filter(source_id=project_summary["source_id"]),
                                    ("text",))
            if text:
                lines += ["", "Ozet: " + text[0]["text"].strip()]
        lines += ["", "Klasorler:"]
        lines += [f"- {folder} {count} belge" for folder, count in sorted(folders.items())]
        folder_summaries = sorted(p for p in summary_rows if p.endswith("/") and p != summaries.PREFIX)
        if folder_summaries:
            lines.append("Klasor ozetleri (get_document ile): " + ", ".join(folder_summaries))

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

    def brief(self, project=None):
        """Yeni sohbetin ilk adimi: 'nerede kalmistik'. Kapsamdaki projeler, son notlar (yerini yenisi
        almislar haric), acik isler ve son degisen belgeler; LLM yok, BRIEF_MAX_CHARS'i gecmez."""
        resolved = self.resolve_project(project)
        names = [resolved] if resolved else sorted(self._projects())
        if not names:
            return "Bu baglantida proje yok."
        rows = store.scroll_all(self.runtime.client(), store.project_filter(project=tuple(names)),
                                ("source_id", "project", "path", "doc_type", "modified_at", "section",
                                 "superseded_by", "kind", "status", "record_id"))
        sources = {}
        for row in rows:
            entry = sources.setdefault(row["source_id"], dict(row, title=None))
            if row.get("section") and not entry["title"]:
                entry["title"] = row["section"].split(" > ")[0]
            if row.get("superseded_by"):
                entry["superseded_by"] = row["superseded_by"]

        def newest(items, n):
            return sorted(items, key=lambda e: e.get("modified_at") or "", reverse=True)[:n]

        def label(entry):
            name = entry["path"] if len(names) == 1 else f"{entry['project']}: {entry['path']}"
            return f"- {(entry.get('modified_at') or '?')[:10]} {name}"

        config = self._projects()
        header = "# Brifing" + ("" if self.scope.is_admin else f" · workspace {self.scope.name}")
        lines = [header, ""]
        for name in names:
            description = config.get(name, {}).get("description", "")
            mode = "" if self.scope.can_write(name) else " [salt okunur]"
            lines.append(f"Proje {name}{mode}" + (f": {description}" if description else ""))
        notes = [e for e in sources.values() if e["doc_type"] == "note" and not e.get("superseded_by")]
        docs = [e for e in sources.values() if e["doc_type"] == "doc"]
        tasks = [e for e in notes if e.get("kind") == projects.TASK_KIND and e.get("status") != "done"]
        if tasks:
            lines += ["", "Acik isler:"]
            lines += [f"- {e.get('record_id') or '?'} {e['title'] or e['path']}" for e in newest(tasks, BRIEF_TASKS)]
        lines += ["", "Son notlar (kararlar ve ilerleme):"]
        lines += [label(e) + (f": {e['title']}" if e["title"] else "") for e in newest(notes, BRIEF_NOTES)] or ["- (not yok)"]
        lines += ["", "Son degisen belgeler:"]
        lines += [label(e) for e in newest(docs, BRIEF_DOCS)] or ["- (belge yok)"]
        lines += ["", "Ayrinti icin search; bir notun tamami icin get_document. Is bitince checkpoint ya da "
                      "add_note ile kaydet, kapanan isi close_task ile kapat."]
        text = "\n".join(lines)
        if len(text) > BRIEF_MAX_CHARS:
            text = text[:BRIEF_MAX_CHARS].rsplit("\n", 1)[0] + "\n[... kisaltildi]"
        return text

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
        project = self._require_writable(project)
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

    def add_note(self, title, body, project=None, supersedes=None, kind="note"):
        project = self._require_writable(project)
        rel, old = self._write_note(project, title, body, kind=kind, supersedes=supersedes)
        self._sync(project, only=[rel] + ([old] if old else []))
        meta = projects.note_meta(self._read_note(project, rel))
        suffix = f" Eski kayit ({old}) geri plana alindi." if old else ""
        return f"Not kaydedildi ve indekslendi: {rel} ({meta['record_id']}, {kind}).{suffix}"

    def checkpoint(self, summary, decisions=None, open_tasks=None, project=None):
        """Oturum kaydi: ozet bir 'session' notu, her karar bir 'decision', her is bir acik 'task' olur.
        Boylece brifing acik isleri ve son kararlari listeleyebilir (plan H2)."""
        project = self._require_writable(project)
        decisions = [d for d in (decisions or []) if d and d.strip()]
        tasks = [t for t in (open_tasks or []) if t and t.strip()]
        if not summary or not summary.strip():
            raise ValueError("summary bos olamaz")
        written, ids = [], []
        for kind, items in (("decision", decisions), (projects.TASK_KIND, tasks)):
            for item in items:
                title = item.strip().split("\n")[0][:80]
                rel, _ = self._write_note(project, title, item, kind=kind)
                written.append(rel)
                ids.append(projects.note_meta(self._read_note(project, rel))["record_id"])
        body = summary.strip()
        for label, items in (("Kararlar", decisions), ("Acik isler", tasks)):
            if items:
                body += f"\n\n## {label}\n" + "\n".join(f"- {i.strip()}" for i in items)
        rel, _ = self._write_note(project, f"Oturum ozeti {datetime.now(timezone.utc):%Y-%m-%d %H:%M}", body,
                                  kind="session")
        written.append(rel)
        self._sync(project, only=written)
        session_id = projects.note_meta(self._read_note(project, rel))["record_id"]
        counts = f"{len(decisions)} karar, {len(tasks)} acik is"
        return (f"Oturum kaydedildi: {rel} ({session_id}), {counts}."
                + (f" Kayitlar: {', '.join(ids)}." if ids else ""))

    def close_task(self, task, note=None, project=None):
        """Acik bir isi kapatir: kisa kimlik (orn. I-0007) ya da not yolu verilir."""
        project = self._require_writable(project)
        rel = self._find_note(project, task)
        raw = self._read_note(project, rel)
        meta = projects.note_meta(raw)
        if meta["kind"] != projects.TASK_KIND:
            raise ValueError(f"{task} bir is kaydi degil (tur: {meta['kind']}).")
        if meta["status"] == "done":
            return f"{rel} zaten kapali."
        suffix = f" · {note.strip()}" if note and note.strip() else ""
        with open(projects.source_file(project, {}, rel, self.notes_dir), "a", encoding="utf-8") as f:
            f.write("\n" + projects.DONE_MARKER.format(date=f"{datetime.now(timezone.utc):%Y-%m-%d}",
                                                       note=suffix) + "\n")
        self._sync(project, only=[rel])
        return f"Is kapatildi: {rel} ({meta['record_id']})."

    # --- not dosyalari ------------------------------------------------------

    def _write_note(self, project, title, body, kind="note", supersedes=None):
        """Not dosyasini yazar; doner: (yeni yol, geri plana alinan eski yol ya da None)."""
        if kind not in projects.KINDS:
            raise ValueError(f"kind su degerlerden biri olmali: {', '.join(projects.KINDS)}")
        if not title.strip() or not body.strip():
            raise ValueError("title ve body bos olamaz")
        old = self._find_note(project, supersedes) if supersedes else None
        folder = os.path.join(self.notes_dir, project)
        os.makedirs(folder, exist_ok=True)
        filename = projects.note_filename(title)
        rel = projects.NOTES_PREFIX + filename
        rec_id = projects.record_id(self._next_record_number(project), kind)
        with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
            f.write(projects.render_note(project, title, body, supersedes=old, rec_id=rec_id, kind=kind))
        if old:
            with open(projects.source_file(project, {}, old, self.notes_dir), "a", encoding="utf-8") as f:
                f.write("\n" + projects.SUPERSEDED_MARKER.format(path=rel) + "\n")
        return rel, old

    def _note_files(self, project):
        folder = os.path.join(self.notes_dir, project)
        if not os.path.isdir(folder):
            return []
        return sorted(projects.NOTES_PREFIX + n for n in os.listdir(folder) if n.endswith(".md"))

    def _read_note(self, project, rel):
        with open(projects.source_file(project, {}, rel, self.notes_dir), encoding="utf-8") as f:
            return f.read(NOTE_HEAD_CHARS)

    def _next_record_number(self, project):
        """Projedeki en buyuk kayit numarasinin bir fazlasi; kimlikler proje icinde tekildir."""
        numbers = [int(meta["record_id"][2:]) for meta in
                   (projects.note_meta(self._read_note(project, rel)) for rel in self._note_files(project))
                   if meta["record_id"]]
        return max(numbers, default=0) + 1

    def _find_note(self, project, reference):
        """Kisa kimlik ya da not yolu -> not yolu. Bulunamazsa anlamli hata."""
        if projects.RECORD_ID_RE.match(reference.strip()):
            wanted = reference.strip()
            for rel in self._note_files(project):
                if projects.note_meta(self._read_note(project, rel))["record_id"] == wanted:
                    return rel
            raise ValueError(f"{wanted} kimlikli kayit bu projede bulunamadi.")
        if not reference.startswith(projects.NOTES_PREFIX) or ".." in reference:
            raise ValueError(f"Kisa kimlik (orn. I-0007) ya da not yolu (notes/...) verin, gelen: {reference}")
        if not os.path.isfile(projects.source_file(project, {}, reference, self.notes_dir)):
            raise ValueError(f"Not bulunamadi: {reference}")
        return reference

    def add_document(self, name, content, project=None, overwrite=False):
        """Claude'un yazdigi tam bir belgeyi (plan, ozet, rapor) motorun not klasorune
        'notes/belgeler/' altina yazar ve indeksler. Proje reposuna dokunmaz."""
        project = self._require_writable(project)
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
