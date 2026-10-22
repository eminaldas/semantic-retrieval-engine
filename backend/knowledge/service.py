"""MCP araclarinin arkasindaki is mantigi. Sunucudan (protokol) ayri tutuldu ki
araclar Qdrant/model olmadan, sahte bir runtime ile test edilebilsin."""

import os

from backend.knowledge import projects, search, store, sync


class KnowledgeService:
    def __init__(self, runtime, config_loader=projects.load_config, notes_dir=projects.NOTES_DIR):
        self.runtime = runtime
        self.config_loader = config_loader
        self.notes_dir = notes_dir

    def _config(self):
        return self.config_loader()

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

    def list_projects(self):
        config = {k: v for k, v in self._config().items() if not k.startswith("_")}
        if not config:
            return "Tanimli proje yok. knowledge_projects.json dosyasina bir proje ekleyin."
        lines = [f"- {name}: {settings.get('description', '')} ({settings.get('root')})".rstrip() for name, settings in sorted(config.items())]
        return "Tanimli projeler:\n" + "\n".join(lines)

    def search(self, query, project=None, doc_type=None, top_k=5):
        if not query or not query.strip():
            raise ValueError("query bos olamaz")
        top_k = max(1, min(int(top_k), 15))
        results = search.search(
            self.runtime.client(), self.runtime.embedder(), self.runtime.reranker(), query,
            project=self.resolve_project(project), doc_type=doc_type, top_k=top_k,
        )
        return search.format_results(results)

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
        project = self.resolve_project(project)
        if not project:
            raise ValueError("Birden fazla proje var; project parametresini verin.")
        rows = store.scroll_all(self.runtime.client(), store.project_filter(source_id=projects.source_id_for(project, path)),
                                ("chunk_id", "location", "text"))
        if not rows:
            return f"Belge bulunamadi: {path}. list_documents ile yollari kontrol edin."
        rows.sort(key=lambda r: r["chunk_id"])
        text = "\n\n".join(f"({r['location']})\n{r['text'].strip()}" for r in rows)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... {len(text) - max_chars} karakter kesildi; search ile ilgili bolumu arayin]"
        return f"{path}\n\n{text}"

    def sync_project(self, project=None, full=False):
        project = self.resolve_project(project)
        if not project:
            raise ValueError("Birden fazla proje var; project parametresini verin.")
        report = sync.sync_project(project, self.runtime.client(), self.runtime.embedder(), config=self._config(),
                                   full=full, log=lambda *_: None, chunk_model=self.runtime.chunker_model())
        return (f"{project}: {len(report['indexed'])} dosya indekslendi ({report['chunks']} parca), "
                f"{len(report['removed'])} dosya kaldirildi.")

    def add_note(self, title, body, project=None):
        project = self.resolve_project(project)
        if not project:
            raise ValueError("Birden fazla proje var; project parametresini verin.")
        if not title.strip() or not body.strip():
            raise ValueError("title ve body bos olamaz")
        folder = os.path.join(self.notes_dir, project)
        os.makedirs(folder, exist_ok=True)
        filename = projects.note_filename(title)
        with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
            f.write(projects.render_note(project, title, body))
        rel = projects.NOTES_PREFIX + filename
        sync.sync_project(project, self.runtime.client(), self.runtime.embedder(), config=self._config(),
                          only=[rel], log=lambda *_: None, chunk_model=self.runtime.chunker_model())
        return f"Not kaydedildi ve indekslendi: {rel}"
