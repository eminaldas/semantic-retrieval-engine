"""Ozet katmani (plan #9, RAPTOR fikrinin klasor agaci hali): "projede neler denendi", "hat/
klasorunun sonucu ne" gibi genel sorularin cevabi tek bir parcada yok; parcalar ayrintiya
bogulur. Her belge, her klasor ve proje icin kisa bir ozet uretilir ve parcalarla AYNI indekse
doc_type="summary" olarak konur (RAPTOR'daki "collapsed tree": arama hangi seviyenin uydugunu
kendisi secer). Ayrintili sorularda reranker yine asil parcayi one alir.

RAPTOR parcalari anlamsal kumelere ayirip ozetler; bizim belgeler zaten klasorlere dizili ve
klasor adlari Claude'un da kullandigi adres (path_prefix), bu yuzden agac klasor agacidir:
  belge ozeti   <- belge metni
  klasor ozeti  <- altindaki belge ve klasor ozetleri (en az MIN_FOLDER_CHILDREN cocuk)
  proje ozeti   <- ust seviye klasor ozetleri + koktaki belge ozetleri

Ozetler girdilerinin hash'iyle diskte saklanir; degismeyen belge ya da klasor yeniden ozetlenmez.

    python -m backend.knowledge.summaries build factprobe [--dry-run]"""

import hashlib
import json
import os
import posixpath
import sys
from datetime import datetime, timezone

from backend.knowledge import projects, store

DEFAULT_MODEL = "gemini-flash-lite-latest"
CALL_SITE = "knowledge_summaries"
PREFIX = "_ozet/"
DOC_TYPE = "summary"
MAX_DOC_CHARS = 400_000
MIN_FOLDER_CHILDREN = 2
SUMMARY_WORDS = {"doc": 90, "folder": 150, "project": 180}

PROMPTS = {
    "doc": """Asagidaki proje belgesini {words} kelimeyi gecmeyen Turkce bir paragrafla ozetle.
Belgenin amacini, ana sonuclarini (sayilar, kararlar, secilen ve birakilan secenekler) ve tarihini
yaz. Belgede olmayan bir sey ekleme. "Bu belge" diye baslama; dogrudan icerikle basla.

<belge yol="{path}">
{text}
</belge>""",
    "folder": """Asagida bir proje klasorundeki belgelerin ve alt klasorlerin ozetleri var. Klasorun
ne icin oldugunu ve vardigi genel sonucu {words} kelimeyi gecmeyen Turkce bir paragrafla yaz:
hangi calismalar yapildi, en onemli sonuclar ve kararlar neler, en guncel durum ne. Ozetlerde
olmayan bir sey ekleme. Onemli bulgunun hangi belgede oldugunu dosya adiyla belirt.

Klasor: {path}

{text}""",
    "project": """Asagida bir projenin klasor ve belge ozetleri var. Projenin ne oldugunu, ana
bilesenlerini, en onemli kararlari ve sonuclari ve guncel durumunu {words} kelimeyi gecmeyen Turkce
bir paragrafla yaz. Ozetlerde olmayan bir sey ekleme.

Proje: {path}

{text}""",
}


def _hash(*parts):
    return hashlib.sha256("\n\x00".join(parts).encode("utf-8")).hexdigest()[:24]


def folder_of(path):
    folder = posixpath.dirname(path.replace("\\", "/"))
    return folder + "/" if folder else ""


def build_tree(doc_paths, min_children=MIN_FOLDER_CHILDREN):
    """Doner: {klasor ("" = proje koku): [cocuk dugumler]}; cocuk ya belge yolu ya da "alt/" klasoru.
    Tek cocuklu klasorler ozetlenmez, cocugu bir ust klasore baglanir (gereksiz ozet katmani olmasin)."""
    children = {"": []}
    for path in sorted(doc_paths):
        node, folder = path, folder_of(path)
        while True:
            children.setdefault(folder, [])
            if node not in children[folder]:
                children[folder].append(node)
            if folder == "":
                break
            node, folder = folder, folder_of(folder.rstrip("/"))

    def collapse(folder):
        out = []
        for child in children.get(folder, []):
            if child.endswith("/") and len(children.get(child, [])) < min_children:
                out.extend(collapse(child))  # tek cocuklu klasorun cocugu yukari tasinir
            else:
                out.append(child)
        return out

    tree = {}
    for folder in children:
        if folder == "" or len(children[folder]) >= min_children:
            tree[folder] = collapse(folder)
    return tree


def order_bottom_up(tree):
    """Alt klasorler ust klasorlerden once; proje koku ("") en sonda."""
    return sorted(tree, key=lambda folder: (-folder.count("/"), folder))


def summary_path(node):
    """Indeksteki yol: _ozet/<belge>, _ozet/<klasor>/ ya da proje icin _ozet/."""
    return PREFIX + node


def node_label(node, project):
    if node == "":
        return f"Proje ozeti: {project}"
    return f"Klasor ozeti: {node}" if node.endswith("/") else f"Belge ozeti: {node}"


def children_text(children, summaries):
    return "\n\n".join(f"[{child}]\n{summaries[child]}" for child in children if summaries.get(child))


def build_records(project, summaries, indexed_at=None):
    """{dugum: ozet} -> indeks kayitlari (dugum basina tek parca)."""
    indexed = indexed_at or datetime.now(timezone.utc).isoformat()
    records = []
    for node, text in sorted(summaries.items()):
        if not text:
            continue
        path = summary_path(node)
        source_id = projects.source_id_for(project, path)
        label = node_label(node, project)
        records.append({
            "chunk_id": f"{source_id}#0001", "project": project, "source_id": source_id, "path": path,
            "location": "ozet", "section": label, "text": text,
            "embed_text": f"Proje: {project} | {label}\n{text}", "chunk_context": None,
            "doc_type": DOC_TYPE, "modified_at": None, "indexed_at": indexed,
            "dirs": projects.dir_prefixes(node if node else ""), "superseded_by": None,
            "kind": None, "status": None, "record_id": None,
        })
    return records


class Summarizer:
    """summarize(kind, path, text) -> ozet; sonuclar girdi hash'iyle diskte saklanir."""

    def __init__(self, project, model=DEFAULT_MODEL, client=None, manifest_dir=projects.MANIFEST_DIR):
        self.project, self.model, self._client = project, model, client
        self.path = os.path.join(manifest_dir, f"{project}_summaries.json")
        self.cache = json.load(open(self.path, encoding="utf-8")) if os.path.exists(self.path) else {}
        self.calls = 0

    def _gemini(self):
        if self._client is None:
            from backend.knowledge.eval import _gemini_client

            self._client = _gemini_client()
        return self._client

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=0, sort_keys=True)

    def summarize(self, kind, path, text):
        key = _hash(kind, path, text, self.model)
        cached = self.cache.get(path)
        if cached and cached["hash"] == key:
            return cached["summary"]
        from google.genai import types

        from backend.shared import gemini_retry, gemini_usage

        prompt = PROMPTS[kind].format(path=path or self.project, text=text[:MAX_DOC_CHARS],
                                      words=SUMMARY_WORDS[kind])

        def _call():
            response = self._gemini().models.generate_content(
                model=self.model, contents=prompt, config=types.GenerateContentConfig(temperature=0.0))
            gemini_usage.log_usage(CALL_SITE, self.model, response)
            return response

        summary = " ".join((gemini_retry.call_with_retry(_call, pacing_key=CALL_SITE).text or "").split())
        self.calls += 1
        self.cache[path] = {"hash": key, "summary": summary}
        return summary


def summarize_project(docs, summarizer, log=print):
    """docs: {yol: tam metin}. Doner: {dugum: ozet} (belgeler, klasorler, "" = proje)."""
    summaries = {}
    for path in sorted(docs):
        summaries[path] = summarizer.summarize("doc", path, docs[path])
    tree = build_tree(docs)
    for folder in order_bottom_up(tree):
        kind = "project" if folder == "" else "folder"
        summaries[folder] = summarizer.summarize(kind, folder, children_text(tree[folder], summaries))
        log(f"  {node_label(folder, summarizer.project)} ({len(tree[folder])} cocuk)")
    # Artik olmayan belge/klasorlerin ozetleri onbellekte birikmesin
    summarizer.cache = {k: v for k, v in summarizer.cache.items() if k in summaries}
    summarizer.save()
    return summaries


def build(project, client, embedder, summarizer, log=print):
    """Ozetleri uretip indekse yazar; projenin eski ozet kayitlari once silinir. Doner: kayit sayisi."""
    from backend.knowledge import goldgen

    docs = {p: t for p, t in goldgen.load_documents(client, project).items() if not p.startswith(PREFIX)}
    summaries = summarize_project(docs, summarizer, log=log)
    records = build_records(project, summaries)
    remove(client, project)
    vectors = embedder.encode([r["embed_text"] for r in records])
    return store.upsert_records(client, records, vectors)


def remove(client, project):
    from qdrant_client import models

    client.delete(store.COLLECTION_NAME, points_selector=models.FilterSelector(
        filter=store.project_filter(project=project, doc_type=DOC_TYPE)))


def main():
    import argparse

    from backend.knowledge.runtime import Runtime
    from backend.knowledge.service import KnowledgeService

    parser = argparse.ArgumentParser(description="Belge/klasor/proje ozetleri")
    parser.add_argument("command", choices=("build", "remove"))
    parser.add_argument("project")
    args = parser.parse_args()
    runtime = Runtime()
    project = KnowledgeService(runtime).resolve_project(args.project)
    client = runtime.client()
    if args.command == "remove":
        remove(client, project)
        print(f"{project}: ozet kayitlari silindi")
        return 0
    summarizer = Summarizer(project)
    total = build(project, client, runtime.embedder(), summarizer)
    print(f"{project}: {total} ozet indekslendi ({summarizer.calls} Gemini cagrisi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
