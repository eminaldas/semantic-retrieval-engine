"""Bir projenin belgelerini artimli olarak indeksler: sadece hash'i degisen ya da
yeni dosyalar yeniden parcalanir, silinen dosyalarin parcalari Qdrant'tan kaldirilir.
Graf cikarimi (Gemini) burada yok - hiz ve maliyet icin sadece vektor + BM25.

Kullanim: python -m backend.knowledge.sync <proje> [--full]"""

import os
import sys
from datetime import datetime, timezone

from backend.knowledge import projects, store


def collect_files(project, settings, notes_dir=projects.NOTES_DIR):
    """{relpath: abs_path} - proje kokundeki belgeler + motorun not klasoru (notes/ onekiyle)."""
    files = {}
    root = settings["root"]
    for rel in projects.iter_project_files(root, settings["include"], settings["exclude"]):
        files[rel] = os.path.join(root, rel)
    project_notes = os.path.join(notes_dir, project)
    if os.path.isdir(project_notes):
        for rel in projects.iter_project_files(project_notes, ["**/*.md"], []):
            files[projects.NOTES_PREFIX + rel] = os.path.join(project_notes, rel)
    return files


def sync_project(project, client, embedding_model, config=None, full=False, only=None, log=print, chunk_model=None,
                 notes_dir=projects.NOTES_DIR, contextualizer=None):
    """only: verilirse sadece bu relpath'ler kontrol edilir (orn. yeni eklenen bir not).
    chunk_model: parca sinirlarini belirleyen model (verilmezse embedding_model).
    contextualizer: projede "contextual": true ise parca basina baglam cumlesi uretir (contextual.py).
    Doner: {'indexed': [...], 'removed': [...], 'chunks': int}."""
    from backend.pipeline import chunker  # agir import (nltk, pypdf) - sadece gerektiginde

    settings = projects.project_settings(config if config is not None else projects.load_config(), project)
    store.ensure_collection(client, embedding_model.get_embedding_dimension())

    files = collect_files(project, settings, notes_dir)
    if only is not None:
        files = {rel: path for rel, path in files.items() if rel in set(only)}
    old_manifest = projects.load_manifest(project)
    new_hashes = {rel: projects.file_hash(path) for rel, path in files.items()}

    if only is not None:
        changed = [rel for rel in sorted(new_hashes) if full or old_manifest.get(rel) != new_hashes[rel]]
        removed = []
    elif full:
        changed, removed = sorted(new_hashes), sorted(p for p in old_manifest if p not in new_hashes)
    else:
        changed, removed = projects.diff_manifest(old_manifest, new_hashes)

    manifest = dict(old_manifest)
    total_chunks = 0
    indexed_at = datetime.now(timezone.utc).isoformat()
    for rel in changed:
        path = files[rel]
        raw_text = chunker.load_document(path)
        chunks = projects.split_oversized(chunker.chunk_document(path, chunk_model or embedding_model)) if raw_text.strip() else []
        contexts = None
        if contextualizer is not None and settings.get("contextual") and chunks:
            contexts = contextualizer.describe(project, rel, raw_text, [text for text, _ in chunks])
        records = projects.build_chunk_records(project, rel, raw_text, chunks, mtime=os.path.getmtime(path),
                                               indexed_at=indexed_at, contexts=contexts)
        store.delete_source(client, projects.source_id_for(project, rel))
        if records:
            vectors = embedding_model.encode([r["embed_text"] for r in records])
            total_chunks += store.upsert_records(client, records, vectors)
        manifest[rel] = new_hashes[rel]
        log(f"  indekslendi: {rel} ({len(records)} parca)")
    for rel in removed:
        store.delete_source(client, projects.source_id_for(project, rel))
        manifest.pop(rel, None)
        log(f"  kaldirildi: {rel}")

    projects.save_manifest(project, manifest)
    return {"indexed": changed, "removed": removed, "chunks": total_chunks}


if __name__ == "__main__":
    from backend.knowledge.runtime import Runtime

    if len(sys.argv) < 2:
        print("Kullanim: python -m backend.knowledge.sync <proje> [--full]")
        sys.exit(1)
    runtime = Runtime()
    report = sync_project(sys.argv[1], runtime.client(), runtime.embedder(), full="--full" in sys.argv,
                          chunk_model=runtime.chunker_model(), contextualizer=runtime.contextualizer(), log=print)
    print(f"\nBASARILI! {len(report['indexed'])} dosya indekslendi ({report['chunks']} parca), "
          f"{len(report['removed'])} dosya kaldirildi.")
