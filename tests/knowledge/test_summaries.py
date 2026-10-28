from backend.knowledge import summaries as sm

DOCS = ["MIMARI.md", "KARARLAR.md", "hat/PLAN.md", "hat/RAPOR.md", "hat/demo/A.md", "hat/demo/B.md",
        "hat/k59/RAPOR.md", "graf/README.md"]


def test_tree_groups_by_folder_and_lifts_single_child_folders():
    tree = sm.build_tree(DOCS)
    assert set(tree) == {"", "hat/", "hat/demo/"}
    assert tree["hat/demo/"] == ["hat/demo/A.md", "hat/demo/B.md"]
    # hat/k59/ tek belgeli: belgesi hat/ altina tasinir; graf/ tek belgeli: koke tasinir
    assert "hat/k59/RAPOR.md" in tree["hat/"] and "hat/demo/" in tree["hat/"]
    assert "graf/README.md" in tree[""] and "hat/" in tree[""] and "MIMARI.md" in tree[""]


def test_bottom_up_order_ends_with_project():
    assert sm.order_bottom_up(sm.build_tree(DOCS)) == ["hat/demo/", "hat/", ""]


def test_records_have_summary_type_paths_and_dirs():
    records = sm.build_records("p", {"": "proje", "hat/": "klasor", "hat/PLAN.md": "belge", "bos/": ""})
    by_path = {r["path"]: r for r in records}
    assert set(by_path) == {"_ozet/", "_ozet/hat/", "_ozet/hat/PLAN.md"}
    assert all(r["doc_type"] == "summary" and r["modified_at"] is None for r in records)
    assert by_path["_ozet/hat/"]["dirs"] == ["hat/"] and by_path["_ozet/hat/PLAN.md"]["dirs"] == ["hat/"]
    assert by_path["_ozet/"]["section"] == "Proje ozeti: p"
    assert by_path["_ozet/hat/"]["chunk_id"].endswith("#0001")


class FakeGemini:
    def __init__(self):
        self.prompts = []
        outer = self

        class Models:
            def generate_content(self, model, contents, config):
                outer.prompts.append(contents)

                class R:
                    text = f"ozet {len(outer.prompts)}"
                    usage_metadata = None
                return R()

        self.models = Models()


def test_summarize_project_caches_unchanged_nodes(tmp_path, monkeypatch):
    from backend.shared import gemini_usage

    monkeypatch.setattr(gemini_usage, "log_usage", lambda *a, **k: None)
    docs = {"hat/A.md": "a", "hat/B.md": "b", "K.md": "k"}
    gemini = FakeGemini()
    first = sm.summarize_project(docs, sm.Summarizer("p", client=gemini, manifest_dir=str(tmp_path)), log=lambda m: None)
    assert set(first) == {"hat/A.md", "hat/B.md", "K.md", "hat/", ""} and len(gemini.prompts) == 5
    assert "[hat/A.md]" in gemini.prompts[3]  # klasor ozeti cocuk ozetlerinden

    docs["hat/B.md"] = "b degisti"
    again = sm.Summarizer("p", client=gemini, manifest_dir=str(tmp_path))
    sm.summarize_project(docs, again, log=lambda m: None)
    assert again.calls == 3  # B, hat/ ve proje yeniden; A ve K onbellekten


def test_build_indexes_summaries_next_to_chunks_and_remove_drops_only_them(tmp_path, monkeypatch):
    """Uctan uca: bellek ici Qdrant, sahte embedder/reranker/Gemini."""
    import hashlib
    import re

    import numpy as np
    from qdrant_client import QdrantClient

    from backend.knowledge import projects, search, store
    from backend.shared import gemini_usage

    monkeypatch.setattr(gemini_usage, "log_usage", lambda *a, **k: None)

    def words(text):
        return set(re.findall(r"\w+", text.lower()))

    class Embedder:
        def encode(self, texts):
            out = []
            for text in ([texts] if isinstance(texts, str) else texts):
                v = np.zeros(32)
                for w in words(text):
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1
                out.append(v / (np.linalg.norm(v) or 1))
            return out[0] if isinstance(texts, str) else np.array(out)

    class Reranker:
        def predict(self, pairs):
            return [len(words(q) & words(t)) / len(words(q)) for q, t in pairs]

    client, embedder = QdrantClient(":memory:"), Embedder()
    store.ensure_collection(client, 32)
    for rel, text in {"K.md": "kok belge", "hat/A.md": "nicemleme hiz", "hat/B.md": "esik secimi"}.items():
        records = projects.build_chunk_records("p", rel, text, [(text, "satir 1-1")], mtime=1.7e9)
        store.upsert_records(client, records, embedder.encode([r["embed_text"] for r in records]))

    summarizer = sm.Summarizer("p", client=FakeGemini(), manifest_dir=str(tmp_path))
    assert sm.build("p", client, embedder, summarizer, log=lambda m: None) == 5

    def paths(**kw):
        return {r["path"] for r in search.search(client, embedder, Reranker(), "ozet hiz", project="p", top_k=10, **kw)}

    assert {"_ozet/", "_ozet/hat/", "_ozet/hat/A.md", "hat/A.md"} <= paths()
    assert not any(p.startswith("_ozet/") for p in paths(doc_type="doc"))
    assert "_ozet/hat/" in paths(path_prefix="hat/") and "_ozet/" not in paths(path_prefix="hat/")

    sm.remove(client, "p")
    assert paths() == {"K.md", "hat/A.md", "hat/B.md"}
