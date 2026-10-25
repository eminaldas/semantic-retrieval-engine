from backend.knowledge import contextual, projects
from backend.knowledge.contextual import ChunkContext, ContextBatch, GeminiContextualizer


class FakeContextualizer(GeminiContextualizer):
    """Gemini yerine: her parcaya numarasini yazar; 'fail' verilirse hata firlatir."""

    def __init__(self, tmp_path, fail=False):
        super().__init__(manifest_dir=str(tmp_path), log=lambda m: None)
        self.prompts = []
        self.fail = fail

    def _call(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("kota doldu")
        count = prompt.count("\n[") + prompt.startswith("[")
        return ContextBatch(contexts=[ChunkContext(i=i, c=f"  parca  {i} ") for i in range(1, count + 1)])


def test_prompt_numbers_chunks_and_carries_document():
    prompt = contextual.build_prompt("a.md", "BELGE", ["bir", "iki"])
    assert "BELGE" in prompt and "[1]\nbir" in prompt and "[2]\niki" in prompt


def test_parse_batch_orders_and_ignores_bad_numbers():
    batch = ContextBatch(contexts=[ChunkContext(i=2, c="ikinci"), ChunkContext(i=9, c="fazla"), ChunkContext(i=1, c="bir")])
    assert contextual.parse_batch(batch, 3) == ["bir", "ikinci", ""]


def test_one_call_per_document_and_cache_skips_known_chunks(tmp_path):
    ctx = FakeContextualizer(tmp_path)
    assert ctx.describe("p", "a.md", "belge", ["x", "y"]) == ["parca 1", "parca 2"]
    assert len(ctx.prompts) == 1
    # yeni bir surec ayni onbellegi diskten okur; sadece yeni parca gonderilir
    again = FakeContextualizer(tmp_path)
    assert again.describe("p", "a.md", "belge", ["x", "z"]) == ["parca 1", "parca 1"]
    assert len(again.prompts) == 1 and "[1]\nz" in again.prompts[0] and "\nx\n" not in again.prompts[0]


def test_large_documents_are_split_into_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(contextual, "MAX_CHUNKS_PER_CALL", 2)
    ctx = FakeContextualizer(tmp_path)
    assert all(ctx.describe("p", "a.md", "belge", ["a", "b", "c"]))
    assert len(ctx.prompts) == 2


def test_failure_falls_back_to_no_context(tmp_path):
    assert FakeContextualizer(tmp_path, fail=True).describe("p", "a.md", "belge", ["x"]) == [""]


def test_context_goes_into_embedding_text_not_into_text():
    [rec] = projects.build_chunk_records("p", "a.md", "# A\nK4.24 kaldirildi.", [("K4.24 kaldirildi.", "satır 2-2")],
                                         contexts=["Esik kararlari bolumunde verdict esiginin kaldirilmasi."])
    assert "Baglam: Esik kararlari" in rec["embed_text"] and rec["text"] == "K4.24 kaldirildi."
    assert rec["chunk_context"].startswith("Esik")
