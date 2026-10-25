from backend.knowledge import goldgen

DOCS = {
    "KARARLAR_OZET.md": "D-12: e5 benzerlik esik 0,91 olarak sabitlendi.",
    "hat/HAT_PLANI.md": "Esik 0,91; kalibrasyon SONUC_50 setinde yapilacak.",
    "graf/README.md": "Benzerlik bandi 0,85-0,93 araliginda.",
}


def test_audit_finds_text_in_unlisted_docs_and_missing_text():
    records = [
        {"id": "s02", "question": "e5 esigi?", "expected_paths": ["KARARLAR_OZET.md"], "expected_text": "esik 0.91"},
        {"id": "x", "question": "?", "expected_paths": ["graf/README.md"], "expected_text": "olmayan ifade"},
        {"id": "ok", "question": "?", "expected_paths": ["graf/README.md"], "expected_text": "bandi"},
        {"id": "n", "question": "?", "expected_paths": [], "expected_text": None},
    ]
    findings = {f["id"]: f for f in goldgen.audit(records, DOCS)}
    assert set(findings) == {"s02", "x"}
    assert findings["x"]["text_not_found"] == ["olmayan ifade"]
    assert findings["s02"]["also_in"] == ["hat/HAT_PLANI.md"]  # virgullu yazim ve buyuk harf de eslesir


def test_record_expands_paths_to_every_doc_with_the_quote():
    candidate = {"type": "sayi", "question": "Esik kac?", "answer": "0,91",
                 "quotes": [{"path": "hat/HAT_PLANI.md", "text": "Esik 0,91"}]}
    record = goldgen.to_record(candidate, DOCS, 1)
    assert record["expected_paths"] == ["KARARLAR_OZET.md", "hat/HAT_PLANI.md"]
    assert record["origin"] == "synthetic" and record["reviewed"] is None


def test_record_rejects_invented_quote():
    candidate = {"type": "neden", "question": "?", "answer": "a",
                 "quotes": [{"path": "graf/README.md", "text": "bu cumle belgede yok"}]}
    assert goldgen.to_record(candidate, DOCS, 1) is None


def test_multihop_needs_two_docs():
    two = {"type": "cok_adim", "question": "?", "answer": "a",
           "quotes": [{"path": "KARARLAR_OZET.md", "text": "D-12: e5"}, {"path": "graf/README.md", "text": "0,85-0,93"}]}
    one = {"type": "cok_adim", "question": "?", "answer": "a", "quotes": [{"path": "KARARLAR_OZET.md", "text": "D-12: e5"}]}
    record = goldgen.to_record(two, DOCS, 1)
    assert len(record["hops"]) == 2 and record["hops"][0]["paths"] == ["KARARLAR_OZET.md"]
    assert goldgen.to_record(one, DOCS, 2) is None


def test_near_miss_becomes_unanswerable():
    record = goldgen.to_record({"type": "yakin_cevapsiz", "question": "?", "answer": "YOK", "quotes": []}, DOCS, 3)
    assert record["expected_paths"] == [] and record["type"] == "cevapsiz"


def test_pick_batches_pairs_from_same_folder():
    docs = {f"hat/{i}.md": "x" * 500 for i in range(3)} | {"tek.md": "x" * 500, "kisa.md": "x"}
    batches = goldgen.pick_batches(docs, n_docs=2, n_pairs=5)
    singles = [b for b in batches if len(b) == 1]
    pairs = [b for b in batches if len(b) == 2]
    assert len(singles) == 2 and all(b[0] != "kisa.md" for b in singles)
    assert pairs and all(p[0].startswith("hat/") and p[1].startswith("hat/") for p in pairs)
