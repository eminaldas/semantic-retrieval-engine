from backend.knowledge import eval as ev


def _r(path, text="", weak=False):
    return {"path": path, "text": text, "low_confidence": weak}


def test_hit_needs_path_and_text():
    record = {"expected_paths": ["hat/SON_HAL_1021.md"], "expected_text": "396 geçti"}
    assert ev.is_hit(record, _r("hat\\SON_HAL_1021.md", "Test: 396  geçti, 5 atlandı"))
    assert not ev.is_hit(record, _r("hat/SON_HAL_1021.md", "baska parca"))
    assert not ev.is_hit(record, _r("KARARLAR_OZET.md", "396 geçti"))


def test_hit_accepts_any_of_listed_texts():
    record = {"expected_paths": ["A.md"], "expected_text": ["birinci yazim", "ikinci yazim"]}
    assert ev.is_hit(record, _r("A.md", "... ikinci yazim ..."))
    assert not ev.is_hit(record, _r("A.md", "ucuncu"))


def test_hit_turkish_case_insensitive():
    record = {"expected_paths": ["A.md"], "expected_text": "ŞİMDİLİK YOK"}
    assert ev.is_hit(record, _r("a.md", "şimdilik yok"))


def test_score_rank_and_abstain():
    record = {"id": "x", "type": "karar", "expected_paths": ["A.md"], "expected_text": None}
    score = ev.score_record(record, [_r("B.md"), _r("A.md"), _r("A.md")])
    assert score["rank"] == 2 and not score["abstained"]

    unanswerable = {"id": "y", "type": "cevapsiz", "expected_paths": [], "expected_text": None}
    assert ev.score_record(unanswerable, [_r("A.md", weak=True)])["abstained"]
    assert not ev.score_record(unanswerable, [_r("A.md", weak=False)])["abstained"]


def test_aggregate():
    scores = [
        {"answerable": True, "type": "karar", "rank": 1, "abstained": False},
        {"answerable": True, "type": "karar", "rank": 4, "abstained": False},
        {"answerable": True, "type": "sayi", "rank": None, "abstained": True},
        {"answerable": False, "type": "cevapsiz", "rank": None, "abstained": True},
    ]
    summary = ev.aggregate(scores)
    assert summary["hit@1"] == 1 / 3
    assert summary["hit@3"] == 1 / 3
    assert summary["hit@5"] == 2 / 3
    assert abs(summary["mrr"] - (1 + 0.25) / 3) < 1e-9
    assert summary["false_abstain"] == 1 / 3
    assert summary["abstain_accuracy"] == 1.0
    assert summary["hit@3_by_type"] == {"karar": 0.5, "sayi": 0.0}


def test_decimal_comma_matches_dot():
    record = {"expected_paths": ["A.md"], "expected_text": "eşik 0.91"}
    assert ev.is_hit(record, _r("A.md", "e5 eşik 0,91 olarak sabitlendi"))


def test_multihop_counts_each_hop():
    record = {"id": "m", "type": "cok_adim", "expected_paths": ["A.md", "B.md"], "expected_text": None,
              "hops": [{"paths": ["A.md"]}, {"paths": ["B.md"], "text": "ikinci"}]}
    both = ev.score_record(record, [_r("A.md"), _r("B.md", "ikinci adim")])
    one = ev.score_record(record, [_r("A.md"), _r("B.md", "baska")])
    assert (both["hops_found"], one["hops_found"]) == (2, 1)
    summary = ev.aggregate([both, one])
    assert summary["multihop_recall"] == 0.5


def test_origin_split_only_when_mixed():
    human = {"answerable": True, "type": "karar", "rank": 1, "abstained": False, "origin": "human"}
    synth = {"answerable": True, "type": "karar", "rank": None, "abstained": False, "origin": "synthetic"}
    assert "hit@3_by_origin" not in ev.aggregate([human])
    assert ev.aggregate([human, synth])["hit@3_by_origin"] == {"human": 1.0, "synthetic": 0.0}
