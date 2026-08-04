import eval_extraction as harness
import step2_extractor as extractor


def make_extraction_result(entities, relations):
    """entities: [(name, type), ...]; relations: [(source_name, source_type, rel_type, target_name, target_type), ...]"""
    ents = [extractor.ExtractedEntity(name=n, type=t, confidence=0.9) for n, t in entities]
    rels = [
        extractor.ExtractedRelation(
            source=extractor.EntityRef(name=sn, type=st),
            relation_type=rt,
            target=extractor.EntityRef(name=tn, type=tt),
            confidence=0.9,
        )
        for sn, st, rt, tn, tt in relations
    ]
    return extractor.ExtractionResult(entities=ents, relations=rels)


class TestLoadGoldenSet:
    def test_reads_jsonl_lines_and_skips_blank_lines(self, tmp_path):
        f = tmp_path / "golden.jsonl"
        f.write_text(
            '{"chunk_id": "ch_0001", "gold_entities": ["A::a"], "gold_relations": []}\n'
            '\n'
            '{"chunk_id": "ch_0002", "gold_entities": [], "gold_relations": [{"source": "A::a", "relation_type": "USES", "target": "B::b"}]}\n',
            encoding="utf-8",
        )
        records = harness.load_golden_set(str(f))
        assert len(records) == 2
        assert records[0]["chunk_id"] == "ch_0001"
        assert records[1]["gold_relations"] == [{"source": "A::a", "relation_type": "USES", "target": "B::b"}]


class TestNormalizeExtractionResult:
    def test_normalizes_entity_names_to_entity_ids(self):
        result = make_extraction_result(entities=[("Shared Buffers", "MECHANISM")], relations=[])
        entity_ids, _ = harness.normalize_extraction_result(result)
        assert entity_ids == {"MECHANISM::shared_buffers"}

    def test_normalizes_relations_to_source_type_target_tuples(self):
        result = make_extraction_result(
            entities=[],
            relations=[("PostgreSQL", "DATABASE", "USES", "Shared Buffers", "MECHANISM")],
        )
        _, relation_tuples = harness.normalize_extraction_result(result)
        assert relation_tuples == {("DATABASE::postgresql", "USES", "MECHANISM::shared_buffers")}

    def test_empty_result_gives_empty_sets(self):
        result = make_extraction_result(entities=[], relations=[])
        entity_ids, relation_tuples = harness.normalize_extraction_result(result)
        assert entity_ids == set()
        assert relation_tuples == set()


class TestComputeExtractionMetrics:
    def test_perfect_match_counts_all_correct(self):
        counts = harness.compute_extraction_metrics(
            gold_entities=["A::a", "B::b"],
            gold_relations=[{"source": "A::a", "relation_type": "USES", "target": "B::b"}],
            predicted_entities={"A::a", "B::b"},
            predicted_relations={("A::a", "USES", "B::b")},
        )
        assert counts == {
            "entity_correct": 2, "entity_predicted": 2, "entity_gold": 2,
            "relation_correct": 1, "relation_predicted": 1, "relation_gold": 1,
        }

    def test_partial_overlap_counts_correctly(self):
        counts = harness.compute_extraction_metrics(
            gold_entities=["A::a", "B::b", "C::c"],
            gold_relations=[{"source": "A::a", "relation_type": "USES", "target": "B::b"}],
            predicted_entities={"A::a", "D::d"},
            predicted_relations=set(),
        )
        assert counts["entity_correct"] == 1
        assert counts["entity_predicted"] == 2
        assert counts["entity_gold"] == 3
        assert counts["relation_correct"] == 0
        assert counts["relation_predicted"] == 0
        assert counts["relation_gold"] == 1

    def test_empty_gold_and_empty_predicted_gives_zero_counts(self):
        counts = harness.compute_extraction_metrics(
            gold_entities=[], gold_relations=[], predicted_entities=set(), predicted_relations=set(),
        )
        assert counts == {
            "entity_correct": 0, "entity_predicted": 0, "entity_gold": 0,
            "relation_correct": 0, "relation_predicted": 0, "relation_gold": 0,
        }

    def test_extra_predictions_not_in_gold_count_as_predicted_but_not_correct(self):
        counts = harness.compute_extraction_metrics(
            gold_entities=[], gold_relations=[],
            predicted_entities={"X::x"}, predicted_relations=set(),
        )
        assert counts["entity_correct"] == 0
        assert counts["entity_predicted"] == 1
        assert counts["entity_gold"] == 0


def make_chunk_result(entity_correct, entity_predicted, entity_gold, relation_correct, relation_predicted, relation_gold):
    return {
        "entity_correct": entity_correct, "entity_predicted": entity_predicted, "entity_gold": entity_gold,
        "relation_correct": relation_correct, "relation_predicted": relation_predicted, "relation_gold": relation_gold,
    }


class TestAggregateExtractionResults:
    def test_micro_averages_across_chunks_not_per_chunk_average(self):
        # Chunk 1: 6/6 entities gold, all found. Chunk 2: 0 gold entities, 0 predicted.
        # A naive per-chunk average of precision would be skewed by chunk 2's undefined ratio;
        # micro-averaging sums counts first (6 correct / 6 predicted / 6 gold overall).
        results = [
            make_chunk_result(6, 6, 6, 5, 5, 5),
            make_chunk_result(0, 0, 0, 0, 0, 0),
        ]
        agg = harness.aggregate_extraction_results(results)
        assert agg["entity_precision"] == 1.0
        assert agg["entity_recall"] == 1.0
        assert agg["entity_f1"] == 1.0
        assert agg["relation_precision"] == 1.0
        assert agg["relation_recall"] == 1.0

    def test_partial_precision_and_recall_computed_correctly(self):
        # Across 2 chunks: 3 correct out of 4 predicted (precision 0.75), 3 correct out of 6 gold (recall 0.5)
        results = [
            make_chunk_result(2, 3, 4, 0, 0, 0),
            make_chunk_result(1, 1, 2, 0, 0, 0),
        ]
        agg = harness.aggregate_extraction_results(results)
        assert agg["entity_precision"] == 3 / 4
        assert agg["entity_recall"] == 3 / 6
        assert agg["entity_f1"] == 2 * (3 / 4) * (3 / 6) / ((3 / 4) + (3 / 6))

    def test_zero_predicted_and_zero_gold_gives_zero_not_error(self):
        results = [make_chunk_result(0, 0, 0, 0, 0, 0)]
        agg = harness.aggregate_extraction_results(results)
        assert agg["entity_precision"] == 0.0
        assert agg["entity_recall"] == 0.0
        assert agg["entity_f1"] == 0.0

    def test_empty_chunk_results_list_returns_zeroes_without_error(self):
        agg = harness.aggregate_extraction_results([])
        assert agg == {
            "entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0,
            "relation_precision": 0.0, "relation_recall": 0.0, "relation_f1": 0.0,
        }
