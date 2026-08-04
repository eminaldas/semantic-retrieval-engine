import json

import step2_extractor as extractor


def make_result(entity_confidences, relation_confidences):
    entities = [
        extractor.ExtractedEntity(name=f"E{i}", type="SERVICE", confidence=c)
        for i, c in enumerate(entity_confidences)
    ]
    ref = extractor.EntityRef(name="E0", type="SERVICE")
    relations = [
        extractor.ExtractedRelation(
            source=ref, relation_type="USES", target=ref, confidence=c
        )
        for c in relation_confidences
    ]
    return extractor.ExtractionResult(entities=entities, relations=relations)


class TestFilterByConfidence:
    def test_drops_entities_and_relations_below_threshold(self):
        result = make_result([0.9, 0.5, 0.55], [0.6, 0.54])
        filtered = extractor.filter_by_confidence(result)
        assert [e.confidence for e in filtered.entities] == [0.9, 0.55]
        assert [r.confidence for r in filtered.relations] == [0.6]

    def test_keeps_everything_above_threshold(self):
        result = make_result([0.8, 0.9], [0.7])
        filtered = extractor.filter_by_confidence(result)
        assert len(filtered.entities) == 2
        assert len(filtered.relations) == 1

    def test_empty_input_stays_empty(self):
        result = make_result([], [])
        filtered = extractor.filter_by_confidence(result)
        assert filtered.entities == []
        assert filtered.relations == []


class TestLoadPromptTemplate:
    def test_extracts_fenced_block_and_keeps_placeholder(self, tmp_path):
        md = tmp_path / "prompt.md"
        md.write_text(
            "# Baslik\n\nbir aciklama\n\n```\nSystem prompt metni\n\"{CHUNK_TEXT}\"\n```\n",
            encoding="utf-8",
        )
        template = extractor.load_prompt_template(str(md))
        assert template == 'System prompt metni\n"{CHUNK_TEXT}"'


class TestLoadChunks:
    def test_reads_jsonl_lines_and_skips_blank_lines(self, tmp_path):
        f = tmp_path / "chunks.jsonl"
        f.write_text(
            '{"chunk_id": "ch_0001", "text": "a"}\n\n{"chunk_id": "ch_0002", "text": "b"}\n',
            encoding="utf-8",
        )
        chunks = list(extractor.load_chunks(str(f)))
        assert [c["chunk_id"] for c in chunks] == ["ch_0001", "ch_0002"]


class TestAlreadyExtractedChunkIds:
    def test_returns_empty_set_when_file_missing(self, tmp_path):
        missing = tmp_path / "extraction.jsonl"
        assert extractor.already_extracted_chunk_ids(str(missing)) == set()

    def test_returns_chunk_ids_from_existing_file(self, tmp_path):
        f = tmp_path / "extraction.jsonl"
        f.write_text(
            '{"chunk_id": "a_ch0001", "entities": [], "relations": []}\n'
            '{"chunk_id": "a_ch0002", "entities": [], "relations": []}\n',
            encoding="utf-8",
        )
        assert extractor.already_extracted_chunk_ids(str(f)) == {"a_ch0001", "a_ch0002"}


class TestSaveExtraction:
    def test_writes_schema_matching_artifact_contract(self, tmp_path):
        result = make_result([0.9], [0.8])
        out_file = tmp_path / "extraction.jsonl"

        with open(out_file, "w", encoding="utf-8") as f:
            extractor.save_extraction("ch_0001", result, f)

        line = out_file.read_text(encoding="utf-8").strip()
        data = json.loads(line)
        assert data["chunk_id"] == "ch_0001"
        assert data["entities"][0]["name"] == "E0"
        assert data["relations"][0]["relation_type"] == "USES"
