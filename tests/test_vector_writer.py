import step4_vector_writer as vw


class TestChunkPointId:
    def test_same_chunk_id_produces_same_point_id(self):
        assert vw.chunk_point_id("ch_0001") == vw.chunk_point_id("ch_0001")

    def test_different_chunk_ids_produce_different_point_ids(self):
        assert vw.chunk_point_id("ch_0001") != vw.chunk_point_id("ch_0002")

    def test_produces_a_valid_uuid_string(self):
        import uuid

        result = vw.chunk_point_id("ch_0001")
        # UUID(...) gecersiz bir string icin ValueError firlatir
        assert str(uuid.UUID(result)) == result


class TestLoadChunks:
    def test_reads_jsonl_lines_and_skips_blank_lines(self, tmp_path):
        f = tmp_path / "chunks.jsonl"
        f.write_text(
            '{"chunk_id": "ch_0001", "text": "a"}\n\n{"chunk_id": "ch_0002", "text": "b"}\n',
            encoding="utf-8",
        )
        chunks = list(vw.load_chunks(str(f)))
        assert [c["chunk_id"] for c in chunks] == ["ch_0001", "ch_0002"]


class TestBuildPoint:
    def test_point_id_matches_chunk_point_id_and_payload_is_complete(self):
        chunk = {
            "chunk_id": "ch_0001",
            "source_id": "doc.txt",
            "location": "chunk_index_0",
            "quote": "Merhaba...",
            "text": "Merhaba dunya.",
        }
        vector = [0.1, 0.2, 0.3]

        point = vw.build_point(chunk, VectorLike(vector))

        assert point.id == vw.chunk_point_id("ch_0001")
        assert point.vector == vector
        assert point.payload == chunk


class VectorLike:
    """model.encode(...) numpy array donduruyor; sadece .tolist() gerektiren minimal sahte."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values
