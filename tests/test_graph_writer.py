import pytest

import step3_graph_writer as gw


class TestNormalizeEntityId:
    def test_lowercases_and_replaces_spaces(self):
        entity_id, normalized_name = gw.normalize_entity_id("Auth Service", "SERVICE")
        assert entity_id == "SERVICE::auth_service"
        assert normalized_name == "auth_service"

    def test_replaces_special_characters(self):
        entity_id, _ = gw.normalize_entity_id("Payment DB (primary)", "DATABASE")
        assert entity_id == "DATABASE::payment_db_primary"

    def test_collapses_repeated_separators_and_strips_edges(self):
        entity_id, _ = gw.normalize_entity_id("  I/O   Load!! ", "METRIC")
        assert entity_id == "METRIC::i_o_load"

    def test_matches_architecture_doc_examples(self):
        assert gw.normalize_entity_id("Deadlock", "FAILURE_MODE")[0] == "FAILURE_MODE::deadlock"

    def test_transliterates_turkish_characters_instead_of_dropping_them(self):
        entity_id, _ = gw.normalize_entity_id("I/O Darboğazı", "FAILURE_MODE")
        assert entity_id == "FAILURE_MODE::i_o_darbogazi"

    def test_transliterates_uppercase_turkish_characters(self):
        entity_id, _ = gw.normalize_entity_id("Ödeme Veri Tabanı", "DATABASE")
        assert entity_id == "DATABASE::odeme_veri_tabani"

    def test_distinct_turkish_words_no_longer_collide(self):
        # Fix oncesi "Yuk Artisi" ve "Islem Yuku" gibi farkli kelimeler, ozel
        # karakterler silindigi icin benzer/cakisan id'lere donusebiliyordu.
        yuk_artisi, _ = gw.normalize_entity_id("Yük Artışı", "METRIC")
        islem_yuku, _ = gw.normalize_entity_id("İşlem Yükü", "METRIC")
        assert yuk_artisi == "METRIC::yuk_artisi"
        assert islem_yuku == "METRIC::islem_yuku"
        assert yuk_artisi != islem_yuku


class FakeReadResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)


class FakeTxForResolve:
    """resolve_entity_id icin: tx.run(...) hep ayni 'mevcut entity' listesini dondurur."""

    def __init__(self, existing_entities):
        self.existing_entities = existing_entities  # [{"entity_id":..., "normalized_name":..., "aliases":[...]}]

    def run(self, query, **params):
        return FakeReadResult(self.existing_entities)


class FakeQdrantHit:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class FakeQdrantResponse:
    def __init__(self, points):
        self.points = points


class FakeQdrantClient:
    def __init__(self, hits=None):
        self._hits = hits or []
        self.upserted_points = []

    def query_points(self, collection_name, query, limit, query_filter=None):
        return FakeQdrantResponse(self._hits)

    def upsert(self, collection_name, points):
        self.upserted_points.extend(points)

    def collection_exists(self, name):
        return True


class FakeEmbeddingModel:
    def encode(self, text):
        return [0.0]  # resolve_entity_id'de gercek deger onemli degil, FakeQdrantClient skoru zaten sabit donduruyor


class TestResolveEntityId:
    def test_exact_match_on_existing_entitys_name(self):
        existing = [{"entity_id": "MECHANISM::wal_write_ahead_logging",
                     "normalized_name": "wal_write_ahead_logging", "aliases": ["WAL", "Write-Ahead Logging"]}]
        tx = FakeTxForResolve(existing)
        qdrant = FakeQdrantClient(hits=[])  # embedding yedegine hic ulasilmamali
        result = gw.resolve_entity_id("Write-Ahead Logging", "MECHANISM", [], tx, qdrant, FakeEmbeddingModel())
        assert result == "MECHANISM::wal_write_ahead_logging"

    def test_exact_match_on_existing_entitys_alias(self):
        existing = [{"entity_id": "MECHANISM::wal_write_ahead_logging",
                     "normalized_name": "wal_write_ahead_logging", "aliases": ["WAL", "Write-Ahead Logging"]}]
        tx = FakeTxForResolve(existing)
        qdrant = FakeQdrantClient(hits=[])
        result = gw.resolve_entity_id("WAL", "MECHANISM", [], tx, qdrant, FakeEmbeddingModel())
        assert result == "MECHANISM::wal_write_ahead_logging"

    def test_new_entitys_own_aliases_are_also_checked(self):
        existing = [{"entity_id": "SERVICE::auth_service",
                     "normalized_name": "auth_service", "aliases": []}]
        tx = FakeTxForResolve(existing)
        qdrant = FakeQdrantClient(hits=[])
        # yeni entity'nin ismi "Auth" ama aliases'inda "Auth Service" var -> kesismeli
        result = gw.resolve_entity_id("Auth", "SERVICE", ["Auth Service"], tx, qdrant, FakeEmbeddingModel())
        assert result == "SERVICE::auth_service"

    def test_falls_back_to_embedding_when_no_exact_match_above_threshold(self):
        tx = FakeTxForResolve([])  # tam eslesme yok
        qdrant = FakeQdrantClient(hits=[FakeQdrantHit(0.92, {"entity_id": "SERVICE::auth_service"})])
        result = gw.resolve_entity_id("Auth servisi", "SERVICE", [], tx, qdrant, FakeEmbeddingModel())
        assert result == "SERVICE::auth_service"

    def test_ignores_embedding_hit_below_threshold(self):
        tx = FakeTxForResolve([])
        qdrant = FakeQdrantClient(hits=[FakeQdrantHit(0.5, {"entity_id": "COMPONENT::gateway"})])
        result = gw.resolve_entity_id("WAL", "MECHANISM", [], tx, qdrant, FakeEmbeddingModel())
        assert result == "MECHANISM::wal"  # normalize_entity_id'nin kendi sonucu, esleme yok

    def test_returns_normalized_id_when_nothing_matches(self):
        tx = FakeTxForResolve([])
        qdrant = FakeQdrantClient(hits=[])
        result = gw.resolve_entity_id("Brand New Thing", "COMPONENT", [], tx, qdrant, FakeEmbeddingModel())
        assert result == "COMPONENT::brand_new_thing"


class TestUpsertEntityEmbedding:
    def test_upserts_a_point_with_expected_payload(self):
        qdrant = FakeQdrantClient()
        gw.upsert_entity_embedding(qdrant, "SERVICE::auth_service", "Auth Service", "SERVICE", FakeEmbeddingModel())
        assert len(qdrant.upserted_points) == 1
        point = qdrant.upserted_points[0]
        assert point.payload == {"entity_id": "SERVICE::auth_service", "type": "SERVICE", "name": "Auth Service"}

    def test_same_entity_id_always_produces_same_point_id(self):
        assert gw.entity_point_id("SERVICE::auth_service") == gw.entity_point_id("SERVICE::auth_service")
        assert gw.entity_point_id("SERVICE::auth_service") != gw.entity_point_id("COMPONENT::gateway")


class FakeSingleResult:
    def __init__(self, value):
        self._value = value

    def single(self):
        return self._value


class FakeTxForChunkExists:
    def __init__(self, existing_chunk_ids):
        self.existing_chunk_ids = existing_chunk_ids

    def run(self, query, **params):
        found = params["chunk_id"] in self.existing_chunk_ids
        return FakeSingleResult({"c": "node"} if found else None)


class TestChunkExists:
    def test_returns_true_when_chunk_already_in_graph(self):
        tx = FakeTxForChunkExists({"ch_0001"})
        assert gw.chunk_exists(tx, "ch_0001") is True

    def test_returns_false_when_chunk_not_in_graph(self):
        tx = FakeTxForChunkExists({"ch_0001"})
        assert gw.chunk_exists(tx, "ch_9999") is False


class TestUpsertRelationValidation:
    def test_rejects_relation_type_outside_ontology(self):
        class ExplodingTx:
            def run(self, *args, **kwargs):
                raise AssertionError("Ontoloji disi relation_type Cypher'a hic gonderilmemeli")

        relation = {
            "source": {"name": "A", "type": "SERVICE"},
            "relation_type": "DESTROYS",  # ontoloji disi
            "target": {"name": "B", "type": "SERVICE"},
            "confidence": 0.9,
        }
        with pytest.raises(ValueError, match="Ontoloji disi"):
            gw.upsert_relation(ExplodingTx(), relation, "ch_0001", resolved_ids={})

    def test_accepts_relation_type_inside_ontology(self):
        calls = []

        class RecordingTx:
            def run(self, query, **params):
                calls.append((query, params))

        relation = {
            "source": {"name": "Auth Service", "type": "SERVICE"},
            "relation_type": "TRIGGERS",
            "target": {"name": "Gateway", "type": "COMPONENT"},
            "confidence": 0.8,
        }
        gw.upsert_relation(RecordingTx(), relation, "ch_0001", resolved_ids={})

        assert len(calls) == 1
        query, params = calls[0]
        assert "TRIGGERS" in query
        assert params["source_id"] == "SERVICE::auth_service"
        assert params["target_id"] == "COMPONENT::gateway"
        assert params["chunk_id"] == "ch_0001"

    def test_new_relation_is_tagged_active_on_create(self):
        calls = []

        class RecordingTx:
            def run(self, query, **params):
                calls.append((query, params))

        relation = {
            "source": {"name": "Auth Service", "type": "SERVICE"},
            "relation_type": "TRIGGERS",
            "target": {"name": "Gateway", "type": "COMPONENT"},
            "confidence": 0.8,
        }
        gw.upsert_relation(RecordingTx(), relation, "ch_0001", resolved_ids={})

        query, _params = calls[0]
        assert "r.status = 'ACTIVE'" in query

    def test_uses_resolved_id_from_resolved_ids_when_available(self):
        calls = []

        class RecordingTx:
            def run(self, query, **params):
                calls.append((query, params))

        relation = {
            "source": {"name": "WAL", "type": "MECHANISM"},
            "relation_type": "USES",
            "target": {"name": "Gateway", "type": "COMPONENT"},
            "confidence": 0.8,
        }
        resolved_ids = {("WAL", "MECHANISM"): "MECHANISM::wal_write_ahead_logging"}

        gw.upsert_relation(RecordingTx(), relation, "ch_0001", resolved_ids=resolved_ids)

        _, params = calls[0]
        assert params["source_id"] == "MECHANISM::wal_write_ahead_logging"  # resolved_ids'ten geldi, naive normalize'dan degil
        assert params["target_id"] == "COMPONENT::gateway"  # resolved_ids'te yok, naive fallback
