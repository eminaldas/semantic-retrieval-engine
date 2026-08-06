"""MODE_A (acik uclu sorgu, A/C verilmeden) icin risk siralamasi modulu.
Neo4j GDS PageRank ile graf-geneli bir "en riskli nokta" siralamasi hesaplar,
ilgili kaniti toplar. docs/superpowers/specs/2026-07-26-risk-ranking-design.md.

step5_query_engine.py'yi import ETMEZ (dongusel import'u onlemek icin) -
RELATION_WEIGHTS gibi paylasilan degerler parametre olarak alinir, capraz
modul importu ile degil."""


def format_risk_summary(ranked_nodes):
    """ranked_nodes -> MODE_B'nin path_summary'sinin yerini alacak metin.
    step5_query_engine.py::build_context_pack'in path_summary_override'ina verilir."""
    if not ranked_nodes:
        return "(Grafta skorlanacak entity bulunamadi.)"

    lines = [f"En yüksek risk skoruna sahip {len(ranked_nodes)} nokta (PageRank):"]
    for i, node in enumerate(ranked_nodes, start=1):
        lines.append(f"{i}. {node['entity_id']} (skor={node['score']:.2f})")
    return "\n".join(lines)


def build_risk_evidence_pool(ranked_nodes, seed_chunk_ids, chunks_by_id):
    """Risk entity'lerinin kaniti + vektor seed kaniti birlesip tekillestirilir.
    step5_query_engine.py::build_evidence_pool ile ayni birlestirme deseni,
    path relationship'leri yerine ranked_nodes'un evidence_chunk_ids'i kullanilir."""
    chunk_ids = set(seed_chunk_ids)
    for node in ranked_nodes:
        chunk_ids.update(node["evidence_chunk_ids"])
    return [chunks_by_id[cid] for cid in chunk_ids if cid in chunks_by_id]


def compute_risk_ranking(session, relation_weights, top_k=5):
    """Neo4j GDS PageRank ile graf-geneli risk siralamasi hesaplar. Canli GDS
    cagrisi - unit test yok (bkz. plan Global Constraints).

    Adimlar: (1) onceki calistirmadan kalan 'riskGraph' projeksiyonunu temizle
    (sunucu surec CLI cagrilari arasinda kalici oldugu icin), (2) her kenara
    RELATION_WEIGHTS'ten gelen agirligi 'pagerank_weight' property'si olarak
    yaz (SUPERSEDED kenarlara 0.0 - projeksiyonda kalsinlar ama PageRank'e
    katkilari sifir olsun), (3) projeksiyonu olustur (SADECE grafta su an en
    az bir kenari olan tipler icin - gds.graph.project bir tip icin hic kenar
    yoksa 'Invalid relationship projection, ... types not found' hatasi verir,
    bu yuzden projeksiyondan once mevcut tipler canli sorgulanir), (4) PageRank
    calistir, (5) projeksiyonu tekrar temizle, (6) her top-k entity icin
    MENTIONED_IN uzerinden kanit chunk_id'lerini topla."""
    rel_types = list(relation_weights.keys())

    session.run("CALL gds.graph.drop('riskGraph', false)")

    # rel_type, RELATION_WEIGHTS.keys() uzerinden geliyor (step5_query_engine.py'de
    # hardcoded sabit) - kullanici girdisi degil, bu yuzden interpolasyon guvenli
    # (step3_graph_writer.py'nin relation_type whitelist-kontrolu ile ayni mantik).
    for rel_type, weight in relation_weights.items():
        session.run(
            f"MATCH ()-[r:{rel_type}]->() "
            f"SET r.pagerank_weight = CASE WHEN coalesce(r.status,'ACTIVE') = 'SUPERSEDED' THEN 0.0 ELSE $weight END",
            weight=float(weight),
        )

    existing_types_result = session.run(
        "MATCH ()-[r]->() WHERE type(r) IN $rel_types RETURN DISTINCT type(r) AS t",
        rel_types=rel_types,
    )
    existing_rel_types = [record["t"] for record in existing_types_result]

    session.run(
        "CALL gds.graph.project('riskGraph', 'Entity', $rel_types, {relationshipProperties: 'pagerank_weight'})",
        rel_types=existing_rel_types,
    )

    result = session.run(
        "CALL gds.pageRank.stream('riskGraph', {relationshipWeightProperty: 'pagerank_weight'}) "
        "YIELD nodeId, score "
        "RETURN gds.util.asNode(nodeId).entity_id AS entity_id, "
        "       gds.util.asNode(nodeId).name AS name, "
        "       gds.util.asNode(nodeId).type AS type, "
        "       score "
        "ORDER BY score DESC "
        "LIMIT $top_k",
        top_k=top_k,
    )
    ranked = [dict(record) for record in result]

    session.run("CALL gds.graph.drop('riskGraph', false)")

    if not ranked:
        return []

    entity_ids = [node["entity_id"] for node in ranked]
    evidence_result = session.run(
        "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk) WHERE e.entity_id IN $entity_ids "
        "RETURN e.entity_id AS entity_id, collect(c.chunk_id) AS chunk_ids",
        entity_ids=entity_ids,
    )
    evidence_by_entity = {record["entity_id"]: record["chunk_ids"] for record in evidence_result}

    for node in ranked:
        node["evidence_chunk_ids"] = evidence_by_entity.get(node["entity_id"], [])

    return ranked
