"""Kendi BM25 sparse vektorumuz - Qdrant'in sunucu tarafi BM25'ine ya da
fastembed'e bagimli olmadan hibrit arama yapabilmek icin.

Dokuman tarafi: her terim icin BM25'in TF doygunluk formulu
    tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
IDF kismini Qdrant koleksiyonundaki Modifier.IDF hesapliyor (koleksiyondaki
tum dokumanlara gore, sorgu aninda) - bu yuzden burada sadece TF var.
Sorgu tarafi: her terim 1.0 agirlikla; skor = sum(idf * doc_tf_weight)."""

import re
import zlib
from collections import Counter

K1 = 1.2
B = 0.75
AVG_DOC_LENGTH = 256  # fastembed'in Qdrant/bm25 varsayilaniyla ayni sabit ortalama

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Turkce buyuk I/İ'nin casefold()'da yanlis kucuk harfe donmesini engeller.
_TURKISH_UPPER = str.maketrans({"I": "ı", "İ": "i"})


def tokenize(text):
    """Kucuk harfe indirilmis kelime listesi. Tek harfli tokenlar (a, b, 1)
    gurultu oldugu icin atilir."""
    lowered = text.translate(_TURKISH_UPPER).lower()
    return [t for t in _TOKEN_RE.findall(lowered) if len(t) > 1]


STEM_PREFIX = "~"  # kok terimi yuzey kelimeyle ayni id'ye dusmesin ("karar" kelimesi != "~karar" koku)
# Sadece kokten eslesme birebir kelimeden zayif sayilir. factprobe altin setinde (54 soru,
# 6'si cekimli) 0 / 0.25 / 0.5 denendi: Hit@3 0.88 / 0.93 / 0.90 (2026-10-22).
STEM_QUERY_WEIGHT = 0.25
_stemmer = None


def stem(token):
    """Turkce Snowball koku: 'kararları' -> 'karar', 'eşiği' -> 'eşik'. Ekli kelimelerin
    birebir BM25'te kacmasini onler; Ingilizce/kod terimlerinde zararsiz."""
    global _stemmer
    if _stemmer is None:
        import snowballstemmer

        _stemmer = snowballstemmer.stemmer("turkish")
    return _stemmer.stemWord(token)


def terms(tokens):
    """Yuzey kelimeler + her kelimenin kok terimi."""
    return tokens + [STEM_PREFIX + stem(t) for t in tokens]


def term_id(token):
    """Kararli terim id'si - ayni token her calistirmada ayni sparse indekse duser."""
    return zlib.crc32(token.encode("utf-8"))


def _merge_ids(weights_by_token):
    """Iki farkli token ayni crc32'ye duserse (nadir) agirliklari toplanir -
    Qdrant sparse vektorde tekrar eden indeks kabul etmiyor."""
    merged = {}
    for token, weight in weights_by_token.items():
        idx = term_id(token)
        merged[idx] = merged.get(idx, 0.0) + weight
    indices = sorted(merged)
    return indices, [merged[i] for i in indices]


def document_sparse_vector(text, k1=K1, b=B, avg_len=AVG_DOC_LENGTH):
    """(indices, values) - Qdrant SparseVector'e dogrudan verilebilir."""
    tokens = tokenize(text)
    if not tokens:
        return [], []
    doc_len = len(tokens)  # kok terimleri uzunluga sayilmaz
    counts = Counter(terms(tokens))
    norm = k1 * (1 - b + b * doc_len / avg_len)
    weights = {tok: tf * (k1 + 1) / (tf + norm) for tok, tf in counts.items()}
    return _merge_ids(weights)


def query_sparse_vector(text):
    tokens = sorted(set(tokenize(text)))
    weights = {tok: 1.0 for tok in tokens}
    for tok in tokens:
        weights[STEM_PREFIX + stem(tok)] = STEM_QUERY_WEIGHT
    return _merge_ids(weights)
