import os
import re
import sys
import json
from datetime import datetime, timezone
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup

from backend.shared.paths import PROJECT_ROOT

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

SUPPORTED_EXTENSIONS = ('.txt', '.md', '.pdf', '.docx', '.html', '.htm')


def slugify(name):
    """Dosya adini (uzantisiz) chunk_id'de guvenle kullanilabilecek bir sekle indirger."""
    stem = os.path.splitext(name)[0]
    return re.sub(r'[^a-zA-Z0-9_]+', '_', stem)


def chunk_id_for(source_name, index):
    """Dosyaya bagli, kararli chunk_id uretir - global degil per-file numaralandirma.
    Yeni bir dosya eklemek mevcut dosyalarin chunk_id'lerini asla etkilemez."""
    return f"{slugify(source_name)}_ch{index + 1:04d}"


def load_document(file_path):
    """.txt/.md/.pdf/.docx/.html/.htm dosyalarını okuyup düz metne çevirir."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == '.docx':
        document = docx.Document(file_path)
        return "\n".join(p.text for p in document.paragraphs)
    if ext in ('.html', '.htm'):
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(separator='\n')
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def locate_sentence_offsets(raw_text, sentences):
    """Her cumlenin raw_text icindeki (start, end) karakter ofsetini sirali
    .find() taramasiyla bulur. NLTK metni yeniden yazmadigi icin normalde her
    cumle raw_text'in tam bir alt string'idir. Bulunamayan bir cumle icin
    (None, None) doner - imlec o durumda ilerlemez."""
    offsets = []
    cursor = 0
    for sentence in sentences:
        idx = raw_text.find(sentence, cursor)
        if idx == -1:
            offsets.append((None, None))
            continue
        start = idx
        end = idx + len(sentence)
        offsets.append((start, end))
        cursor = end
    return offsets


def load_pdf_with_page_offsets(file_path):
    """load_document'in PDF dalindaki ayni mantik (sayfalari '\n' ile birlestirir),
    ama her sayfanin birlesik metin icindeki baslangic karakter ofsetini de
    (page_offsets[i] = (i+1). sayfanin baslangic ofseti, 0-indeksli) dondurur."""
    reader = PdfReader(file_path)
    text_parts = []
    page_offsets = []
    cursor = 0
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_offsets.append(cursor)
        text_parts.append(page_text)
        cursor += len(page_text) + 1  # +1: "\n".join ayiricisi
    return "\n".join(text_parts), page_offsets


def format_line_location(raw_text, start_offset, end_offset):
    """start_offset/end_offset (raw_text icindeki karakter ofsetleri) araligindaki
    1-indeksli satir numaralarini hesaplar."""
    start_line = raw_text.count('\n', 0, start_offset) + 1
    end_line = raw_text.count('\n', 0, end_offset) + 1
    if start_line == end_line:
        return f"satır {start_line}"
    return f"satır {start_line}-{end_line}"


def format_page_location(start_offset, end_offset, page_offsets):
    """start_offset/end_offset'in hangi sayfada (1-indeksli) oldugunu page_offsets'e
    gore bulur - page_offsets[i] = (i+1). sayfanin baslangic ofseti (artan sirali)."""
    def page_of(offset):
        page_num = 1
        for i, page_start in enumerate(page_offsets):
            if offset >= page_start:
                page_num = i + 1
        return page_num

    start_page = page_of(start_offset)
    end_page = page_of(end_offset)
    if start_page == end_page:
        return f"sayfa {start_page}"
    return f"sayfa {start_page}-{end_page}"


def _resolve_location(start_offset, end_offset, fallback_index, raw_text, page_offsets):
    """start_offset/end_offset gecerliyse gercek bir satir/sayfa araligi, degilse
    (cumle bulunamadiysa) eski chunk_index_N bicimine geri duser."""
    if start_offset is None or end_offset is None:
        return f"chunk_index_{fallback_index}"
    if page_offsets is not None:
        return format_page_location(start_offset, end_offset, page_offsets)
    return format_line_location(raw_text, start_offset, end_offset)


_ORDINAL_DOT_RE = re.compile(r'(\d{1,3})\.(?=\s|$)')
_ORDINAL_PLACEHOLDER = '․'  # ONE DOT LEADER - gorunuse noktaya benzer, NLTK cumle sonu saymiyor


def split_into_sentences(text):
    """NLTK'nin Turkce cumle bolucusu, '1. Grup' gibi Turkce sira sayisi
    kaliplarindaki noktayi cumle sonu saniyor. Tokenize etmeden once bu
    kaliplari bir yer tutucuya cevirip, tokenize sonrasi geri yukleriz -
    boylece nokta goruntude hicbir sey degismez, sadece bolme davranisi
    duzelir. docs/superpowers/specs/2026-08-01-chunker-ordinal-split-fix-design.md"""
    protected = _ORDINAL_DOT_RE.sub(lambda m: m.group(1) + _ORDINAL_PLACEHOLDER, text)
    sentences = sent_tokenize(protected, language='turkish')
    restored = [s.replace(_ORDINAL_PLACEHOLDER, '.') for s in sentences]
    return [s.strip() for s in restored if len(s.strip()) > 0]


def get_embeddings(sentences, model):
    """Cümleleri embedding modeliyle vektörlere çevirir."""
    return model.encode(sentences)


def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def create_chunks_with_ranges(sentences, similarities, threshold=0.36):
    """create_chunks ile ayni gruplama mantigi, ama her chunk'in hangi cumle-indeks
    araligini (start_idx, end_idx - ikisi de dahil) kapladigini da dondurur."""
    if not sentences:
        return []

    chunks = []
    current_chunk = [sentences[0]]
    current_start = 0

    for i, sim_score in enumerate(similarities):
        # Benzerlik eşiğin altındaysa, konunun koptuğunu varsay ve yeni parçaya geç
        if sim_score < threshold:
            chunks.append((" ".join(current_chunk), current_start, i))
            current_chunk = [sentences[i + 1]]
            current_start = i + 1
        else:
            current_chunk.append(sentences[i + 1])

    # Döngü bitince kalan son parçayı da ekle
    if current_chunk:
        chunks.append((" ".join(current_chunk), current_start, len(sentences) - 1))

    return chunks


def create_chunks(sentences, similarities, threshold=0.36):
    """Benzerlik skorlarına göre cümleleri anlamsal bloklara ayırır.
    create_chunks_with_ranges'in ince bir sarmalayicisi - sadece metin listesi doner."""
    return [text for text, _, _ in create_chunks_with_ranges(sentences, similarities, threshold)]


def chunk_document(file_path, model, threshold=0.36):
    """Tek bir dosyayı yükler, cümlelere böler, anlamsal chunk'lara ayırır ve her
    chunk için gerçek konum bilgisini (satır ya da sayfa araligi) hesaplar.
    Doner: list[(chunk_text, location)]."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        raw_text, page_offsets = load_pdf_with_page_offsets(file_path)
    else:
        raw_text = load_document(file_path)
        page_offsets = None

    sentences = split_into_sentences(raw_text)
    offsets = locate_sentence_offsets(raw_text, sentences)

    if len(sentences) < 2:
        return [
            (s, _resolve_location(start, end, idx, raw_text, page_offsets))
            for idx, (s, (start, end)) in enumerate(zip(sentences, offsets))
        ]

    embeddings = get_embeddings(sentences, model)
    similarities = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(sentences) - 1)
    ]
    chunk_ranges = create_chunks_with_ranges(sentences, similarities, threshold)

    result = []
    for idx, (chunk_text, start_idx, end_idx) in enumerate(chunk_ranges):
        start_offset, _ = offsets[start_idx]
        _, end_offset = offsets[end_idx]
        location = _resolve_location(start_offset, end_offset, idx, raw_text, page_offsets)
        result.append((chunk_text, location))
    return result


def find_source_files(sources_dir):
    """sources_dir altındaki desteklenen (.txt/.md/.pdf) tüm dosyaları bulur."""
    return sorted(
        os.path.join(sources_dir, name)
        for name in os.listdir(sources_dir)
        if name.lower().endswith(SUPPORTED_EXTENSIONS)
    )


def already_ingested_sources(chunks_file):
    """chunks_file'da (varsa) hangi source_id'lerin zaten var oldugunu dondurur.
    Dosya yoksa bos kume doner - henuz hicbir sey islenmemis demektir."""
    if not os.path.exists(chunks_file):
        return set()
    sources = set()
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sources.add(json.loads(line)['source_id'])
    return sources


def save_to_jsonl(chunk_records, output_file):
    """Mimari sözleşmeye (v0.1) uygun şekilde JSONL formatında kaydeder.
    Artimli: dosya zaten varsa APPEND edilir (uzerine yazilmaz), her source_id
    kendi 0'dan baslayan yerel numaralandirmasini kullanir (chunk_id_for) -
    boylece yeni bir dosya eklemek mevcut dosyalarin chunk_id'lerini asla etkilemez.

    chunk_records: [(chunk_text, source_filename, location), ...]
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mode = 'a' if os.path.exists(output_file) else 'w'
    counters = {}
    with open(output_file, mode, encoding='utf-8') as f:
        for chunk_text, source_filename, location in chunk_records:
            index = counters.get(source_filename, 0)
            counters[source_filename] = index + 1

            # Quote (Alıntı) için ilk 5 kelimeyi al
            quote = " ".join(chunk_text.split()[:5]) + "..."

            chunk_data = {
                "chunk_id": chunk_id_for(source_filename, index),
                "source_id": source_filename,
                "location": location,
                "text": chunk_text,
                "quote": quote,
                "ingested_at": datetime.now(timezone.utc).isoformat()
            }
            # JSONL kuralı: Her satırda bir JSON objesi olmalı (indentasyon yok)
            f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    BASE_DIR = PROJECT_ROOT
    SOURCES_DIR = os.path.join(BASE_DIR, "sources")
    OUTPUT_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")

    # Argüman verilirse tek dosya, verilmezse sources/ altındaki tüm desteklenen
    # dosyalar aday olur. Ikisinde de zaten islenmis (chunks.jsonl'da source_id'si
    # var olan) dosyalar atlanir - tekrar chunk'lamak duplicate chunk_id yaratirdi.
    if len(sys.argv) > 1:
        candidate_files = [sys.argv[1]]
    else:
        candidate_files = find_source_files(SOURCES_DIR)

    already_ingested = already_ingested_sources(OUTPUT_FILE)
    source_files = [f for f in candidate_files if os.path.basename(f) not in already_ingested]
    skipped_files = [f for f in candidate_files if os.path.basename(f) in already_ingested]
    if skipped_files:
        print(f"Zaten islenmis, atlaniyor: {[os.path.basename(f) for f in skipped_files]}")

    if not source_files:
        print("Islenecek yeni dosya yok." if skipped_files else f"HATA: {SOURCES_DIR} altında desteklenen bir dosya bulunamadı.")
    else:
        print("Model yükleniyor ve dökümanlar anlamsal analizden geçiriliyor...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        all_records = []
        for file_path in source_files:
            if not os.path.exists(file_path):
                print(f"HATA: {file_path} bulunamadı, atlanıyor.")
                continue

            source_name = os.path.basename(file_path)
            chunks = chunk_document(file_path, embedding_model)
            all_records.extend((chunk_text, source_name, location) for chunk_text, location in chunks)
            print(f"  {source_name}: {len(chunks)} parça")

        save_to_jsonl(all_records, OUTPUT_FILE)

        print(f"\nBAŞARILI! {len(source_files)} döküman, toplam {len(all_records)} anlamsal parçaya bölündü.")
        print(f"Veriler mimari sözleşmeye uygun olarak şuraya kaydedildi: \n{OUTPUT_FILE}")
