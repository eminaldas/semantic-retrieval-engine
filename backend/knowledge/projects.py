"""Dis projelerin (orn. nehaber-mobile) belgelerini motora baglayan saf yardimcilar:
proje ayarlari, dosya tarama, degisiklik tespiti (manifest), baslik baglami.

Cekirdek korpus (sources/ + artifacts/chunks.jsonl) bu modulden etkilenmez;
proje belgeleri ayri bir Qdrant koleksiyonunda (store.py) yasar."""

import fnmatch
import hashlib
import json
import os
import re
from datetime import datetime, timezone

from backend.shared.paths import PROJECT_ROOT

CONFIG_FILE = os.path.join(PROJECT_ROOT, "knowledge_projects.json")
NOTES_DIR = os.path.join(PROJECT_ROOT, "knowledge_notes")
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "artifacts", "knowledge")
NOTES_PREFIX = "notes/"
DOCUMENTS_SUBDIR = "belgeler/"  # add_document ile yazilan tam belgeler; not degil, belge sayilir
SUPERSEDED_MARKER = "_Yerini alan not: {path}_"
SUPERSEDED_RE = re.compile(r"^_Yerini alan not: (\S+?)_\s*$", re.MULTILINE)
DEFAULT_INCLUDE = ["**/*.md"]
DEFAULT_EXCLUDE = ["node_modules/**", ".git/**", ".expo/**", "**/node_modules/**"]
SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf", ".docx", ".html", ".htm")

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_LINE_RANGE_RE = re.compile(r"satır (\d+)")


def load_config(config_file=CONFIG_FILE):
    if not os.path.exists(config_file):
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def project_settings(config, project):
    if project not in config:
        known = ", ".join(sorted(config)) or "(hic proje tanimli degil)"
        raise KeyError(f"'{project}' projesi knowledge_projects.json'da yok. Tanimli projeler: {known}")
    settings = dict(config[project])
    settings.setdefault("include", DEFAULT_INCLUDE)
    settings.setdefault("exclude", DEFAULT_EXCLUDE)
    return settings


def _matches(relpath, patterns):
    # "**/x.md" kokteki "x.md"yi de yakalasin diye "**/" onekini ayrica deniyoruz.
    for pattern in patterns:
        if fnmatch.fnmatch(relpath, pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatch(relpath, pattern[3:]):
            return True
    return False


def iter_project_files(root, include, exclude):
    """root altindaki, include'a uyan ve exclude'a uymayan dosyalarin
    '/' ayracli goreli yollarini sirali dondurur."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = [d for d in dirnames if not _matches(f"{rel_dir}{d}/", exclude) and not _matches(f"{rel_dir}{d}/x", exclude)]
        for name in filenames:
            relpath = f"{rel_dir}{name}"
            if not name.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            if _matches(relpath, include) and not _matches(relpath, exclude):
                found.append(relpath)
    return sorted(found)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def diff_manifest(old, new):
    """old/new: {relpath: hash}. Doner: (degisen_ya_da_yeni, silinen) - ikisi de sirali."""
    changed = sorted(p for p, h in new.items() if old.get(p) != h)
    removed = sorted(p for p in old if p not in new)
    return changed, removed


def manifest_path(project, manifest_dir=MANIFEST_DIR):
    return os.path.join(manifest_dir, f"{project}_manifest.json")


def load_manifest(project, manifest_dir=MANIFEST_DIR):
    path = manifest_path(project, manifest_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(project, manifest, manifest_dir=MANIFEST_DIR):
    os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path(project, manifest_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)


def source_id_for(project, relpath):
    return f"{project}:{relpath}"


def heading_path_at(raw_text, line_no):
    """line_no (1 tabanli) satirindan onceki son Markdown baslik zinciri,
    orn. ['Forum Mobil', 'Faz 1', 'Bildirimler']. Kod bloklari icindeki '#'
    satirlari baslik sayilmaz."""
    stack = []
    in_code = False
    for idx, line in enumerate(raw_text.splitlines(), start=1):
        if idx > line_no:
            break
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        match = _HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            stack = [h for h in stack if h[0] < level] + [(level, match.group(2))]
    return [title for _, title in stack]


def start_line_of(location):
    """chunker'in 'satır 12-18' konumundan baslangic satirini cikarir; yoksa None."""
    match = _LINE_RANGE_RE.search(location or "")
    return int(match.group(1)) if match else None


def context_prefix(project, relpath, headings):
    """Contextual retrieval'in LLM'siz, ucuz hali: parcanin hangi proje/belge/
    bolumden geldigini metnin basina ekler. Hem dense hem BM25 vektoru bu
    onekle uretilir; Claude'a donen metin onek icermez."""
    parts = [f"Proje: {project}", f"Belge: {relpath}"]
    if headings:
        parts.append("Bölüm: " + " > ".join(headings))
    return " | ".join(parts)


MAX_CHUNK_CHARS = 1500  # cross-encoder zaten ~512 token'dan sonrasini goremiyor


def _pack(pieces, sep, max_chars):
    parts, current = [], ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        current = piece
    if current:
        parts.append(current)
    return parts


def split_oversized(chunks, max_chars=MAX_CHUNK_CHARS):
    """Anlamsal chunker kod bloklari/tablolar iceren Markdown'da yuzlerce satirlik
    parcalar uretebiliyor; bunlar hem rerank'i yavaslatiyor hem de isabeti
    bulaniklastiriyor. max_chars'i asan parcayi once paragraf, sonra satir,
    en son karakter sinirinda boler. Alt parcalar ust parcanin konumunu tasir."""
    result = []
    for text, location in chunks:
        if len(text) <= max_chars:
            result.append((text, location))
            continue
        pieces = []
        for para in _pack(text.split("\n\n"), "\n\n", max_chars):
            if len(para) <= max_chars:
                pieces.append(para)
                continue
            for line_group in _pack(para.split("\n"), "\n", max_chars):
                pieces.extend(line_group[i:i + max_chars] for i in range(0, len(line_group), max_chars))
        result.extend((piece, location) for piece in pieces if piece.strip())
    return result


def dir_prefixes(relpath):
    """'graf/alt/x.md' -> ['graf/', 'graf/alt/']: path_prefix filtresi Qdrant'ta
    tam esleme (keyword) ile calissin diye her ust klasor ayri deger olarak saklanir."""
    parts = relpath.split("/")[:-1]
    return ["/".join(parts[:i]) + "/" for i in range(1, len(parts) + 1)]


def superseded_by(raw_text):
    """Yerine yeni not yazilmis bir notun isaret satirindaki yeni notun yolu; yoksa None."""
    match = SUPERSEDED_RE.search(raw_text)
    return match.group(1) if match else None


def doc_type_for(relpath):
    if relpath.startswith(NOTES_PREFIX + DOCUMENTS_SUBDIR):
        return "doc"
    return "note" if relpath.startswith(NOTES_PREFIX) else "doc"


def build_chunk_records(project, relpath, raw_text, chunks, mtime=None, indexed_at=None, contexts=None):
    """chunks: chunker.chunk_document ciktisi [(text, location), ...].
    contexts: parca basina LLM baglam cumlesi (contextual.py) ya da None.
    Doner: payload'a ve embedding'e hazir kayitlar."""
    source_id = source_id_for(project, relpath)
    modified = datetime.fromtimestamp(mtime, timezone.utc).isoformat() if mtime else None
    indexed = indexed_at or datetime.now(timezone.utc).isoformat()
    replaced_by = superseded_by(raw_text)
    dirs = dir_prefixes(relpath)
    records = []
    for index, (text, location) in enumerate(chunks):
        line = start_line_of(location)
        headings = heading_path_at(raw_text, line) if line else []
        sentence = contexts[index] if contexts and index < len(contexts) else ""
        prefix = context_prefix(project, relpath, headings) + (f"\nBaglam: {sentence}" if sentence else "")
        records.append({
            "chunk_id": f"{source_id}#{index + 1:04d}",
            "project": project,
            "source_id": source_id,
            "path": relpath,
            "location": location,
            "section": " > ".join(headings),
            "text": text,
            "embed_text": prefix + "\n" + text,
            "chunk_context": sentence or None,
            "doc_type": doc_type_for(relpath),
            "modified_at": modified,
            "indexed_at": indexed,
            "dirs": dirs,
            "superseded_by": replaced_by,
        })
    return records


def source_file(project, settings, relpath, notes_dir=NOTES_DIR):
    """Indeksteki goreli yolun diskteki dosyasi (notlar motorun not klasorunde)."""
    if relpath.startswith(NOTES_PREFIX):
        return os.path.join(notes_dir, project, *relpath[len(NOTES_PREFIX):].split("/"))
    return os.path.join(settings["root"], *relpath.split("/"))


def document_relpath(name):
    """add_document adi -> 'notes/belgeler/<slug>.md'. Klasor ve '..' kabul edilmez."""
    stem = name[:-3] if name.lower().endswith(".md") else name
    return f"{NOTES_PREFIX}{DOCUMENTS_SUBDIR}{slugify_title(stem)}.md"


def slugify_title(title):
    table = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    slug = re.sub(r"[^a-z0-9]+", "-", title.translate(table).lower()).strip("-")
    return slug[:60] or "not"


def note_filename(title, now=None):
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y-%m-%d-%H%M')}-{slugify_title(title)}.md"


def render_note(project, title, body, now=None, supersedes=None):
    now = now or datetime.now(timezone.utc)
    replaces = f" · yerine gectigi not: {supersedes}" if supersedes else ""
    return f"# {title}\n\n_Proje: {project} · {now.strftime('%Y-%m-%d %H:%M')} UTC{replaces}_\n\n{body.strip()}\n"
