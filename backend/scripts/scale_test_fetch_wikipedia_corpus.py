"""sources_scale_test/ icin Turkce Wikipedia'dan gercek makale metni ceker.
Olcek testi icin tek seferlik bir arac. Kategori-tabanli: hangi makalelerin var
oldugunu tahmin etmek yerine, gercek Wikipedia kategorilerinin uye sayfalarini
API'den ceker (var olduklari garanti). docs/superpowers/specs/2026-07-29-scale-test-design.md."""

import os
import re
import time

import requests

from backend.shared.paths import PROJECT_ROOT

API_URL = "https://tr.wikipedia.org/w/api.php"
USER_AGENT = "OtonomSemantikBilgiAgi-ScaleTest/1.0 (egitim/arastirma amacli, tek seferlik olcum)"
CATEGORIES = [
    "Veritabanları",
    "Bilgisayar ağları",
    "İşletim sistemleri",
    "Dağıtık hesaplama",
    "Veri yapıları",
    "Yazılım mühendisliği",
]
TARGET_COUNT = 100
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10


def _get_with_retry(params):
    """429/gecici hatalarda geri cekilip tekrar dener - Wikipedia API'sine
    saygili davranmak icin. Tum denemeler basarisiz olursa None doner (script
    durmaz, cagiran taraf bu makaleyi/kategoriyi atlar)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
            else:
                print(f"    UYARI: {MAX_RETRIES} denemede basarisiz ({exc}), atlaniyor.")
    return None


def fetch_category_members(category, limit=30):
    """Bir Wikipedia kategorisinin uye sayfa basliklarini (sadece ana makale
    ad alani, ns=0) doner. Kategori yoksa/bossa/hata alirsa bos liste doner - hata firlatmaz."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Kategori:{category}",
        "cmlimit": limit,
        "cmnamespace": 0,
        "format": "json",
    }
    data = _get_with_retry(params)
    if data is None:
        return []
    members = data.get("query", {}).get("categorymembers", [])
    return [m["title"] for m in members]


def fetch_article_text(title):
    """Bir makalenin tam duz-metnini doner. Makale yoksa/bossa/hata alirsa None doner."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
    }
    data = _get_with_retry(params)
    if data is None:
        return None
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "").strip()
        if extract:
            return extract
    return None


def slugify_title(title):
    """Dosya adi icin guvenli bir slug uretir."""
    slug = title.strip().lower()
    slug = re.sub(r"[^a-z0-9ığüşöç]+", "_", slug)
    return slug.strip("_")


if __name__ == "__main__":
    BASE_DIR = PROJECT_ROOT
    OUTPUT_DIR = os.path.join(BASE_DIR, "sources_scale_test")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_titles = []
    seen = set()
    for category in CATEGORIES:
        print(f"Kategori taraniyor: {category}")
        titles = fetch_category_members(category)
        for title in titles:
            if title not in seen:
                seen.add(title)
                all_titles.append(title)
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"{len(all_titles)} benzersiz makale basligi bulundu (hedef: {TARGET_COUNT}).")

    saved_count = 0
    for i, title in enumerate(all_titles, start=1):
        if saved_count >= TARGET_COUNT:
            break
        text = fetch_article_text(title)
        if not text or len(text) < 200:
            print(f"  [{i}/{len(all_titles)}] '{title}': ATLANDI (bos/cok kisa)")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
        file_path = os.path.join(OUTPUT_DIR, f"{slugify_title(title)}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        saved_count += 1
        print(f"  [{i}/{len(all_titles)}] '{title}': KAYDEDILDI ({len(text)} karakter) -> {file_path}")
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nBASARILI! {saved_count} makale {OUTPUT_DIR}'e kaydedildi.")
