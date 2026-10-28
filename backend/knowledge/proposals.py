"""Yeni proje onerileri: Claude bir klasoru proje olarak onerebilir, ama proje ancak cihazda onaylaninca eklenir.

Neden onay: klasor secimini modele birakmak, bir belgedeki ya da web sayfasindaki gizli talimatin Claude'u
baska bir klasoru (orn. ~/.ssh) indekslemeye yonlendirmesine kapi acar. Bu yuzden:
  - oneri yalnizca workspace'in 'project_base' klasorunun altindaki, gizli olmayan bir klasor olabilir;
  - oneri bir dosyaya yazilir, knowledge_projects.json'a dokunulmaz;
  - onay ve ret yalnizca bu komut satirindan (MCP'de onay araci yok).

    python -m backend.knowledge.proposals list
    python -m backend.knowledge.proposals approve <ad> [--home]
    python -m backend.knowledge.proposals reject <ad>

Onaydan sonra sunucu yeniden baslatilir (workspace kapsami acilista okunur); izleyici yeni projeyi indeksler."""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

from backend.knowledge import projects

PROPOSALS_FILE = os.path.join(projects.MANIFEST_DIR, "project_proposals.json")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
MAX_PENDING = 5  # bir workspace'in bekleyen onerisi; model oneri yagdirmasin
MAX_DESCRIPTION_CHARS = 200
# Repolardaki sanal ortam ve bagimlilik klasorlerinin README'leri projenin belgesi degil (factprobe'da 74 yerine yuzlerce dosya)
EXCLUDE = projects.DEFAULT_EXCLUDE + [".*/**", "**/.*/**", "venv*/**", "**/venv*/**", "**/site-packages/**",
                                      "dist/**", "build/**"]


def load(path=PROPOSALS_FILE):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(proposals, path=PROPOSALS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(proposals, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def resolve_folder(base, folder):
    """base altindaki goreli klasoru tam yola cevirir; disari cikan, gizli ya da olmayan klasor hata verir."""
    if not folder or not folder.strip():
        raise ValueError("folder bos olamaz")
    rel = folder.strip().replace("\\", "/").strip("/")
    if os.path.isabs(folder) or re.match(r"^[A-Za-z]:", folder):
        raise ValueError(f"folder '{os.path.basename(base)}' klasorune gore goreli olmali (orn. 'yeni-repo'), tam yol degil")
    parts = rel.split("/")
    if any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        raise ValueError("folder '..' ya da gizli klasor (nokta ile baslayan) iceremez")
    base_real = os.path.realpath(base)
    path = os.path.realpath(os.path.join(base_real, *parts))
    if os.path.commonpath([base_real, path]) != base_real or path == base_real:
        raise ValueError("folder izin verilen ana klasorun altinda olmali")  # sembolik baglanti disari cikamasin
    if not os.path.isdir(path):
        raise ValueError(f"Klasor bulunamadi: {rel}")
    return path


def count_documents(root):
    return len(projects.iter_project_files(root, projects.DEFAULT_INCLUDE, EXCLUDE))


def propose(scope, config, name, folder, description="", path=PROPOSALS_FILE, now=None):
    """Oneriyi kaydeder; doner: modele gosterilecek metin. Proje eklemez."""
    if not scope.project_base:
        raise ValueError("Bu workspace'te yeni proje onerilemez (project_base tanimli degil).")
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise ValueError("name kucuk harf, rakam ve '-' olmali (en fazla 32 karakter), orn. 'yeni-repo'")
    if name in config:
        raise ValueError(f"'{name}' adli bir proje zaten var; baska bir ad secin.")
    description = " ".join((description or "").split())[:MAX_DESCRIPTION_CHARS]
    root = resolve_folder(scope.project_base, folder)
    proposals = load(path)
    if name in proposals:
        raise ValueError(f"'{name}' icin bekleyen bir oneri zaten var; onay bekleniyor.")
    if any(p["root"] == root for p in proposals.values()) or any(
            os.path.realpath(s.get("root", "")) == root for k, s in config.items() if not k.startswith("_")):
        raise ValueError("Bu klasor zaten bir proje ya da bekleyen bir oneri.")
    if sum(1 for p in proposals.values() if p["workspace"] == scope.name) >= MAX_PENDING:
        raise ValueError(f"Bu workspace'in {MAX_PENDING} bekleyen onerisi var; once onlar onaylanmali ya da reddedilmeli.")
    documents = count_documents(root)
    proposals[name] = {"workspace": scope.name, "root": root, "description": description, "documents": documents,
                       "proposed_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")}
    save(proposals, path)
    return (f"Oneri kaydedildi: '{name}' ({documents} Markdown belge). Proje henuz eklenmedi; kullanicinin "
            f"bilgisayarda onaylamasi gerekiyor:\n  python -m backend.knowledge.proposals approve {name}\n"
            "Kullaniciya bunu soyleyin. Onaydan ve sunucu yeniden baslatildiktan sonra project='" + name +
            "' ile aranabilir.")


def approve(name, config_file=projects.CONFIG_FILE, path=PROPOSALS_FILE, home=False):
    """Oneriyi knowledge_projects.json'a proje olarak ekler ve onerinin workspace'ine 'write' rolunde baglar.
    Eski config .bak olarak saklanir. Doner: eklenen proje ayarlari."""
    proposals = load(path)
    if name not in proposals:
        raise ValueError(f"'{name}' adli bekleyen oneri yok.")
    item = proposals[name]
    config = projects.load_config(config_file)
    if name in config:
        raise ValueError(f"'{name}' adli proje zaten var.")
    workspace = config.get("_workspaces", {}).get(item["workspace"])
    if workspace is None:
        raise ValueError(f"Onerinin workspace'i ('{item['workspace']}') artik tanimli degil.")
    config[name] = {"root": item["root"], "description": item["description"], "exclude": EXCLUDE}
    workspace.setdefault("projects", {})[name] = "write"
    if home:
        workspace["home"] = name
    if os.path.exists(config_file):
        shutil.copyfile(config_file, config_file + ".bak")
    tmp = config_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    from backend.knowledge.workspaces import load_workspaces

    try:
        load_workspaces(projects.load_config(tmp))  # bozuk config yazilip sunucu acilmaz hale gelmesin
    except ValueError:
        os.remove(tmp)
        raise
    os.replace(tmp, config_file)
    del proposals[name]
    save(proposals, path)
    return config[name]


def reject(name, path=PROPOSALS_FILE):
    proposals = load(path)
    if proposals.pop(name, None) is None:
        raise ValueError(f"'{name}' adli bekleyen oneri yok.")
    save(proposals, path)


def main():
    parser = argparse.ArgumentParser(description="Yeni proje onerilerini listele, onayla ya da reddet")
    parser.add_argument("command", choices=("list", "approve", "reject"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--home", action="store_true", help="onaylanan proje workspace'in ev projesi olsun")
    args = parser.parse_args()
    try:
        if args.command == "list":
            proposals = load()
            if not proposals:
                print("Bekleyen oneri yok.")
            for name, p in sorted(proposals.items()):
                print(f"{name}: {p['root']} ({p['documents']} belge, workspace {p['workspace']}, {p['proposed_at']})"
                      + (f"\n  {p['description']}" if p["description"] else ""))
            return 0
        if not args.name:
            parser.error("proje adi gerekli")
        if args.command == "reject":
            reject(args.name)
            print(f"{args.name}: oneri reddedildi.")
            return 0
        settings = approve(args.name, home=args.home)
    except ValueError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    print(f"{args.name} eklendi: {settings['root']}. Sunucuyu yeniden baslatin (artifacts\\logs\\restart_server.ps1); "
          "izleyici projeyi birkac dakika icinde indeksler.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
