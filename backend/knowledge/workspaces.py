"""knowledge_projects.json'daki '_workspaces' bolumu: her workspace hangi projeleri okur ve yazar.

    "_workspaces": {
      "factprobe": {
        "description": "FactProbe calismalari",
        "projects": {"factprobe": "write", "fake-news": "read"},
        "home": "factprobe",
        "instructions": "Bu workspace'e ozel kurallar...",
        "external_llm": true,
        "ask_daily_usd": 0.5
      }
    }

'home' notlarin yazildigi ve proje verilmeyen cagrilarda kullanilan projedir; verilmezse tek yazilabilir
proje. 'external_llm': false ise ask araci (Gemini) bu workspace'te kapalidir. 'ask_daily_usd' ask icin
gunluk harcama siniridir (sunucu yeniden baslayinca sayac sifirlanir). Hatali bolum sunucuyu actirmaz: yanlis yazilmis bir alan sessizce genis yetki vermesin.

Kontrol: python -m backend.knowledge.workspaces"""

import re
import sys
from dataclasses import dataclass

from backend.knowledge import projects
from backend.knowledge.scope import Scope

SECTION = "_workspaces"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")  # /w/<ad>/mcp adresinde kullanilir
ROLES = ("read", "write")
FIELDS = {"description", "projects", "home", "instructions", "external_llm", "ask_daily_usd"}
MAX_INSTRUCTIONS_CHARS = 2000


@dataclass(frozen=True)
class Workspace:
    name: str
    scope: Scope
    description: str = ""
    instructions: str = ""


def load_workspaces(config):
    """config: load_config ciktisi. Doner: {ad: Workspace}. Hata varsa hepsini tek ValueError'da toplar."""
    section = config.get(SECTION)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"{SECTION} bir nesne olmali")
    known = {k for k in config if not k.startswith("_")}
    errors, result = [], {}
    for name, raw in section.items():
        problems = _check(name, raw, known)
        if problems:
            errors += [f"{name}: {p}" for p in problems]
            continue
        roles = raw["projects"]
        write = [p for p, role in roles.items() if role == "write"]
        home = raw.get("home") or (write[0] if len(write) == 1 else None)
        scope = Scope.workspace(name, read=[p for p, role in roles.items() if role == "read"], write=write, default=home,
                                external_llm=raw.get("external_llm", True), ask_daily_usd=raw.get("ask_daily_usd"))
        result[name] = Workspace(name, scope, raw.get("description", ""), raw.get("instructions", ""))
    if errors:
        raise ValueError(f"{SECTION} hatali:\n- " + "\n- ".join(errors))
    return result


def _check(name, raw, known):
    if not isinstance(name, str) or not NAME_RE.match(name):
        return ["ad kucuk harf, rakam ve '-' olmali (en fazla 32 karakter)"]
    if not isinstance(raw, dict):
        return ["ayarlar bir nesne olmali"]
    problems = [f"bilinmeyen alan '{key}'" for key in sorted(set(raw) - FIELDS)]
    roles = raw.get("projects")
    if not isinstance(roles, dict) or not roles:
        return problems + ["'projects' en az bir proje icermeli, orn. {\"factprobe\": \"write\"}"]
    for project, role in roles.items():
        if project not in known:
            problems.append(f"'{project}' projesi tanimli degil")
        if role not in ROLES:
            problems.append(f"'{project}' icin rol 'read' ya da 'write' olmali, gelen: {role!r}")
    home = raw.get("home")
    if home is not None and roles.get(home) != "write":
        problems.append(f"home '{home}' bu workspace'te 'write' rolunde bir proje olmali")
    for key in ("description", "instructions"):
        if key in raw and not isinstance(raw[key], str):
            problems.append(f"'{key}' metin olmali")
    if "external_llm" in raw and not isinstance(raw["external_llm"], bool):
        problems.append("'external_llm' true ya da false olmali")
    budget = raw.get("ask_daily_usd")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget < 0):
        problems.append("'ask_daily_usd' sifir ya da pozitif bir sayi olmali")
    if len(raw.get("instructions") or "") > MAX_INSTRUCTIONS_CHARS:
        problems.append(f"'instructions' en fazla {MAX_INSTRUCTIONS_CHARS} karakter")
    return problems


def describe(workspace):
    scope = workspace.scope
    lines = [f"{workspace.name}: {workspace.description}".rstrip(": ")]
    for project in sorted(scope.readable):
        role = "yazar" if scope.can_write(project) else "okur"
        lines.append(f"  - {project} ({role}" + (", ev projesi)" if project == scope.default else ")"))
    if not scope.external_llm:
        lines.append("  dis LLM kapali (ask yok)")
    elif scope.ask_daily_usd is not None:
        lines.append(f"  ask gunluk siniri: ${scope.ask_daily_usd:g}")
    if workspace.instructions:
        lines.append(f"  talimat: {len(workspace.instructions)} karakter")
    return "\n".join(lines)


def main():
    try:
        found = load_workspaces(projects.load_config())
    except ValueError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
    if not found:
        print(f"Tanimli workspace yok ({SECTION} bolumu bos ya da eksik). /mcp tum projeleri gorur.")
        return
    print("\n".join(describe(w) for w in found.values()))


if __name__ == "__main__":
    main()
