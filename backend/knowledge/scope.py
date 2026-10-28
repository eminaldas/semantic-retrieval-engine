"""Bir baglantinin (workspace) gorebildigi ve yazabildigi projeler.

Kapsam modele birakilmaz: MCP araclarina gelen 'project' argumani kapsami yalnizca daraltabilir.
Yonetici kapsami (readable=None) bugunku tek baglantinin davranisidir: tum projeler.
Workspace'ler config'ten okunur (plan Asama 2); bu modul yalnizca kurallari tasir."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scope:
    name: str = "*"
    readable: frozenset | None = None  # None = tum projeler (yonetici)
    writable: frozenset | None = None  # None = okunabilen her proje
    default: str | None = None  # proje verilmeyen cagrilarda kullanilir; yazma araclarinda ev projesi
    external_llm: bool = True  # False: belge metni dis LLM'e (Gemini) gitmez, ask araci kapali
    ask_daily_usd: float | None = None  # ask icin gunluk harcama siniri; None = sinirsiz
    project_base: str | None = None  # propose_project yalnizca bu klasorun altindaki klasorleri onerebilir

    @classmethod
    def admin(cls):
        return cls()

    @classmethod
    def workspace(cls, name, read=(), write=(), default=None, external_llm=True, ask_daily_usd=None,
                  project_base=None):
        """read/write: proje adlari; yazilabilen proje okunabilir de sayilir."""
        writable = frozenset(write)
        readable = frozenset(read) | writable
        if default is not None and default not in readable:
            raise ValueError(f"'{name}' workspace'inin varsayilan projesi '{default}' okunabilir projeleri arasinda degil")
        return cls(name=name, readable=readable, writable=writable, default=default, external_llm=external_llm,
                   ask_daily_usd=ask_daily_usd, project_base=project_base)

    @property
    def is_admin(self):
        return self.readable is None

    def can_read(self, project):
        return self.readable is None or project in self.readable

    def can_write(self, project):
        if not self.can_read(project):
            return False
        return self.writable is None or project in self.writable

    @property
    def read_only(self):
        return self.writable is not None and not self.writable

    def visible(self, names):
        """names icinden kapsamin gorebildikleri, sirali."""
        return sorted(n for n in names if self.can_read(n))
