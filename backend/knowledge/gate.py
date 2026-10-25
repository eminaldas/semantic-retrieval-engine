"""Gerileme kapisi: altin setleri kosar, ozet metrikler esiklerin altina duserse hata verir.

    python -m backend.knowledge.gate [--config knowledge_eval/gate.json]

Push oncesi kancadan (tools/hooks/pre-push) cagrilir; elle de calistirilabilir. Esikler
knowledge_eval/gate.json'da durur (altin setler gibi gitignore'da, makineye ozel):

    {"project": "factprobe",
     "sets": [{"name": "v1.1", "golden_set": "factprobe.jsonl",
               "min": {"hit@3": 0.97, "abstain_accuracy": 1.0}, "max": {"false_abstain": 0.03}}],
     "warn_max": {"latency_p50": 1.0}}

min/max ihlali kapiyi kapatir; warn_max sadece uyarir (sure pilde ~5 kat oynar, push'u durdurmamali).
Gemini'ye dokunmaz. Cikis kodu: 0 gecti ya da yapilandirma yok (kapi bu makinede kurulmamis),
1 esik ihlali ya da olcum calismadi."""

import json
import os
import sys


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_limits(summary, limits):
    """Saf kontrol: [(metrik, deger, sinir, 'min'|'max')] ihlal listesi. Ozette olmayan metrik ihlaldir."""
    failures = []
    for kind in ("min", "max"):
        for metric, bound in (limits.get(kind) or {}).items():
            value = summary.get(metric)
            if value is None or (kind == "min" and value < bound) or (kind == "max" and value > bound):
                failures.append((metric, value, bound, kind))
    return failures


def check_warnings(summary, warn_max):
    return [(m, summary[m], b) for m, b in (warn_max or {}).items() if summary.get(m) is not None and summary[m] > b]


def _fmt(value):
    return "yok" if value is None else f"{value:.3f}"


def main():
    import argparse

    from backend.shared.paths import PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Altin set gerileme kapisi")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "knowledge_eval", "gate.json"))
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Kapi atlandi: {args.config} yok (bu makinede altin set kurulu degil).")
        return 0
    config = load_config(args.config)
    base = os.path.dirname(args.config)

    from backend.knowledge import eval as ev
    from backend.knowledge.runtime import Runtime

    try:
        runtime = Runtime()
        failed = False
        for item in config["sets"]:
            summary, _ = ev.run_eval(config["project"], golden_path=os.path.join(base, item["golden_set"]),
                                     label=f"gate-{item['name']}", runtime=runtime, verbose=False)
            failures = check_limits(summary, item)
            print(f"[{'GECTI' if not failures else 'KALDI'}] {item['name']} ({summary['n']} soru)")
            for metric, value, bound, kind in failures:
                print(f"    {metric} = {_fmt(value)}  ({kind} {bound})")
            for metric, value, bound in check_warnings(summary, config.get("warn_max")):
                print(f"    uyari: {metric} = {value:.2f} > {bound} (sarj takili mi?)")
            failed = failed or bool(failures)
    except Exception as exc:  # Qdrant kapali, model yuklenemedi...: olcum yoksa kapi da yok sayilmaz
        print(f"Kapi calismadi: {type(exc).__name__}: {exc}")
        print("Servisleri ac (tools\\windows\\start_rag.ps1) ya da bilerek atlamak icin: git push --no-verify")
        return 1
    if failed:
        print("Gerileme var. Bilerek kabul ediyorsan knowledge_eval/gate.json esiklerini guncelle "
              "ya da git push --no-verify.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
