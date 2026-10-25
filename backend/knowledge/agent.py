"""Arastirma ajani (ReAct): Gemini soruyu alt sorulara boler, motorun kendi araclariyla
(search, get_document, overview) arar, sonuc zayifsa farkli kelimelerle yeniden arar ve
kaynak gostererek cevaplar. Claude MCP uzerinden bu dongunun aynisini kendisi kurabiliyor;
bu modul motoru LLM'den bagimsiz, tek komutla soru cevaplayan bir sistem olarak da gosterir.

Proje secimi modele birakilmaz: ajan sadece kendisine verilen projede arar (fake-news gibi
ayri projeler yanlislikla karismasin).

Kullanim:
  python -m backend.knowledge.agent "soru" [--project factprobe]
  python -m backend.knowledge.agent --eval [--limit 20] [--judge]   # altin sette kaynak isabeti (+ LLM yargic)
  python -m backend.knowledge.agent --eval --long-context [--judge]  # arama yok, tum belgeler baglamda (longctx)"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

from backend.shared import tracing

DEFAULT_MODEL = "gemini-flash-latest"
CALL_SITE = "knowledge_agent"
MAX_STEPS = 6
MAX_TOOL_CHARS = 6000  # bir arac sonucunun modele giden en fazla uzunlugu

SYSTEM_PROMPT = """Sen bir proje belgeleri arastirma asistanisin. Sadece araclardan gelen belge
parcalarina dayanarak Turkce cevap ver.

Yontem:
1. Soru birden fazla konu iceriyorsa alt sorulara bol ve her birini ayri ara (bir cagrida en fazla 4).
2. Sonuclar 'zayif eslesme' ya da UYARI iceriyorsa esanlamli kelimelerle, ekleri cikararak ya da
   karar numarasi/dosya adi gibi ozel terimlerle bir kez daha ara. Proje yapisini bilmiyorsan overview.
3. Bir parca yetmiyorsa ilgili belgeyi get_document ile oku.
4. Yeterli kanit varsa dur ve cevapla. Gereksiz arama yapma.

Cevap kurallari:
- Her iddiadan sonra kaynagi [belge/yolu.md] biciminde yaz.
- Ayni konuda celisen parcalar varsa yeni tarihliyi esas al ve eskisini belirt.
- 'zayif eslesme', 'eski not' ya da 'guncel degil' isaretli parcalari kesin kanit sayma.
- Sorulan bilginin kendisi belgelerde yoksa cevabin ilk cumlesi tam olarak "Belgelerde bulamadim." olsun;
  ardindan en fazla bir cumleyle yakin olarak ne bulundugunu kaynakla ekleyebilirsin. Genel bilgini proje
  bilgisi gibi sunma. Tum sonuclar UYARI ya da 'zayif eslesme' ise bu da bulunamadi demektir."""

FINAL_NUDGE = ("Arac kullanmadan, simdiye kadar gelen kanitla simdi cevapla. "
               "Kanit yoksa ilk cumlen 'Belgelerde bulamadim.' olsun.")
NOT_FOUND_RE = re.compile(
    r"belgelerde\s+bulamad|belgeler(?:de|inde)\s+(?:\S+\s+){0,12}?(?:yer\s+)?"
    r"(?:almiyor|almamaktadir|bulunmuyor|bulunmamaktadir|yok\b)", re.IGNORECASE)
NOT_FOUND_SPAN = 300  # 'bulamadim' cevabin basinda aranir; govdedeki yan notlar sayilmaz
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
# Tirnak icindeki alinti; Turkce ek kesme isareti (KARARLAR'da) harften sonra gelir, alinti sanilmaz.
_QUOTED_RE = re.compile(r"(?<!\w)'[^'\n]{1,120}'(?!\w)|\"[^\"\n]{1,120}\"|“[^”\n]{1,120}”|(?<!\w)‘[^’\n]{1,120}’(?!\w)")
_PATH_RE = re.compile(r"[\w./\\-]+\.(?:md|txt|pdf|docx|html?)", re.IGNORECASE)


@dataclass
class AgentResult:
    answer: str
    steps: list = field(default_factory=list)  # [(arac, argumanlar)]
    evidence: list = field(default_factory=list)  # modele giden arac ciktilari; yargic bunlara bakar
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # prompt_tokens'in onbellekten okunan kismi (~%10 fiyat)
    seconds: float = 0.0

    @property
    def citations(self):
        """Koseli parantez icindeki belge yollari: [a.md], [a.md, b/c.md (satır 3)] ..."""
        return sorted({p for inner in _BRACKET_RE.findall(self.answer) for p in _PATH_RE.findall(inner)})

    @property
    def says_not_found(self):
        # Belgeden alintilanan ifade ("ŞİMDİLİK YOK" gibi) cevabin kendi hukmu sayilmaz.
        head = _QUOTED_RE.sub(" ", self.answer[:NOT_FOUND_SPAN])
        return bool(NOT_FOUND_RE.search(_ascii_fold(head)))


def _answer_text(response):
    """Dusunce olmayan metin parcalarini birlestirir; mode NONE iken gelen function_call yok sayilir."""
    candidate = (response.candidates or [None])[0]
    parts = getattr(getattr(candidate, "content", None), "parts", None) or []
    return "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()


def _ascii_fold(text):
    return text.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU"))


def tool_declarations():
    from google.genai import types

    def decl(name, description, properties, required=()):
        return types.FunctionDeclaration(name=name, description=description, parameters=types.Schema(
            type="OBJECT", properties={k: types.Schema(**v) for k, v in properties.items()}, required=list(required)))

    return [types.Tool(function_declarations=[
        decl("search", "Proje belgelerinde hibrit arama. Tek soru icin query, bagimsiz 2-4 soru icin queries.",
             {"query": {"type": "STRING"},
              "queries": {"type": "ARRAY", "items": types.Schema(type="STRING")},
              "path_prefix": {"type": "STRING", "description": "Klasor ya da dosya, orn. graf/"},
              "modified_after": {"type": "STRING", "description": "YYYY-AA-GG"}}),
        decl("get_document", "Bir belgenin tamamini getirir (uzunsa kesilir).",
             {"path": {"type": "STRING"}}, required=["path"]),
        decl("overview", "Projenin klasorleri, son degisen belgeleri ve notlari.", {}),
    ])]


class ResearchAgent:
    def __init__(self, service, project=None, model=DEFAULT_MODEL, client=None, max_steps=MAX_STEPS):
        self.service = service
        self.project = service.resolve_project(project)
        self.model = model
        self._client = client
        self.max_steps = max_steps

    def _gemini(self):
        if self._client is None:
            from dotenv import load_dotenv
            from google import genai

            from backend.shared.paths import PROJECT_ROOT

            load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
            self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        return self._client

    def run_tool(self, name, args):
        """Modelin istedigi araci calistirir; proje her zaman ajaninki, hata metin olarak doner."""
        args = {k: v for k, v in (args or {}).items() if v not in (None, "", [])}
        params = json.dumps(args, ensure_ascii=False)
        with tracing.span(f"tool.{name}", tracing.TOOL, **{"tool.name": name, "tool.parameters": params,
                                                           "input.value": tracing.clip(params)}) as span:
            text = self._run_tool(name, args)
            span.set("output.value", tracing.clip(text))
            return text

    def _run_tool(self, name, args):
        try:
            if name == "search":
                allowed = {"query", "queries", "path_prefix", "modified_after"}
                text = self.service.search(project=self.project, **{k: v for k, v in args.items() if k in allowed})
            elif name == "get_document":
                text = self.service.get_document(args.get("path", ""), project=self.project, max_chars=MAX_TOOL_CHARS)
            elif name == "overview":
                text = self.service.overview(project=self.project)
            else:
                text = f"Bilinmeyen arac: {name}"
        except (ValueError, KeyError, TypeError) as exc:
            text = f"Hata: {exc}"
        return text[:MAX_TOOL_CHARS]

    def ask(self, question):
        with tracing.span("agent.ask", tracing.AGENT, **{"input.value": tracing.clip(question),
                                                         "rag.project": self.project}) as span:
            result = self._ask(question)
            span.update({
                "output.value": tracing.clip(result.answer),
                "rag.tool_calls": len(result.steps),
                "rag.citations": result.citations,
                "rag.says_not_found": result.says_not_found,
                "llm.token_count.prompt": result.prompt_tokens,
                "llm.token_count.completion": result.output_tokens,
                "rag.cost_usd": cost_usd(result, self.model),
            })
            return result

    def _ask(self, question):
        from google.genai import types

        from backend.shared import gemini_retry, gemini_usage

        started = time.monotonic()
        result = AgentResult(answer="")
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        final = False
        for step in range(self.max_steps + 2):
            if not final and step >= self.max_steps:  # adim siniri: araclari kapat, eldekiyle cevaplat
                final = True
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=FINAL_NUDGE)]))
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.0, tools=tool_declarations(),
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                    mode="NONE" if final else "AUTO")),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            def _call():
                response = self._gemini().models.generate_content(model=self.model, contents=contents, config=config)
                gemini_usage.log_usage(CALL_SITE, self.model, response)
                return response

            with tracing.span("gemini", tracing.LLM, **{"llm.model_name": self.model, "rag.step": step,
                                                        "rag.tools_enabled": not final}) as span:
                response = gemini_retry.call_with_retry(_call, pacing_key=CALL_SITE)
                usage = response.usage_metadata
                if usage is not None:
                    result.prompt_tokens += usage.prompt_token_count or 0
                    result.output_tokens += usage.candidates_token_count or 0
                    span.update({"llm.token_count.prompt": usage.prompt_token_count,
                                 "llm.token_count.completion": usage.candidates_token_count,
                                 "llm.token_count.total": getattr(usage, "total_token_count", None)})
                calls = response.function_calls or []
                span.set("output.value", ", ".join(c.name for c in calls) if calls else tracing.clip(_answer_text(response)))
            if not calls or final:
                result.answer = _answer_text(response)
                if result.answer or final:
                    break
                # metinsiz STOP: bir kez araclari kapatip eldekiyle cevap iste
                final = True
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=FINAL_NUDGE)]))
                continue
            contents.append(response.candidates[0].content)
            parts = []
            for call in calls:
                args = dict(call.args or {})
                result.steps.append((call.name, args))
                output = self.run_tool(call.name, args)
                result.evidence.append(output)
                parts.append(types.Part.from_function_response(name=call.name, response={"result": output}))
            contents.append(types.Content(role="user", parts=parts))
        result.seconds = time.monotonic() - started
        return result


def cost_usd(result, model=DEFAULT_MODEL):
    from backend.shared.gemini_usage import _estimate_cost

    return _estimate_cost(model, int(result.prompt_tokens - 0.9 * result.cached_tokens), result.output_tokens)


def score_answer(record, result):
    """Altin set kaydi icin: cevaplanabilirse beklenen belge kaynak gosterilmis mi,
    cevapsizsa 'bulamadim' denmis mi."""
    if not record["expected_paths"]:
        return {"ok": result.says_not_found, "kind": "abstain"}
    cited = {c.replace("\\", "/").lower() for c in result.citations}
    hit = any(e.replace("\\", "/").lower() in cited for e in record["expected_paths"])
    return {"ok": hit and not result.says_not_found, "kind": "cite"}


def judge_record(client, record, result):
    """Cevaplanabilir soruda ve ajan bir cevap verdiyse yargica sorar; 'bulamadim' cevaplari ve
    cevapsiz sorular zaten score_answer ile olculuyor."""
    from backend.knowledge import judge

    if not record["expected_paths"] or result.says_not_found:
        return None
    reference = record.get("reference_answer") or record.get("expected_text")
    return judge.judge_answer(client, record["question"], result.evidence, result.answer, reference=reference)


def judge_summary(verdicts):
    judged = [v for v in verdicts if v and v["error"] is None]
    if not judged:
        return {}
    cost = sum(cost_usd(AgentResult("", prompt_tokens=v["prompt_tokens"], output_tokens=v["output_tokens"])) or 0
               for v in judged)
    return {"judged": len(judged), "judge_errors": sum(1 for v in verdicts if v and v["error"]),
            "grounded_rate": sum(v["grounded"] for v in judged) / len(judged),
            "correct_rate": sum(v["correct"] for v in judged) / len(judged),
            "judge_cost_usd": round(cost, 4)}


def _evaluate(service, project, limit, use_judge=False, golden_path=None, agent=None, mode="agent"):
    from backend.knowledge import eval as golden
    from backend.shared.paths import PROJECT_ROOT

    records = golden.load_golden_set(golden_path or os.path.join(PROJECT_ROOT, "knowledge_eval", f"{project}.jsonl"))
    records = records[:limit] if limit else records
    agent = agent or ResearchAgent(service, project=project)
    rows, totals = [], {"cite": [0, 0], "abstain": [0, 0]}
    cost, seconds, steps, verdicts = 0.0, [], [], []
    for record in records:
        result = agent.ask(record["question"])
        verdict = score_answer(record, result)
        totals[verdict["kind"]][0] += verdict["ok"]
        totals[verdict["kind"]][1] += 1
        cost += cost_usd(result) or 0.0
        seconds.append(result.seconds)
        steps.append(len(result.steps))
        judged = judge_record(agent._gemini(), record, result) if use_judge else None
        verdicts.append(judged)
        rows.append({"id": record["id"], "ok": verdict["ok"], "steps": result.steps, "citations": result.citations,
                     "answer": result.answer[:400], "judge": judged})
        mark = "" if not judged else f" grounded={judged['grounded']} correct={judged['correct']}"
        print(f"{'OK' if verdict['ok'] else '--'} {record['id']} adim={len(result.steps)} "
              f"kaynak={result.citations[:3]} {result.seconds:.1f}s{mark}", flush=True)
    summary = {
        "questions": len(records),
        "citation_accuracy": totals["cite"][0] / max(1, totals["cite"][1]),
        "abstain_accuracy": totals["abstain"][0] / max(1, totals["abstain"][1]),
        "mean_tool_calls": sum(steps) / max(1, len(steps)),
        "median_seconds": sorted(seconds)[len(seconds) // 2] if seconds else 0,
        "total_cost_usd": round(cost, 4),
        **judge_summary(verdicts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = os.path.join(PROJECT_ROOT, "artifacts", "knowledge_eval", f"{project}_agent.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": DEFAULT_MODEL, "mode": mode,
                            "summary": summary, "rows": rows}, ensure_ascii=False) + "\n")


def main():
    from backend.knowledge.runtime import Runtime
    from backend.knowledge.service import KnowledgeService

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--project")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge", action="store_true", help="cevaplari LLM yargica da puanlat (grounded/correct)")
    parser.add_argument("--golden-set", default=None)
    parser.add_argument("--long-context", action="store_true",
                        help="arama yerine projenin tum belgelerini tek seferde modele ver (taban cizgisi)")
    args = parser.parse_args()
    service = KnowledgeService(Runtime())
    tracing.init_tracing("rag-agent-eval" if args.eval else "rag-agent")
    try:
        return _run_cli(service, args, parser)
    finally:
        tracing.flush()


def _long_context_reader(service, project):
    from backend.knowledge import goldgen
    from backend.knowledge.longctx import MAX_CORPUS_TOKENS, LongContextReader

    reader = LongContextReader(goldgen.load_documents(service.runtime.client(), project))
    tokens = reader.corpus_tokens()
    full = cost_usd(AgentResult("", prompt_tokens=tokens)) or 0
    print(f"Uzun baglam: {len(reader.docs)} belge, {tokens} token; soru basina ~${full:.3f} "
          f"(onbellekle ~${full / 10:.4f} + saatlik onbellek ucreti)", file=sys.stderr)
    if tokens > MAX_CORPUS_TOKENS:
        raise SystemExit(f"Korpus {tokens} token, sinir {MAX_CORPUS_TOKENS}: bu proje tek baglama sigmiyor.")
    return reader


def _run_cli(service, args, parser):
    if args.long_context:
        project = args.project or service.resolve_project(None)
        reader = _long_context_reader(service, project)
        try:
            if args.eval:
                return _evaluate(service, project, args.limit, use_judge=args.judge, golden_path=args.golden_set,
                                 agent=reader, mode="long-context")
            if not args.question:
                parser.error("soru ya da --eval gerekli")
            result = reader.ask(args.question)
            print(result.answer)
            print(f"\n[{result.seconds:.1f} sn, {result.prompt_tokens}+{result.output_tokens} token "
                  f"({result.cached_tokens} onbellekten), ~${cost_usd(result) or 0:.4f}]", file=sys.stderr)
            return None
        finally:
            reader.close()
    if args.eval:
        return _evaluate(service, args.project or service.resolve_project(None), args.limit,
                         use_judge=args.judge, golden_path=args.golden_set)
    if not args.question:
        parser.error("soru ya da --eval gerekli")
    result = ResearchAgent(service, project=args.project).ask(args.question)
    for name, call_args in result.steps:
        print(f"> {name}({json.dumps(call_args, ensure_ascii=False)})", file=sys.stderr)
    print(result.answer)
    print(f"\n[{len(result.steps)} arac cagrisi, {result.seconds:.1f} sn, "
          f"{result.prompt_tokens}+{result.output_tokens} token, ~${cost_usd(result) or 0:.4f}]", file=sys.stderr)


if __name__ == "__main__":
    main()
