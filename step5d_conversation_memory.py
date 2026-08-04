"""Konusma hafizasi: kalici gecmis + kayan pencere + ozet mekanizmasi.
step5b_entity_linker.py/step5c_risk_ranking.py ile ayni konvansiyon - bagimsiz,
hem CLI (step5_query_engine.py) hem dashboard.py tarafindan import edilir.
docs/superpowers/specs/2026-08-02-conversation-memory-design.md."""

import json
import os
from datetime import datetime, timezone

from google.genai import types
from pydantic import BaseModel

from backend.shared import gemini_retry
from backend.shared import gemini_usage

DEFAULT_MODEL = "gemini-flash-lite-latest"  # step2/step5b ile ayni varsayilan
WINDOW_SIZE = 5  # ham tutulan son tur sayisi - bunu asan turlar ozete katilir


def load_history(history_file):
    """conversation_history.jsonl'i okur. Dosya yoksa bos liste doner."""
    if not os.path.exists(history_file):
        return []
    turns = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def load_summary_state(summary_file):
    """conversation_summary.json'i okur. Dosya yoksa/bozuksa varsayilan (bos) state doner."""
    if not os.path.exists(summary_file):
        return {"summary_text": "", "summarized_through_turn_id": 0}
    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"summary_text": "", "summarized_through_turn_id": 0}


def next_turn_id(history):
    """Bir sonraki turn_id - mevcut en yuksek turn_id + 1 (gecmis bossa 1)."""
    if not history:
        return 1
    return max(t["turn_id"] for t in history) + 1


def select_window_turns(history, summarized_through_turn_id):
    """Henuz ozete katilmamis (turn_id > summarized_through_turn_id) turlari,
    turn_id sirasiyla dondurur."""
    return sorted(
        (t for t in history if t["turn_id"] > summarized_through_turn_id),
        key=lambda t: t["turn_id"],
    )


def build_conversation_context(summary_text, window_turns):
    """Rewrite prompt'una enjekte edilecek metni kurar. Hem summary_text hem
    window_turns bossa "" doner - bu, prepare_question'in LLM cagrisini
    tamamen atlamasi icin kullanilan sinyal (ilk soruda gereksiz maliyet yok)."""
    if not summary_text and not window_turns:
        return ""
    parts = []
    if summary_text:
        parts.append(f"Onceki konusmanin ozeti: {summary_text}")
    for turn in window_turns:
        parts.append(f"S: {turn['question']}\nC: {turn['answer_text']}")
    return "\n\n".join(parts)


def turns_to_fold(window_turns, window_size=WINDOW_SIZE):
    """window_turns turn_id'ye gore artan sirali kabul edilir (select_window_turns
    zaten bu sirada dondurur). window_size'i asan en eski turlari (oldest-first)
    dondurur - bunlar ozete katilacak, pencerede kalmayacak turlar."""
    if len(window_turns) <= window_size:
        return []
    return window_turns[:len(window_turns) - window_size]


def append_turn(history_file, turn_record):
    """conversation_history.jsonl'a tek bir tur ekler (append, hic satir degistirilmez)."""
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn_record, ensure_ascii=False) + "\n")


def save_summary_state(summary_file, summary_text, summarized_through_turn_id):
    """conversation_summary.json'un uzerine yazar (tek obje, append degil)."""
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {"summary_text": summary_text, "summarized_through_turn_id": summarized_through_turn_id},
            f, ensure_ascii=False,
        )


def clear_conversation(history_file, summary_file):
    """Kalici konusma gecmisini ve ozetini sifirlar. Dosyalar yoksa da hata vermez."""
    for path in (history_file, summary_file):
        if os.path.exists(path):
            os.remove(path)


class RewrittenQuestion(BaseModel):
    rewritten_question: str


class SummaryUpdate(BaseModel):
    summary_text: str


def load_prompt_template(prompt_path):
    """docs/prompts/*.md icindeki fenced code block'u okur (step2/step5/step5b ile ayni sozlesme)."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.index("```\n") + len("```\n")
    end = content.rindex("```")
    return content[start:end].strip()


def rewrite_question(client, model, prompt_template, conversation_context, question):
    """conversation_context bossa (ilk soru) LLM'e hic gitmeden question'i degismeden dondurur."""
    if not conversation_context:
        return question

    prompt = prompt_template.replace("{CONVERSATION_CONTEXT}", conversation_context).replace("{QUESTION}", question)

    def _call():
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RewrittenQuestion,
            ),
        )
        gemini_usage.log_usage("step5d_rewrite_question", model, response)
        result = response.parsed or RewrittenQuestion.model_validate_json(response.text)
        return result.rewritten_question

    return gemini_retry.call_with_retry(_call, pacing_key="step5d_rewrite_question")


def fold_turn_into_summary(client, model, prompt_template, existing_summary, turn):
    prompt = (
        prompt_template
        .replace("{EXISTING_SUMMARY}", existing_summary or "(henuz ozet yok)")
        .replace("{QUESTION}", turn["question"])
        .replace("{ANSWER}", turn["answer_text"])
    )

    def _call():
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SummaryUpdate,
            ),
        )
        gemini_usage.log_usage("step5d_fold_summary", model, response)
        result = response.parsed or SummaryUpdate.model_validate_json(response.text)
        return result.summary_text

    return gemini_retry.call_with_retry(_call, pacing_key="step5d_fold_summary")


def prepare_question(history_file, summary_file, question, client, model, rewrite_prompt_template):
    """Gecmisi okur, baglami kurar, soruyu yeniden yazar. rewrite_question basarisiz
    olursa (3 denemeden sonra) orijinal question'a sessizce duser - hafiza bir eklenti,
    basarisizligi cagiran script'i cokertmemeli."""
    history = load_history(history_file)
    summary_state = load_summary_state(summary_file)
    window_turns = select_window_turns(history, summary_state["summarized_through_turn_id"])
    context = build_conversation_context(summary_state["summary_text"], window_turns)

    if not context:
        return question

    try:
        return rewrite_question(client, model, rewrite_prompt_template, context, question)
    except Exception as exc:
        print(f"UYARI: soru yeniden yazma basarisiz oldu, orijinal soru kullanilacak: {exc}")
        return question


def record_turn(history_file, summary_file, turn_record, client, model, summary_prompt_template):
    """turn_record'a turn_id/timestamp atar, gecmise ekler, pencere tastiysa
    en eski tur(lar)i ozete katar. fold_turn_into_summary basarisiz olursa ozet
    guncellenmez (eski ozet korunur) - turn yine de eklenmis olur, kayip yok."""
    history = load_history(history_file)
    turn_record = dict(turn_record)
    turn_record["turn_id"] = next_turn_id(history)
    turn_record["timestamp"] = datetime.now(timezone.utc).isoformat()
    append_turn(history_file, turn_record)

    history.append(turn_record)
    summary_state = load_summary_state(summary_file)
    window_turns = select_window_turns(history, summary_state["summarized_through_turn_id"])
    to_fold = turns_to_fold(window_turns)

    summary_text = summary_state["summary_text"]
    summarized_through = summary_state["summarized_through_turn_id"]
    for turn in to_fold:
        try:
            summary_text = fold_turn_into_summary(client, model, summary_prompt_template, summary_text, turn)
            summarized_through = turn["turn_id"]
        except Exception as exc:
            print(f"UYARI: konusma ozetleme basarisiz oldu, ozet guncellenmedi: {exc}")
            break

    if summarized_through != summary_state["summarized_through_turn_id"]:
        save_summary_state(summary_file, summary_text, summarized_through)
