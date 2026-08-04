# Conversation Summary Prompt v0.1 (Conversation Memory module)

Bir konusma turu, aktif pencereden (son WINDOW_SIZE tur) dusunce, tamamen
atilmak yerine mevcut ozete katilir (`step5d_conversation_memory.py::fold_turn_into_summary`).
`{EXISTING_SUMMARY}` yerine mevcut ozet (veya "(henuz ozet yok)"), `{QUESTION}`/`{ANSWER}`
yerine dusen turun soru/cevap metni basilir. Prompt Ingilizce'dir, kod ve
dokumantasyon Turkce kalir.

---

```
You are maintaining a running summary of an ongoing conversation, so that older turns are not lost even after they leave the active context window.

# EXISTING SUMMARY
{EXISTING_SUMMARY}

# TURN TO FOLD IN (leaving the active window)
Question: {QUESTION}
Answer: {ANSWER}

# TASK
Produce an updated summary that incorporates any information from this turn that could matter for understanding future follow-up questions (entities discussed, facts established, conclusions reached). Keep it concise - a few sentences, cumulative with the existing summary, not a transcript.

Rules:
1. Preserve the language of the EXISTING SUMMARY (or the TURN if the summary is empty).
2. Do not invent information not present in the EXISTING SUMMARY or the TURN.
3. Respond with ONLY a valid JSON object. No markdown formatting, no explanations, no conversational text.

# EXPECTED OUTPUT FORMAT
{
  "summary_text": "string"
}
```
