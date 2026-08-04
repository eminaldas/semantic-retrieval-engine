# Conversation Rewrite Prompt v0.1 (Conversation Memory module)

Bu, bir takip sorusunu (follow-up question) onceki konusma baglamiyla kendi kendine
yeten (standalone) bir soruya donusturmek icin kullanilan LLM promptudur
(`step5d_conversation_memory.py::rewrite_question`). `{CONVERSATION_CONTEXT}` yerine
onceki konusmanin ozeti + son turlari, `{QUESTION}` yerine kullanicinin yeni sorusu
basilir. Bu adim SADECE soruyu yeniden yazmak icindir - rewrite edilmis soru daha
sonra normal (gecmisten habersiz) retrieval+Answerer zincirinden gecer, boylece
Answerer hicbir zaman ham konusma gecmisini gormez (docs/01_architecture_v0.1.md'nin
"sadece retrieved evidence" ilkesi korunur). Prompt Ingilizce'dir, kod ve
dokumantasyon Turkce kalir.

---

```
You are rewriting a follow-up question so it can be understood on its own, without needing the conversation history below.

# CONVERSATION HISTORY
{CONVERSATION_CONTEXT}

# NEW QUESTION
"{QUESTION}"

# TASK
Rewrite the NEW QUESTION into a standalone question that a person with NO access to the conversation history could still fully understand - resolve pronouns, ellipsis, and implicit references (e.g. "the same thing", "what about X") using the conversation history above.

Rules:
1. Preserve the original language of the NEW QUESTION (Turkish stays Turkish, English stays English).
2. Preserve the original intent exactly - do not add new questions, do not answer the question, do not add information not implied by the NEW QUESTION itself.
3. If the NEW QUESTION is already fully self-contained (does not depend on the conversation history at all), return it unchanged.
4. Respond with ONLY a valid JSON object. No markdown formatting, no explanations, no conversational text.

# EXPECTED OUTPUT FORMAT
{
  "rewritten_question": "string"
}
```
