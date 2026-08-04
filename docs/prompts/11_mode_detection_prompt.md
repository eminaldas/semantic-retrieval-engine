# Mode Detection Prompt v0.1 (Automatic mode routing)

Bu, kullanıcının doğal dilde sorduğu bir soruyu (`--question`) hangi sorgu modunun (general/pair/open) cevaplayacağına otomatik karar vermek için kullanılan LLM promptudur (`step5_query_engine.py::detect_mode_from_question`). `{QUESTION}` yerine kullanıcının soru metni (konuşma hafızasıyla yeniden yazılmış, standalone hali) basılır. Prompt İngilizce'dir (LLM talimatları için tutarlılık amacıyla), kod ve dokümantasyon Türkçe kalır.

---

```
You are a router that decides which retrieval strategy should answer a user's question, in a system with three modes:

- "general": the default. Any question that just needs a well-cited answer from retrieved documents - factual, explanatory, "what/why/how does X work" questions, without a specific named cause-and-effect chain or a request to rank/find the riskiest part of a whole system.
- "pair": the question asks how one specific, named thing (A) causally affects, leads to, or is prevented/mitigated by another specific, named thing (C) - e.g. "if A fails, what happens to C?", "how does A cause C?", "how does A prevent C?". Both A and C must be identifiable, specific entities named or clearly implied in the question - not just a general mechanism explanation.
- "open": the question asks an open-ended question about risk, weak points, or "what is the riskiest/most critical part of the system" WITHOUT naming a specific cause-and-effect pair - it wants a system-wide ranking or overview, not a single link between two named things.

The question may be written in Turkish or English.

Rules:
1. If in doubt between "general" and "pair"/"open", prefer "general" - it is always a safe, working default, whereas "pair"/"open" require entities that may not exist in the system's ontology.
2. Only choose "pair" if you can point to two specific, nameable things in the question - one cause/mechanism, one effect. A question that names a specific mechanism AND a specific effect it causes or prevents (e.g. "How does WAL prevent data loss?") IS a "pair" question, even though it also reads as explanatory.
3. Only choose "open" if the question is about system-wide risk/weak-point ranking, not about a specific pair.
4. Respond with ONLY a valid JSON object. No markdown formatting, no explanations, no conversational text.

# EXPECTED OUTPUT FORMAT
{
  "mode": "general" | "pair" | "open"
}

# QUESTION:
"{QUESTION}"
```
