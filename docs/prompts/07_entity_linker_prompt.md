# Entity Linker Prompt v0.1 (Entity Linking module)

Bu, kullanıcının doğal dilde sorduğu bir soruyu (`--question`) A (cause) ve C (target) varlıklarına otomatik olarak çözmek için kullanılan LLM promptudur (`step5b_entity_linker.py::call_entity_linker`). `{QUESTION}` yerine kullanıcının soru metni basılır. Prompt İngilizce'dir (LLM talimatları için tutarlılık amacıyla), kod ve dokümantasyon Türkçe kalır.

---

```
You are an expert at parsing cause-and-effect questions about system architecture and failure analysis.

The question may be written in Turkish or English. Identify two phrases in it:
- cause_phrase: the exact phrase referring to the entity that is the CAUSE (the "A" in "if A happens, what happens to C") - the thing that fails, changes, or acts first.
- target_phrase: the exact phrase referring to the entity that is the TARGET/EFFECT (the "C") - the thing that is affected as a consequence.

Rules:
1. Extract phrases as close to verbatim from the question as possible. Do not paraphrase into different wording, but you may drop grammatical case-suffixes that are not part of the entity's name (e.g. Turkish "-i"/"-ini" object suffixes).
2. Do not include verbs, conditionals, or question words in the phrases - only the entity-referring noun phrase itself.
3. Never leave a field empty - if the question only clearly names one entity, infer the most plausible second one from context.
4. Respond with ONLY a valid JSON object. No markdown formatting, no explanations, no conversational text.

# EXPECTED OUTPUT FORMAT
{
  "cause_phrase": "string",
  "target_phrase": "string"
}

# QUESTION:
"{QUESTION}"
```
