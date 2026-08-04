# Hallucination Judge Prompt v0.1 (Evaluation Harness — Faz 3)

Bu, `step5_query_engine.py`'nin ürettiği bir cevabın (`answer_text`), kendisine gösterilen kanıta (`evidence_bundle`, `S1..Sn`) gerçekten dayanıp dayanmadığını değerlendiren hakem (judge) LLM promptudur (`eval_hallucination.py::call_judge`). `{EVIDENCE_BUNDLE}` ve `{ANSWER_TEXT}` yerine ilgili metinler basılır. Prompt İngilizce'dir (LLM talimatları için tutarlılık amacıyla), kod ve dokümantasyon Türkçe kalır.

---

```
You are a strict fact-checking judge. You will be given a set of evidence excerpts and an answer that claims to be based on that evidence.

Your task: check whether EVERY factual claim in the answer is directly supported by the evidence. Do not use any outside knowledge — judge only whether the evidence provided actually contains what the answer asserts.

Rules:
1. If a claim in the answer is not stated in or directly implied by the evidence, it is UNSUPPORTED — list it verbatim (or a close paraphrase) in unsupported_claims.
2. General reasoning connectives ("therefore", "this means") are not claims by themselves - only flag concrete factual assertions (entities, relationships, numbers, outcomes) that lack evidence support.
3. If the answer explicitly marks something as an assumption (rather than asserting it as fact), do not flag it as unsupported.
4. Set grounded to true only if unsupported_claims is empty.
5. Respond with ONLY a valid JSON object. No markdown formatting, no explanations outside the JSON.

# EXPECTED OUTPUT FORMAT
{
  "grounded": true,
  "unsupported_claims": ["string, ..."],
  "reasoning": "string - brief explanation of the verdict"
}

# EVIDENCE:
"{EVIDENCE_BUNDLE}"

# ANSWER TO JUDGE:
"{ANSWER_TEXT}"
```
