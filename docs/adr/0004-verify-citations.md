# ADR 0004: Verify every citation against retrieved passages

**Status:** accepted

## Context
Small local models format citations convincingly whether or not the quote is real. A citation
the reader cannot trust is worse than none.

## Decision
- The final answer is constrained to a JSON schema (`AnswerDraft`: answer + citations of
  `chunk_id` and `quote`) through Ollama's structured output.
- The agent records every passage the tools returned. A citation is accepted only if its
  `chunk_id` was returned in this run and its quote appears in that passage word for word
  (whitespace and typographic quotes normalised, at least 20 characters).
- If any citation fails, the model gets one repair round with the reasons. Citations that still
  fail are returned in `rejected_citations` and the answer is marked `grounded: false`.
- An answer that used filings tools but cites nothing is also not grounded.

## Consequences
- `grounded` means checked, not that the model said so. The eval reports it per question.
- One extra model call when a repair is needed; the eval counts how often.
- The check is lexical: it proves the quote exists, not that it supports the claim. Judging
  support needs a semantic evaluation, which is the job of the llm-evals-observability project.
