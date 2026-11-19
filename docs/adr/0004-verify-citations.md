# ADR 0004: Verify every citation against retrieved passages

**Status:** accepted

## Context
Small local models format citations convincingly whether or not the quote is real. A citation
the reader cannot trust is worse than none.

## Decision
- The final answer is constrained to a JSON schema (`AnswerDraft`: answer + citations of
  `chunk_id` and `quote`) through Ollama's structured output.
- The agent records every passage the tools returned in the run. A citation is accepted only if
  its `chunk_id` is one of them. An invented `chunk_id` gets one repair round with the ids that
  failed; what still fails goes to `rejected_citations` and the answer is `grounded: false`.
- The quote shown to the reader is always real text of the cited passage: the model's quote if
  it appears there word for word (`quote_source: model`), otherwise the passage sentence that
  shares the most search terms with the answer (`quote_source: selected`).
- An answer that used the filings tools but cites nothing is not grounded.

## History
The first version also required the model's quote to match word for word and sent the answer
back otherwise. Eval run 3 (`eval/history/run3`) showed the cost: qwen2.5:3b picked the right
passage far more often than it copied a sentence exactly, and when asked to fix a quote it
dropped the citation. Five answers with correct content ended with no citations. Anchoring
on the passage id keeps the guarantee that matters (every citation points at text the tools
really returned, and every quote is that text) without depending on a 3B model's copying.

## Consequences
- `grounded` means checked, not that the model said so. The eval reports it per question,
  and how many quotes were the model's own versus selected.
- The check proves the cited passage was retrieved and the quote is in it, not that the passage
  supports the claim. Judging support needs a semantic evaluation, which is the job of the
  llm-evals-observability project.
