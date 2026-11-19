# sec-mcp-agent

[![tests](https://github.com/DiegoTDDD/sec-mcp-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/DiegoTDDD/sec-mcp-agent/actions/workflows/tests.yml)
[![eval](https://github.com/DiegoTDDD/sec-mcp-agent/actions/workflows/eval.yml/badge.svg)](https://github.com/DiegoTDDD/sec-mcp-agent/actions/workflows/eval.yml)

Two MCP servers, one over SEC 10-K filings and one over a fraud model, and a tool-calling
agent that answers from them with **verified citations**. Everything runs locally and free: a
3B model served by Ollama, BM25 search, and the XGBoost model from
[realtime-fraud-detection-ml](https://github.com/DiegoTDDD/realtime-fraud-detection-ml). The
same servers plug into Claude Desktop.

## Problem

LLM answers about filings are only useful if every claim can be traced to the source, and
small models invent sources that look real. Tools are also usually wired into one agent, so
nothing else can use them.

## Solution

```
                 ┌──────────────────────────── MCP (stdio or in-memory) ───────────────────────────┐
question ──► Agent ──► tool calls ──► sec-filings server     search_filings · get_chunk · list_companies
   │            │                     (BM25 over 10-K chunks)  filing://{ticker}/{section} · risk_review prompt
   │            │
   │            └────► tool calls ──► fraud-scoring server   score_transaction (+ TreeSHAP reasons)
   │                                  (XGBoost, same features   fraud://model-card
   │                                   as training, by test)
   ▼
JSON answer (schema-constrained) ──► citation verifier ──► AgentAnswer
                                     chunk_id retrieved?      answer, verified citations,
                                     quote word for word?     rejected citations, grounded,
                                     one repair round         fraud scores, tool trace, tokens
```

- **MCP servers** built on the official Python SDK 2.x, with tools, resources and a prompt;
  every tool is read-only and returns structured output.
- **Agent** discovers tools from every server, loops on tool calls (step limit), then produces
  the final answer under a JSON schema (Pydantic).
- **Citation verifier**: a citation counts only if its passage was actually retrieved in this
  run and the quote is in it word for word; otherwise the model gets one repair round, and what
  still fails is reported, never passed through.
- **Fraud tool parity**: the model is trained by the fraud project's own code at a pinned
  commit, and a check rebuilds 2,000 training rows as tool calls: features match to 2e-13 and
  probabilities to 5e-7. It caught a sub-second mismatch in the tool's first version (ADR 0003).

## Results

<!-- EVAL:START -->
![Demo: the agent answers from 10-K filings with verified citations](docs/demo.gif)

The demo is drawn from two answers recorded in the final evaluation run, not staged
([`scripts/render_demo.py`](scripts/render_demo.py); [MP4](docs/demo.mp4)).

Eval run [44542469896](https://github.com/DiegoTDDD/sec-mcp-agent/actions/runs/44542469896):
qwen2.5:3b through Ollama on a CPU-only GitHub runner, temperature 0, 22 questions
([`eval/results.md`](eval/results.md), one transcript per question).

| Metric | Value |
| --- | --- |
| **Questions passed** | **17 of 22** |
| Filings questions passed (right company, every citation verified, expected evidence cited) | 12 of 16 (75%) |
| Citation precision (citations on the company asked about) | 100% |
| Out-of-scope companies answered without citing anything | 2 of 2 |
| Fraud questions scored with exactly the stated transaction | 3 of 4 |
| Quotes copied exactly by the model / selected by the verifier from the cited passage | 15 / 16 |
| Median / p90 latency per question | 97 s / 144 s |

### Six runs: what each one exposed

Every run's results and transcripts are kept in [`eval/history/`](eval/history/). Each fix
came from reading the transcripts, not from changing the questions.

| Run | Passed | What the transcripts showed | Fix |
| --- | --- | --- | --- |
| [1](eval/history/run1/results.md) | 7 / 22 | Most tool calls failed validation: the model wrote "RISK FACTORS" or "MD&A" for an enum, copied a nested JSON schema into the fraud tool, answered 3 questions from memory; TSLA missing (its latest 10-K was an amendment) | Lenient section and company names, flat tool parameters, one reminder to use tools, skip 10-K/A ([cacf91e](https://github.com/DiegoTDDD/sec-mcp-agent/commit/cacf91e)) |
| 2 | 8 of 14 before the job limit | One JSON-constrained reply never ended and held the runner for 2.5 hours | Token cap, HTTP timeout, a failed question no longer stops the run |
| [3](eval/history/run3/results.md) | 11 / 22 | Right passages, but the model could not copy a sentence exactly and dropped citations when asked to fix them; searches without a ticker drifted to other companies | Citations anchored on the passage id, quote selected from the passage when inexact; company detected in the query ([7b7cbc4](https://github.com/DiegoTDDD/sec-mcp-agent/commit/7b7cbc4)) |
| [4](eval/history/run4/results.md) | 11 / 22 | The whole answer object written as a string inside `answer` (8 failures); searching other companies after "not in the index" | Unwrap nested JSON; stop on unknown companies ([4067bf7](https://github.com/DiegoTDDD/sec-mcp-agent/commit/4067bf7)) |
| [5](eval/history/run5/results.md) | 15 / 22 | Nested JSON that does not even parse (mixed quotes, cut-off strings) | Salvage chunk ids and answer text ([80086e0](https://github.com/DiegoTDDD/sec-mcp-agent/commit/80086e0)) |
| **6** | **17 / 22** | | |
<!-- EVAL:END -->

## Use it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

export SEC_IDENTITY="Your Name you@example.com"   # SEC requires it of automated clients
python scripts/build_index.py                     # data/chunks.jsonl: 20 companies, latest 10-K
bash scripts/build_fraud_model.sh                 # models/fraud/ + feature parity check

ollama pull qwen2.5:3b
python -m sec_mcp_agent "What does NVIDIA report about US export controls?"
python -m sec_mcp_agent "Card card_0001 spent R$ 4,850 on electronics in Manaus at 2026-03-01T03:12:00. Fraud?"
python scripts/run_eval.py                        # eval/results.md
```

**Claude Desktop**: copy `claude_desktop_config.example.json` into Claude Desktop's config,
replacing the paths. Both servers then appear as tools, the `filing://` resources can be
attached to a chat, and `risk_review` shows up as a prompt.

## Tests

`pytest -q -p anyio`: 62 tests. Every tool goes through a real MCP client (listing, schemas,
structured results, validation errors); fraud features follow the training definitions
(window edges, second truncation, unknown inputs); the agent loop runs with a scripted model
through tool errors, unknown tools, the step limit, invalid JSON and fabricated or repaired
citations. CI also starts both servers over stdio.

## Design decisions

- [ADR 0001](docs/adr/0001-mcp-servers-as-the-tool-boundary.md): tools as MCP servers, not functions inside the agent.
- [ADR 0002](docs/adr/0002-bm25-search.md): BM25 inside the server instead of the hybrid retriever.
- [ADR 0003](docs/adr/0003-fraud-tool-reuses-the-training-features.md): the fraud tool reproduces the training features, proven by a parity check.
- [ADR 0004](docs/adr/0004-verify-citations.md): verify every citation against retrieved passages.
- [ADR 0005](docs/adr/0005-local-model-with-ollama.md): a local model through Ollama behind a one-method interface.

## Limits

- The verifier proves a quote exists in a retrieved passage, not that it supports the claim.
- BM25 misses paraphrases; the agent retries with other words, and the eval shows where that
  is not enough.
- A 3B model on a CPU is slow (tens of seconds per question) and makes tool-use mistakes; the
  eval reports them.
- The fraud model is only as good as its training: it under-flags impossible travel when a
  card has no recent history, and the tool reports that honestly with its reasons.
- The eval checks the fraud tool's arguments, not the wording of the answer: in run 6 one answer
  says the model "flags" a purchase it did not flag (t04), and on fraud questions the model
  sometimes lists feature names as citations, which the verifier rejects (`grounded: false`).
  Checking that the answer text matches the tool result is the job of an answer-faithfulness
  metric, planned in llm-evals-observability.

## Stack

Python 3.11, MCP Python SDK 2.x, Pydantic, Ollama (qwen2.5:3b), rank-bm25, XGBoost,
edgartools, pytest, ruff, GitHub Actions.

## License

MIT
