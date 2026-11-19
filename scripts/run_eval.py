"""Run the agent on eval/questions.json with a local Ollama model and score every answer.

Filings questions   searched the right company; every citation verified; citations are on the
                    right company; a cited passage contains the expected evidence.
Out-of-scope        the company is not indexed: the agent must not cite anything.
Fraud questions     the agent passed the exact transaction to score_transaction.

Writes eval/results.json, eval/results.md and eval/transcripts/<id>.txt.

    python scripts/run_eval.py --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import anyio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sec_mcp_agent.__main__ import render  # noqa: E402
from sec_mcp_agent.agent.agent import Agent  # noqa: E402
from sec_mcp_agent.agent.llm import OllamaChat  # noqa: E402
from sec_mcp_agent.agent.schema import AgentAnswer  # noqa: E402
from sec_mcp_agent.filings.corpus import load_chunks  # noqa: E402
from sec_mcp_agent.filings.search import FilingsIndex  # noqa: E402
from sec_mcp_agent.filings.server import build_server as filings_server  # noqa: E402
from sec_mcp_agent.fraud.model import FraudModel, Transaction  # noqa: E402
from sec_mcp_agent.fraud.server import build_server as fraud_server  # noqa: E402

EVAL = ROOT / "eval"


def score_filings(q: dict, a: AgentAnswer, index: FilingsIndex) -> dict[str, Any]:
    searched = any(
        t.tool == "search_filings"
        and t.ok
        and (t.arguments.get("ticker") or "").upper() == q["ticker"]
        for t in a.tool_calls
    )
    on_target = [c for c in a.citations if c.ticker == q["ticker"]]
    texts = [index.by_id[c.chunk_id].text.lower() for c in on_target]
    evidence = any(kw in text for kw in q["evidence"] for text in texts)
    return {
        "searched_company": searched,
        "grounded": a.grounded,
        "cited": bool(a.citations),
        "citations_on_target": len(on_target) / len(a.citations) if a.citations else None,
        "evidence_found": evidence,
        "passed": a.grounded
        and bool(on_target)
        and len(on_target) == len(a.citations)
        and evidence,
    }


def score_out_of_scope(a: AgentAnswer) -> dict[str, Any]:
    return {"cited": bool(a.citations), "passed": not a.citations and not a.rejected_citations}


def score_fraud(q: dict, a: AgentAnswer, model: FraudModel) -> dict[str, Any]:
    expected = Transaction(**q["transaction"])
    calls = [t for t in a.tool_calls if t.tool == "score_transaction" and t.ok]
    exact = False
    for call in calls:
        try:
            args = call.arguments.get("transaction", call.arguments)
            got = Transaction(**{k: v for k, v in args.items() if v is not None})
        except Exception:
            continue
        exact = exact or got == expected
    truth = model.score(expected)
    return {
        "called_tool": bool(calls),
        "exact_arguments": exact,
        "model_flagged": truth.flagged,
        "model_probability": truth.probability,
        "passed": exact,
    }


def failed_row(q: dict, exc: Exception, seconds: float) -> dict[str, Any]:
    return {
        "id": q["id"], "type": q["type"],
        "score": {"passed": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"},
        "seconds": round(seconds, 1), "tools": "-", "tool_calls": 0, "citations": 0,
        "rejected_citations": 0, "repaired": False, "llm_calls": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "answer": None,
    }  # fmt: skip


def summarize(rows: list[dict]) -> dict[str, Any]:
    def rate(kind: str, key: str) -> float | None:
        vals = [
            r["score"].get(key, False if "error" in r["score"] else None)
            for r in rows
            if r["type"] == kind
        ]
        vals = [v for v in vals if v is not None]
        return round(sum(bool(v) for v in vals) / len(vals), 3) if vals else None

    filings = [r for r in rows if r["type"] == "filings"]
    on_target = [
        r["score"]["citations_on_target"]
        for r in filings
        if r["score"].get("citations_on_target") is not None
    ]
    return {
        "questions": len(rows),
        "errors": sum("error" in r["score"] for r in rows),
        "passed": sum(r["score"]["passed"] for r in rows),
        "filings_pass_rate": rate("filings", "passed"),
        "searched_company_rate": rate("filings", "searched_company"),
        "grounded_rate": rate("filings", "grounded"),
        "evidence_rate": rate("filings", "evidence_found"),
        "citation_precision": round(statistics.mean(on_target), 3) if on_target else None,
        "rejected_citations": sum(r["rejected_citations"] for r in rows),
        "repaired_answers": sum(r["repaired"] for r in rows),
        "out_of_scope_pass_rate": rate("out_of_scope", "passed"),
        "fraud_exact_arguments_rate": rate("fraud", "exact_arguments"),
        "median_seconds": round(statistics.median(r["seconds"] for r in rows), 1),
        "p90_seconds": round(sorted(r["seconds"] for r in rows)[round(0.9 * (len(rows) - 1))], 1),
        "mean_tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 2),
        "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
    }


def write_markdown(summary: dict, rows: list[dict], model: str, path: Path) -> None:
    pct = lambda v: "n/a" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    lines = [
        "# Agent evaluation", "",
        f"Model `{model}` served by Ollama on a CPU-only GitHub Actions runner, temperature 0. "
        f"{summary['questions']} questions; {summary['passed']} passed; "
        f"{summary['errors']} did not finish (model timeout or error).", "",
        "| Metric | Value |", "| --- | --- |",
        f"| Filings questions passed (right company, verified citations, expected evidence) | {pct(summary['filings_pass_rate'])} |",
        f"| Searched the right company | {pct(summary['searched_company_rate'])} |",
        f"| Grounded (every citation verified against a retrieved passage) | {pct(summary['grounded_rate'])} |",
        f"| Cited passage contains the expected evidence | {pct(summary['evidence_rate'])} |",
        f"| Citation precision (citations on the asked company) | {pct(summary['citation_precision'])} |",
        f"| Out-of-scope questions answered without citing anything | {pct(summary['out_of_scope_pass_rate'])} |",
        f"| Fraud questions scored with exactly the stated transaction | {pct(summary['fraud_exact_arguments_rate'])} |",
        f"| Answers sent back once by the citation verifier | {summary['repaired_answers']} |",
        f"| Citations still rejected after that | {summary['rejected_citations']} |",
        f"| Median / p90 latency per question | {summary['median_seconds']} s / {summary['p90_seconds']} s |",
        f"| Mean tool calls per question | {summary['mean_tool_calls']} |",
        f"| Total tokens | {summary['total_tokens']:,} |",
        "", "## Per question", "",
        "| id | type | passed | tools | citations | seconds | notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]  # fmt: skip
    for r in rows:
        s = r["score"]
        notes = ", ".join(f"{k}={v}" for k, v in s.items() if k != "passed")
        lines.append(
            f"| {r['id']} | {r['type']} | {'yes' if s['passed'] else 'no'} | {r['tools']} "
            f"| {r['citations']} | {r['seconds']} | {notes} |"
        )
    lines += ["", "Full answers: [transcripts/](transcripts/)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    questions = json.loads((EVAL / "questions.json").read_text(encoding="utf-8"))
    if args.only:
        questions = [q for q in questions if q["id"] in args.only.split(",")]
    index = FilingsIndex(load_chunks(ROOT / "data" / "chunks.jsonl"))
    fraud = FraudModel.load(ROOT / "models" / "fraud")
    llm = OllamaChat(args.model)
    agent = Agent({"sec-filings": filings_server(index), "fraud-scoring": fraud_server(fraud)}, llm)
    transcripts = EVAL / "transcripts"
    transcripts.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        try:
            answer = await agent.ask(q["question"])
        except Exception as exc:  # a timeout or crash fails one question, not the run
            rows.append(failed_row(q, exc, time.perf_counter() - t0))
            (transcripts / f"{q['id']}.txt").write_text(
                f"Q: {q['question']}\n\nFAILED: {type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            print(f"{q['id']}: FAILED {type(exc).__name__} {rows[-1]['seconds']}s", flush=True)
            continue
        if q["type"] == "filings":
            score = score_filings(q, answer, index)
        elif q["type"] == "out_of_scope":
            score = score_out_of_scope(answer)
        else:
            score = score_fraud(q, answer, fraud)
        rows.append({
            "id": q["id"], "type": q["type"], "score": score,
            "seconds": round(time.perf_counter() - t0, 1),
            "tools": " ".join(t.tool + ("" if t.ok else "!") for t in answer.tool_calls) or "-",
            "tool_calls": len(answer.tool_calls), "citations": len(answer.citations),
            "rejected_citations": len(answer.rejected_citations),
            "repaired": answer.repaired, "llm_calls": answer.llm_calls,
            "prompt_tokens": answer.prompt_tokens, "completion_tokens": answer.completion_tokens,
            "answer": answer.model_dump(),
        })  # fmt: skip
        (transcripts / f"{q['id']}.txt").write_text(
            f"Q: {q['question']}\n\n{render(answer)}\n", encoding="utf-8"
        )
        print(f"{q['id']}: passed={score['passed']} {rows[-1]['seconds']}s", flush=True)
    summary = summarize(rows)
    (EVAL / "results.json").write_text(
        json.dumps({"model": llm.name, "summary": summary, "rows": rows}, indent=1, default=str),
        encoding="utf-8",
    )
    write_markdown(summary, rows, llm.name, EVAL / "results.md")
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--only", default=None, help="Comma-separated question ids")
    anyio.run(run, parser.parse_args())


if __name__ == "__main__":
    main()
