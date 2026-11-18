"""Command line: ask the agent a question, with both MCP servers started over stdio.

python -m sec_mcp_agent "What supply chain risks does Apple report?"
python -m sec_mcp_agent --json "Is this transaction fraud? card_0042, R$ 4,800 ..."
"""

from __future__ import annotations

import argparse
import os
import sys

import anyio
from mcp.client.stdio import StdioServerParameters

from sec_mcp_agent.agent.agent import Agent, to_json
from sec_mcp_agent.agent.llm import OllamaChat
from sec_mcp_agent.agent.schema import AgentAnswer


def stdio_server(module: str) -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", module], env=dict(os.environ))


def render(answer: AgentAnswer) -> str:
    lines = [answer.answer, ""]
    for i, c in enumerate(answer.citations, 1):
        lines.append(
            f"[{i}] {c.company} ({c.ticker}) 10-K {c.filing_date}, {c.section}, "
            f'{c.chunk_id}\n    "{c.quote}"'
        )
    for r in answer.rejected_citations:
        lines.append(f"[rejected] {r.chunk_id}: {r.reason}")
    for s in answer.fraud_scores:
        reasons = ", ".join(
            f"{r['feature']}={r['value']:g} ({r['log_odds']:+.2f})" for r in s["top_reasons"]
        )
        lines.append(
            f"[fraud model] p={s['probability']:.3f} flagged={s['flagged']} "
            f"threshold={s['threshold']:.3f}; {reasons}"
        )
    tools = ", ".join(f"{t.tool}{'' if t.ok else ' (error)'}" for t in answer.tool_calls)
    lines += [
        "",
        f"grounded={answer.grounded} | tools: {tools or 'none'} | {answer.model}, "
        f"{answer.llm_calls} LLM calls, {answer.seconds:.1f}s",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--model", default=None, help="Ollama model (default: qwen2.5:3b)")
    parser.add_argument("--json", action="store_true", help="Print the full structured answer")
    args = parser.parse_args()
    agent = Agent(
        {
            "sec-filings": stdio_server("sec_mcp_agent.filings.server"),
            "fraud-scoring": stdio_server("sec_mcp_agent.fraud.server"),
        },
        OllamaChat(args.model),
    )
    answer = anyio.run(agent.ask, args.question)
    print(to_json(answer) if args.json else render(answer))  # noqa: T201


if __name__ == "__main__":
    main()
