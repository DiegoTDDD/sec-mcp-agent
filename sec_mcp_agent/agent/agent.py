"""A tool-calling agent that talks to MCP servers and returns a checked, structured answer.

Loop:
  1. Connect to every MCP server (stdio for real use, in-memory in tests) and expose all their
     tools to the model.
  2. Let the model call tools until it stops asking (at most `max_steps` rounds).
  3. Ask for the final answer constrained to the AnswerDraft JSON schema.
  4. Check every citation against the passages the tools actually returned: the chunk_id must
     have been retrieved and the quote must appear in it word for word. If any fails, give the
     model one chance to repair; citations that still fail are reported as rejected, never
     passed through.
"""

from __future__ import annotations

import json
import re
import time
from contextlib import AsyncExitStack
from typing import Any

from mcp.client import Client
from pydantic import ValidationError

from sec_mcp_agent.agent.llm import ChatModel, ChatReply, Message
from sec_mcp_agent.agent.schema import (
    AgentAnswer,
    AnswerDraft,
    Citation,
    RejectedCitation,
    ToolCallRecord,
)
from sec_mcp_agent.filings.search import tokenize

SYSTEM_PROMPT = """You answer questions about US public companies' 10-K filings and about card \
transactions, using only the tools you are given.

- For questions about companies, call search_filings (use the ticker filter when the company is \
named) before answering. Search again with other words if the passages do not answer the question.
- For questions about a transaction, call score_transaction and report the probability, whether \
it is flagged and the top reasons.
- Never invent facts, numbers or chunk_ids. If the tools do not answer the question, say so."""

TOOL_REMINDER = """Do not answer from memory. Call the tools first: search_filings for questions \
about companies, score_transaction for transactions. If no tool can help, say so."""

FINAL_INSTRUCTION = """Now write the final answer as JSON with "answer" and "citations". Each \
citation is {"chunk_id": "<a chunk_id returned by a tool>", "quote": "<a sentence copied from \
that passage>"}. Cite every passage you rely on, using chunk_ids exactly as the tools gave them. \
Use an empty list only if you used no filings."""

MIN_QUOTE_CHARS = 20
TOOL_TEXT_LIMIT = 6000
QUOTE_CHARS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", " ": " "})


def normalize(text: str) -> str:
    text = text.translate(QUOTE_CHARS)
    return re.sub(r"\s+", " ", text).strip().strip(".…").strip()


class Agent:
    def __init__(self, servers: dict[str, Any], llm: ChatModel, max_steps: int = 6) -> None:
        """`servers` maps a name to anything mcp.client.Client accepts: an MCPServer object,
        StdioServerParameters, or a URL."""
        self.servers = servers
        self.llm = llm
        self.max_steps = max_steps

    async def ask(self, question: str) -> AgentAnswer:
        start = time.perf_counter()
        async with AsyncExitStack() as stack:
            clients = {
                name: await stack.enter_async_context(Client(target))
                for name, target in self.servers.items()
            }
            routes: dict[str, str] = {}
            tool_specs: list[dict] = []
            for server_name, client in clients.items():
                for tool in (await client.list_tools()).tools:
                    routes[tool.name] = server_name
                    tool_specs.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": tool.input_schema,
                            },
                        }
                    )
            run = _Run(question, self.llm)
            messages: list[Message] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]

            reminded = False
            for _ in range(self.max_steps):
                reply = run.chat(messages, tools=tool_specs)
                if not reply.tool_calls:
                    if run.records or reminded:
                        break
                    # The first eval run: a 3B model sometimes answered from memory without
                    # calling anything. One reminder, then accept its choice.
                    reminded = True
                    messages.append({"role": "assistant", "content": reply.content})
                    messages.append({"role": "user", "content": TOOL_REMINDER})
                    continue
                messages.append(
                    {
                        "role": "assistant",
                        "content": reply.content,
                        "tool_calls": [
                            {"function": {"name": c.name, "arguments": c.arguments}}
                            for c in reply.tool_calls
                        ],
                    }
                )
                for call in reply.tool_calls:
                    text = await run.call_tool(clients, routes, call.name, call.arguments)
                    messages.append({"role": "tool", "tool_name": call.name, "content": text})

            messages.append({"role": "user", "content": FINAL_INSTRUCTION})
            draft = run.final_answer(messages)
            citations, rejected = run.check(draft)
            repaired = bool(rejected)
            if rejected:
                messages.append({"role": "assistant", "content": draft.model_dump_json()})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "These chunk_ids were not returned by any tool: "
                            + ", ".join(r.chunk_id for r in rejected)
                            + ". Rewrite the JSON citing only chunk_ids that appear in the "
                            "tool results above."
                        ),
                    }
                )
                draft = run.final_answer(messages)
                kept = citations
                citations, rejected = run.check(draft)
                ids = {c.chunk_id for c in citations}
                citations += [c for c in kept if c.chunk_id not in ids]

        used_filings = any(r.tool in {"search_filings", "get_chunk"} for r in run.records)
        grounded = not rejected and (bool(citations) or not used_filings)
        return AgentAnswer(
            question=question,
            answer=draft.answer,
            citations=citations,
            rejected_citations=rejected,
            fraud_scores=run.fraud_scores,
            grounded=grounded,
            repaired=repaired,
            tool_calls=run.records,
            model=self.llm.name,
            llm_calls=run.llm_calls,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            seconds=round(time.perf_counter() - start, 2),
        )


class _Run:
    """State of one question: retrieved passages, tool records and token counts."""

    def __init__(self, question: str, llm: ChatModel) -> None:
        self.question = question
        self.llm = llm
        self.passages: dict[str, dict[str, Any]] = {}
        self.fraud_scores: list[dict[str, Any]] = []
        self.records: list[ToolCallRecord] = []
        self.llm_calls = self.prompt_tokens = self.completion_tokens = 0

    def chat(
        self, messages: list[Message], tools: list[dict] | None = None, schema: dict | None = None
    ) -> ChatReply:
        reply = self.llm.chat(messages, tools=tools, schema=schema)
        self.llm_calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens
        return reply

    async def call_tool(
        self,
        clients: dict[str, Client],
        routes: dict[str, str],
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        server = routes.get(name)
        t0 = time.perf_counter()
        if server is None:
            self.records.append(
                ToolCallRecord(
                    server="?",
                    tool=name,
                    arguments=arguments,
                    ok=False,
                    seconds=0.0,
                    error="unknown tool",
                )
            )
            return f"Error: there is no tool called {name!r}."
        result = await clients[server].call_tool(name, arguments)
        seconds = round(time.perf_counter() - t0, 3)
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            self.records.append(
                ToolCallRecord(
                    server=server,
                    tool=name,
                    arguments=arguments,
                    ok=False,
                    seconds=seconds,
                    error=text[:500],
                )
            )
            return f"Error: {text}"
        self.records.append(
            ToolCallRecord(server=server, tool=name, arguments=arguments, ok=True, seconds=seconds)
        )
        data = result.structured_content or {}
        payload = data.get("result", data) if isinstance(data, dict) else data
        if name == "search_filings":
            for hit in payload:
                self.passages[hit["chunk_id"]] = hit
            text = (
                "\n\n".join(
                    f"[chunk_id: {h['chunk_id']}] {h['company']} ({h['ticker']}), 10-K filed "
                    f"{h['filing_date']}, section {h['section']}\n{h['text']}"
                    for h in payload
                )
                or "No passages matched."
            )
        elif name == "get_chunk":
            self.passages[payload["chunk_id"]] = payload
        elif name == "score_transaction":
            self.fraud_scores.append(payload)
        return text[:TOOL_TEXT_LIMIT]

    def final_answer(self, messages: list[Message]) -> AnswerDraft:
        schema = AnswerDraft.model_json_schema()
        reply = self.chat(messages, schema=schema)
        try:
            return AnswerDraft.model_validate_json(reply.content)
        except ValidationError as exc:
            retry = [
                *messages,
                {"role": "assistant", "content": reply.content},
                {
                    "role": "user",
                    "content": f"That is not valid JSON for the schema: "
                    f"{exc.errors()[0]['msg']}. Answer again.",
                },
            ]
            reply = self.chat(retry, schema=schema)
            try:
                return AnswerDraft.model_validate_json(reply.content)
            except ValidationError:
                return AnswerDraft(answer=reply.content.strip() or "No answer.", citations=[])

    def check(self, draft: AnswerDraft) -> tuple[list[Citation], list[RejectedCitation]]:
        """Accept a citation only if its passage was retrieved in this run.

        The quote shown to the reader is always real text from that passage: the model's own
        quote when it is there word for word, otherwise the passage sentence closest to the
        answer. Eval run 3 showed why: a 3B model picks the right passage far more reliably
        than it copies a sentence exactly, and asking it to retry made it drop citations.
        """
        valid: list[Citation] = []
        rejected: list[RejectedCitation] = []
        seen: set[str] = set()
        for c in draft.citations:
            passage = self.passages.get(c.chunk_id.strip())
            if passage is None:
                reason = "chunk_id was not returned by any tool"
                rejected.append(RejectedCitation(chunk_id=c.chunk_id, quote=c.quote, reason=reason))
                continue
            if passage["chunk_id"] in seen:
                continue
            seen.add(passage["chunk_id"])
            quote = normalize(c.quote)
            if len(quote) >= MIN_QUOTE_CHARS and quote in normalize(passage["text"]):
                text, source = c.quote.strip(), "model"
            else:
                text, source = (
                    best_sentence(passage["text"], f"{draft.answer} {c.quote}"),
                    "selected",
                )
            valid.append(
                Citation(
                    chunk_id=passage["chunk_id"],
                    ticker=passage["ticker"],
                    company=passage["company"],
                    filing_date=passage["filing_date"],
                    section=passage["section"],
                    quote=text,
                    quote_source=source,
                )
            )
        return valid, rejected


def best_sentence(passage: str, target: str) -> str:
    """The passage sentence sharing the most search terms with the answer."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", passage) if len(s.strip()) > 20]
    if not sentences:
        return passage.strip()[:300]
    wanted = set(tokenize(target))
    return max(sentences, key=lambda s: len(wanted & set(tokenize(s))))  # first wins on ties


def to_json(answer: AgentAnswer) -> str:
    return json.dumps(answer.model_dump(), indent=2, default=str)
