"""The agent loop, driven by a scripted model against the real MCP servers (in memory)."""

import json

import pytest

from sec_mcp_agent.agent.agent import Agent, normalize
from sec_mcp_agent.agent.llm import ChatReply, ScriptedChat, ToolCallRequest
from sec_mcp_agent.filings.server import build_server as filings_server
from sec_mcp_agent.fraud.server import build_server as fraud_server

pytestmark = pytest.mark.anyio
QUOTE = "Changes in tariffs and other trade measures could adversely affect the Company's business."


def search(query: str, **kw) -> ChatReply:
    return ChatReply(tool_calls=[ToolCallRequest("search_filings", {"query": query, **kw})])


def final(answer: str, *citations: tuple[str, str]) -> ChatReply:
    return ChatReply(
        content=json.dumps(
            {"answer": answer, "citations": [{"chunk_id": c, "quote": q} for c, q in citations]}
        )
    )


def first_chunk_id(messages: list) -> str:
    tool_text = next(m["content"] for m in messages if m["role"] == "tool")
    return tool_text.split("[chunk_id: ")[1].split("]")[0]


@pytest.fixture
def servers(index, fraud_model):
    return {"sec-filings": filings_server(index), "fraud-scoring": fraud_server(fraud_model)}


async def test_answer_with_a_verified_citation(servers):
    llm = ScriptedChat(
        [
            search("tariffs", ticker="AAPL"),
            ChatReply(content="I have what I need."),
            lambda msgs: final("Tariffs could hurt margins.", (first_chunk_id(msgs), QUOTE)),
        ]
    )
    answer = await Agent(servers, llm).ask("What does Apple say about tariffs?")
    assert answer.grounded and not answer.rejected_citations
    [c] = answer.citations
    assert (c.ticker, c.section, c.filing_date) == ("AAPL", "risk_factors", "2025-10-31")
    assert [t.tool for t in answer.tool_calls] == ["search_filings"]
    assert answer.llm_calls == 3


async def test_model_sees_every_tool_from_both_servers(servers):
    llm = ScriptedChat(
        [ChatReply(content="ok"), ChatReply(content="ok"), final("No tools needed.")]
    )
    await Agent(servers, llm).ask("hi")
    names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert names == {"list_companies", "search_filings", "get_chunk", "score_transaction"}
    assert llm.calls[-1]["schema"]["title"] == "AnswerDraft"


async def test_inexact_quote_on_a_retrieved_passage_is_replaced_by_a_real_sentence(servers):
    llm = ScriptedChat([
        search("tariffs", ticker="AAPL"),
        ChatReply(content="done"),
        lambda msgs: final("Tariffs could reduce margins.",
                           (first_chunk_id(msgs), "Tariffs will double next year.")),
    ])  # fmt: skip
    answer = await Agent(servers, llm).ask("Apple tariffs?")
    [c] = answer.citations
    assert answer.grounded and not answer.repaired
    # A real sentence of the passage, the one closest to the answer, not the invented quote.
    assert c.quote_source == "selected"
    assert c.quote == (
        "New or increased tariffs on the Company's products or components could reduce "
        "gross margins and demand."
    )


async def test_exact_quote_is_kept_as_the_model_wrote_it(servers):
    llm = ScriptedChat(
        [
            search("tariffs", ticker="AAPL"),
            ChatReply(content="done"),
            lambda msgs: final("x", (first_chunk_id(msgs), QUOTE)),
        ]
    )
    [c] = (await Agent(servers, llm).ask("q")).citations
    assert c.quote_source == "model" and c.quote == QUOTE


async def test_citation_that_stays_wrong_is_reported_not_passed_through(servers):
    bad = final(
        "JPM faces tariffs.",
        ("JPM-2025-02-14-risk_factors-000", QUOTE),
        ("MADE-UP-ID", "Something that was never retrieved at all."),
    )
    llm = ScriptedChat([search("tariffs", ticker="AAPL"), ChatReply(content="done"), bad, bad])
    answer = await Agent(servers, llm).ask("q")
    assert not answer.grounded and answer.citations == []
    reasons = {r.chunk_id: r.reason for r in answer.rejected_citations}
    assert reasons["MADE-UP-ID"] == "chunk_id was not returned by any tool"
    assert reasons["JPM-2025-02-14-risk_factors-000"] == "chunk_id was not returned by any tool"


async def test_answer_from_filings_without_citations_is_not_grounded(servers):
    llm = ScriptedChat(
        [search("interest rates", ticker="JPM"), ChatReply(content="done"), final("Rates matter.")]
    )
    answer = await Agent(servers, llm).ask("q")
    assert not answer.grounded


async def test_quotes_tolerate_whitespace_and_typographic_quotes(servers):
    curly = QUOTE.replace("'", "’").replace(" could ", "  could\n")
    llm = ScriptedChat(
        [
            search("tariffs", ticker="AAPL"),
            ChatReply(content="done"),
            lambda msgs: final("ok", (first_chunk_id(msgs), curly)),
        ]
    )
    assert (await Agent(servers, llm).ask("q")).grounded
    assert normalize("A  b’s.") == "A b's"


async def test_tool_errors_go_back_to_the_model_and_are_recorded(servers):
    llm = ScriptedChat(
        [
            search("risk", ticker="NOPE"),
            ChatReply(tool_calls=[ToolCallRequest("delete_everything", {})]),
            ChatReply(content="done"),
            final("I could not find that company."),
        ]
    )
    answer = await Agent(servers, llm).ask("q")
    tool_msgs = [m["content"] for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0].startswith("Error:") and "not in the index" in tool_msgs[0]
    assert "no tool called" in tool_msgs[1]
    assert [(t.tool, t.ok) for t in answer.tool_calls] == [
        ("search_filings", False),
        ("delete_everything", False),
    ]


async def test_fraud_question_uses_the_model_score(servers, fraud_model):
    home = fraud_model.home_city["card_0000"]
    away = "Manaus" if home != "Manaus" else "Recife"
    tx = {
        "card_id": "card_0000",
        "amount": 4500,
        "category": "electronics",
        "city": away,
        "timestamp": "2026-03-01T02:10:00",
    }
    llm = ScriptedChat(
        [
            ChatReply(tool_calls=[ToolCallRequest("score_transaction", tx)]),
            ChatReply(content="done"),
            final("Flagged as likely fraud."),
        ]
    )
    answer = await Agent(servers, llm).ask("Is this fraud?")
    [score] = answer.fraud_scores
    assert score["flagged"] is True and answer.grounded


async def test_step_limit_stops_a_model_that_never_stops_calling_tools(servers):
    llm = ScriptedChat([search("risk")] * 3 + [final("Stopped.")])
    answer = await Agent(servers, llm, max_steps=3).ask("q")
    assert len(answer.tool_calls) == 3 and answer.answer == "Stopped."


async def test_invalid_json_gets_one_retry_then_falls_back_to_text(servers):
    llm = ScriptedChat(
        [
            ChatReply(content="done"),
            ChatReply(content="done"),
            ChatReply(content="not json"),
            ChatReply(content="still not json"),
        ]
    )
    answer = await Agent(servers, llm).ask("q")
    assert answer.answer == "still not json" and answer.citations == []
    assert "not valid JSON" in llm.calls[-1]["messages"][-1]["content"]


async def test_invented_chunk_id_gets_one_repair_round(servers):
    llm = ScriptedChat([
        search("tariffs", ticker="AAPL"),
        ChatReply(content="done"),
        final("x", ("AAPL-made-up-000", QUOTE)),
        lambda m: final("x", (first_chunk_id(m), QUOTE)),
    ])  # fmt: skip
    answer = await Agent(servers, llm).ask("q")
    assert answer.repaired and answer.grounded and len(answer.citations) == 1
    assert "AAPL-made-up-000" in llm.calls[-1]["messages"][-1]["content"]


async def test_citations_valid_before_the_repair_are_kept(servers):
    llm = ScriptedChat([
        search("tariffs", ticker="AAPL"),
        ChatReply(content="done"),
        lambda m: final("x", (first_chunk_id(m), QUOTE), ("BAD-ID", "nothing")),
        final("x"),  # the model drops everything when asked to fix one
    ])  # fmt: skip
    answer = await Agent(servers, llm).ask("q")
    assert answer.repaired and len(answer.citations) == 1 and answer.grounded


async def test_model_that_skips_tools_is_reminded_once(servers):
    llm = ScriptedChat([
        ChatReply(content="From memory: Apple faces tariffs."),
        search("tariffs", ticker="Apple"),
        ChatReply(content="done"),
        lambda m: final("Tariffs.", (first_chunk_id(m), QUOTE)),
    ])  # fmt: skip
    answer = await Agent(servers, llm).ask("Apple tariffs?")
    assert "Do not answer from memory" in llm.calls[1]["messages"][-1]["content"]
    assert answer.grounded and answer.tool_calls[0].arguments["ticker"] == "Apple"


async def test_reminder_is_not_repeated(servers):
    llm = ScriptedChat([ChatReply(content="no"), ChatReply(content="still no"), final("Unknown.")])
    answer = await Agent(servers, llm).ask("q")
    assert answer.tool_calls == [] and answer.llm_calls == 3


@pytest.mark.parametrize(
    "nested",
    [
        lambda cid: json.dumps({"answer": "Tariffs hurt margins.",
                                "citations": [{"chunk_id": cid, "quote": QUOTE}]}),
        lambda cid: json.dumps([{"chunk_id": cid, "quote": QUOTE}]),
        lambda cid: json.dumps([cid]),
    ],
)  # fmt: skip
async def test_json_nested_in_the_answer_field_is_recovered(servers, nested):
    llm = ScriptedChat([
        search("tariffs", ticker="AAPL"),
        ChatReply(content="done"),
        lambda m: ChatReply(content=json.dumps({"answer": nested(first_chunk_id(m)),
                                                "citations": []})),
        final("Tariffs hurt margins."),  # only used when the answer text was missing
    ])  # fmt: skip
    answer = await Agent(servers, llm).ask("Apple tariffs?")
    assert answer.grounded and len(answer.citations) == 1
    assert answer.answer == "Tariffs hurt margins."
    assert not answer.answer.startswith(("{", "["))


def test_unwrap_leaves_ordinary_answers_alone():
    from sec_mcp_agent.agent.agent import unwrap
    from sec_mcp_agent.agent.schema import AnswerDraft

    d = AnswerDraft(answer="[E2] shows tariffs matter.", citations=[])
    assert unwrap(d) == d


def test_broken_nested_json_is_salvaged():
    from sec_mcp_agent.agent.agent import unwrap
    from sec_mcp_agent.agent.schema import AnswerDraft

    broken = (
        '{"citations": [{\'chunk_id": "JPM-2026-02-13-mda-341", "quote": "The governance '
        "structure for the Global Cyber"
    )  # mixed quotes, cut off
    d = unwrap(AnswerDraft(answer=broken, citations=[]))
    assert [c.chunk_id for c in d.citations] == ["JPM-2026-02-13-mda-341"] and d.answer == ""
    d = unwrap(
        AnswerDraft(
            answer='{"answer": "Prices are volatile.", "citations": [{"chunk_id": '
            '"CVX-2026-02-20-risk_factors-002", "quote": "Prices',
            citations=[],
        )
    )
    assert d.answer == "Prices are volatile."
    assert [c.chunk_id for c in d.citations] == ["CVX-2026-02-20-risk_factors-002"]
