"""Chunking, BM25 search and the sec-filings MCP server, called through a real MCP client."""

import json

import pytest
from mcp.client import Client

from sec_mcp_agent.filings.corpus import chunk_section, load_chunks, save_chunks
from sec_mcp_agent.filings.search import FilingsIndex, tokenize
from sec_mcp_agent.filings.server import build_server

pytestmark = pytest.mark.anyio


def test_chunks_keep_paragraphs_whole_and_respect_the_limit():
    paras = ["First paragraph. " * 5, "Second one is short.", "Third. " * 10]
    chunks = chunk_section("\n\n".join(p.strip() for p in paras), max_chars=120)
    assert all(len(c) <= 120 for c in chunks)
    joined = "\n\n".join(chunks)
    assert "Second one is short." in joined  # never cut through a paragraph that fits


def test_overlong_paragraph_is_split_at_sentence_ends():
    para = " ".join(f"Sentence number {i} ends here." for i in range(40))
    chunks = chunk_section(para, max_chars=200)
    assert len(chunks) > 1
    assert all(c.endswith("here.") for c in chunks)


def test_chunk_ids_are_stable_and_round_trip(chunks, tmp_path):
    assert chunks[0].chunk_id == "AAPL-2025-10-31-risk_factors-000"
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    path = tmp_path / "chunks.jsonl"
    assert save_chunks(chunks, path) == len(chunks)
    assert load_chunks(path) == chunks


def test_tokenizer_folds_plurals_and_drops_stopwords():
    assert tokenize("The risks of tariffs") == ["risk", "tariff"]
    assert tokenize("business") == ["business"]


def test_search_ranks_the_relevant_passage_first(index):
    hit = index.search("outsourcing partners manufacturing", k=1)[0]
    assert hit.ticker == "AAPL" and "outsourcing partners" in hit.text
    assert index.search("cybersecurity attack", k=1)[0].ticker == "JPM"


def test_search_filters_by_ticker_and_section(index):
    hits = index.search("interest rates prices demand", k=10, ticker="xom")
    assert hits and {h.ticker for h in hits} == {"XOM"}
    hits = index.search("net sales", k=10, section="mda")
    assert hits and {h.section for h in hits} == {"mda"}


def test_search_with_no_matching_terms_returns_nothing(index):
    assert index.search("the of and") == []
    assert index.search("zzzqqq") == []


def test_empty_index_is_rejected():
    with pytest.raises(ValueError):
        FilingsIndex([])


async def test_server_lists_tools_resources_and_prompts(index):
    async with Client(build_server(index)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {"list_companies", "search_filings", "get_chunk"}
        assert all(t.annotations.read_only_hint for t in tools.values())
        props = tools["search_filings"].input_schema["properties"]
        assert set(props) == {"query", "ticker", "section", "k"}
        templates = (await client.list_resource_templates()).resource_templates
        assert [t.uri_template for t in templates] == ["filing://{ticker}/{section}"]
        assert [p.name for p in (await client.list_prompts()).prompts] == ["risk_review"]


async def test_search_tool_returns_structured_hits(index):
    async with Client(build_server(index)) as client:
        result = await client.call_tool(
            "search_filings", {"query": "tariffs gross margins", "ticker": "AAPL", "k": 2}
        )
        assert not result.is_error
        hits = result.structured_content["result"]
        assert len(hits) <= 2 and hits[0]["ticker"] == "AAPL" and "tariffs" in hits[0]["text"]
        assert {"chunk_id", "company", "filing_date", "section", "score", "text"} <= set(hits[0])


@pytest.mark.parametrize(
    ("written", "section"),
    [("RISK FACTORS", "risk_factors"), ("Item 1A. Risk Factors", "risk_factors"),
     ("MD&A", "mda"), ("Management's Discussion and Analysis", "mda"), ("MDA", "mda"),
     ("Business", "business"), ("", None), (None, None)],
)  # fmt: skip
def test_section_names_models_write_are_understood(written, section):
    from sec_mcp_agent.filings.server import resolve_section

    assert resolve_section(written) == section


async def test_company_names_and_lowercase_tickers_resolve(index):
    async with Client(build_server(index)) as client:
        for ticker in ["aapl", "Apple", "apple inc."]:
            res = await client.call_tool(
                "search_filings", {"query": "tariffs", "ticker": ticker, "section": "Risk Factors"}
            )
            assert not res.is_error
            assert {h["ticker"] for h in res.structured_content["result"]} == {"AAPL"}
        bad = await client.call_tool("search_filings", {"query": "x", "section": "R&D"})
        assert bad.is_error and "risk_factors" in bad.content[0].text


async def test_unknown_ticker_is_a_tool_error_the_model_can_read(index):
    async with Client(build_server(index)) as client:
        result = await client.call_tool("search_filings", {"query": "risk", "ticker": "ZZZZ"})
        assert result.is_error
        assert (
            "not in the index" in result.content[0].text
            and "AAPL (Apple Inc.)" in result.content[0].text
        )


async def test_invalid_arguments_are_rejected(index):
    async with Client(build_server(index)) as client:
        result = await client.call_tool("search_filings", {"query": "risk", "k": 50})
        assert result.is_error


async def test_get_chunk_and_list_companies(index, chunks):
    async with Client(build_server(index)) as client:
        got = await client.call_tool("get_chunk", {"chunk_id": chunks[0].chunk_id})
        assert got.structured_content["text"] == chunks[0].text
        missing = await client.call_tool("get_chunk", {"chunk_id": "nope"})
        assert missing.is_error
        companies = (await client.call_tool("list_companies", {})).structured_content["result"]
        assert [c["ticker"] for c in companies] == ["AAPL", "JPM", "XOM"]
        aapl = companies[0]
        assert aapl["sections"] == ["mda", "risk_factors"] and aapl["filing_dates"] == [
            "2025-10-31"
        ]


async def test_section_resource_returns_the_full_section_with_chunk_ids(index):
    async with Client(build_server(index)) as client:
        res = await client.read_resource("filing://AAPL/risk_factors")
        text = res.contents[0].text
        assert text.startswith("Apple Inc. (AAPL) 10-K filed 2025-10-31, Item 1A. Risk Factors")
        assert "[AAPL-2025-10-31-risk_factors-000]" in text and "tariffs" in text


async def test_risk_review_prompt_asks_for_citations(index):
    async with Client(build_server(index)) as client:
        prompt = await client.get_prompt("risk_review", {"ticker": "jpm"})
        text = prompt.messages[0].content.text
        assert "JPM" in text and "chunk_id" in text and "word for word" in text


def test_fixture_is_valid_json():
    from conftest import FIXTURES

    assert len(json.loads((FIXTURES / "filings.json").read_text())) == 5
