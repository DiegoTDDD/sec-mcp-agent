"""MCP server `sec-filings`: search and read 10-K sections.

Tools      list_companies, search_filings, get_chunk
Resources  filing://{ticker}/{section}   the latest filing's full section text
Prompts    risk_review(ticker)            a ready-made question that forces cited answers

Run over stdio (what Claude Desktop and the agent use):
    python -m sec_mcp_agent.filings.server
The chunk file comes from SEC_MCP_CHUNKS (default: data/chunks.jsonl).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from sec_mcp_agent.filings.corpus import SECTION_NAMES, Chunk, load_chunks
from sec_mcp_agent.filings.search import FilingsIndex, SearchHit

SECTIONS = ("business", "risk_factors", "mda")
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
DEFAULT_CHUNKS = Path("data/chunks.jsonl")


def resolve_section(value: str | None) -> str | None:
    """Accept the names models actually write: "Risk Factors", "Item 1A", "MD&A", "Management's
    Discussion and Analysis", "BUSINESS"... The first eval run showed a strict enum turned most
    searches into validation errors."""
    if value is None or not value.strip():
        return None
    v = re.sub(r"[^a-z0-9]", "", value.lower())
    if "risk" in v or v in {"item1a", "1a"}:
        return "risk_factors"
    if v.startswith(("mda", "md&a", "management", "discussion")) or v in {"item7", "7"}:
        return "mda"
    if v.startswith("business") or v in {"item1", "1"}:
        return "business"
    # Anything else ("R&D", "competition", "CONTRACTORS"...) is not a 10-K section we index:
    # search them all rather than fail. Replaying the model's real calls (llm-evals-observability)
    # showed these were most of the calls still rejected after the first fixes.
    return None


def resolve_ticker(value: str | None, companies: dict[str, str]) -> str | None:
    """Accept a ticker in any case or a company name ("Apple", "JPMorgan Chase")."""
    if value is None or not value.strip():
        return None
    v = value.strip().upper()
    if v in companies:
        return v
    norm = re.sub(r"[^a-z0-9]", "", value.lower())
    for ticker, name in companies.items():
        if norm and norm in re.sub(r"[^a-z0-9]", "", name.lower()):
            return ticker
    raise ToolError(
        f"{value!r} is not in the index. Companies: "
        + ", ".join(f"{t} ({n})" for t, n in sorted(companies.items()))
    )


GENERIC_WORDS = {"bank", "general", "international", "united", "american", "first", "the"}
SUFFIXES = re.compile(
    r"[,.]?\s+(inc|corp|corporation|co|company|ltd|plc|group|holdings)\b.*$", re.I
)


def company_aliases(ticker: str, name: str) -> list[str]:
    """Lowercase phrases that name one company in a query: 'apple', 'jpmorgan chase',
    'bank of america'. Generic first words ('bank', 'general') are not used alone."""
    full = SUFFIXES.sub("", name).strip().lower().rstrip(" &,.")
    aliases = {full}
    first = full.split()[0] if full else ""
    if len(first) >= 4 and first not in GENERIC_WORDS:
        aliases.add(first)
    return sorted(a for a in aliases if a)


def detect_company(query: str, companies: dict[str, str]) -> str | None:
    """The company a query names, when exactly one matches (by name, or by a ticker of two or
    more capital letters). Eval run 3: the model often wrote "JPMorgan Chase cybersecurity risk"
    with no ticker filter, and lexical search then drifted to other companies."""
    lowered = query.lower()
    found = set()
    for ticker, name in companies.items():
        if (len(ticker) >= 2 and re.search(rf"\b{re.escape(ticker)}\b", query)) or any(
            re.search(rf"\b{re.escape(a)}\b", lowered) for a in company_aliases(ticker, name)
        ):
            found.add(ticker)
    return found.pop() if len(found) == 1 else None


def resolve_company(query: str, ticker: str | None, companies: dict[str, str]) -> str | None:
    """The ticker filter search_filings applies: the explicit one, else one named in the query."""
    return resolve_ticker(ticker, companies) or detect_company(query, companies)


class Company(BaseModel):
    ticker: str
    company: str
    filing_dates: list[str]
    sections: list[str]


def build_server(index: FilingsIndex) -> MCPServer:
    server = MCPServer(
        "sec-filings",
        instructions=(
            "Search and read 10-K filings (Business, Risk Factors, MD&A). Every passage has a "
            "chunk_id; cite chunk_ids and quote the passage text exactly when answering."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    def list_companies() -> list[Company]:
        """List the companies in the index, with their filing dates and sections."""
        out = []
        for ticker, company in sorted(index.companies.items()):
            keys = [k for k in index.sections if k[0] == ticker]
            dates = sorted({c.filing_date for k in keys for c in index.sections[k]}, reverse=True)
            out.append(
                Company(
                    ticker=ticker,
                    company=company,
                    filing_dates=dates,
                    sections=sorted(k[1] for k in keys),
                )
            )
        return out

    @server.tool(annotations=READ_ONLY)
    def search_filings(
        query: Annotated[str, Field(description="What to look for, in plain words")],
        ticker: Annotated[
            str | None, Field(description="Restrict to one company: ticker (AAPL) or name")
        ] = None,
        section: Annotated[
            str | None,
            Field(
                description="Optional: business, risk_factors or mda. Leave empty to search all."
            ),
        ] = None,
        k: Annotated[int, Field(ge=1, le=10, description="Number of passages")] = 5,
    ) -> list[SearchHit]:
        """Search 10-K passages by keywords. Returns passages with chunk_id, company, filing
        date, section and text, best match first. If no ticker is given and the query names
        exactly one indexed company, results are restricted to that company."""
        return index.search(
            query,
            k=k,
            ticker=resolve_company(query, ticker, index.companies),
            section=resolve_section(section),
        )

    @server.tool(annotations=READ_ONLY)
    def get_chunk(
        chunk_id: Annotated[str, Field(description="A chunk_id from search_filings")],
    ) -> Chunk:
        """Read one passage by its chunk_id."""
        chunk = index.by_id.get(chunk_id)
        if chunk is None:
            raise ToolError(f"No chunk with id {chunk_id!r}")
        return chunk

    @server.resource(
        "filing://{ticker}/{section}",
        mime_type="text/plain",
        description="Full text of a section of the company's latest 10-K",
    )
    def section_text(ticker: str, section: str) -> str:
        chunks = index.latest_section(ticker, section)
        if not chunks:
            raise ResourceError(f"No {section} section for {ticker}")
        head = (
            f"{chunks[0].company} ({chunks[0].ticker}) 10-K filed {chunks[0].filing_date}, "
            f"{SECTION_NAMES.get(section, section)}"
        )
        return head + "\n\n" + "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)

    @server.prompt()
    def risk_review(ticker: str) -> str:
        """Ask for the main risks a company reports, with citations."""
        return (
            f"Using the sec-filings tools, list the three most material risks {ticker.upper()} "
            "reports in its latest 10-K Risk Factors. For each risk, cite the chunk_id and quote "
            "one sentence from it word for word. Say so if the filings do not cover something."
        )

    return server


def main() -> None:
    path = Path(os.environ.get("SEC_MCP_CHUNKS", DEFAULT_CHUNKS))
    build_server(FilingsIndex(load_chunks(path))).run("stdio")


if __name__ == "__main__":
    main()
