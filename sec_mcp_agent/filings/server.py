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
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from sec_mcp_agent.filings.corpus import SECTION_NAMES, Chunk, load_chunks
from sec_mcp_agent.filings.search import FilingsIndex, SearchHit

Section = Literal["business", "risk_factors", "mda"]
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
DEFAULT_CHUNKS = Path("data/chunks.jsonl")


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
            str | None, Field(description="Restrict to one company, e.g. AAPL")
        ] = None,
        section: Annotated[
            Section | None, Field(description="Restrict to one 10-K section")
        ] = None,
        k: Annotated[int, Field(ge=1, le=10, description="Number of passages")] = 5,
    ) -> list[SearchHit]:
        """Search 10-K passages by keywords. Returns passages with chunk_id, company, filing
        date, section and text, best match first."""
        if ticker and ticker.upper() not in index.companies:
            raise ToolError(
                f"Unknown ticker {ticker!r}. Known: {', '.join(sorted(index.companies))}"
            )
        return index.search(query, k=k, ticker=ticker, section=section)

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
