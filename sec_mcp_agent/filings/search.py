"""BM25 search over filing chunks, with ticker and section filters.

BM25 needs no model download and no GPU, so the server starts in a second on any laptop and
in CI. ADR 0002 explains why lexical search is enough here and how the sec-filings-rag
project's hybrid retriever relates.
"""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from sec_mcp_agent.filings.corpus import Chunk

STOPWORDS = frozenset(
    (
        "a an and are as at be been but by can could did do does for from had has have "
        "how in into is it its may might more most of on or our such than that the their "
        "them there these they this those to was we were what when where which while who "
        "will with would you your company companies"
    ).split()
)
TOKEN = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        # Light plural folding: "risks" and "risk" should match; "business" stays as is.
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


class SearchHit(BaseModel):
    chunk_id: str
    ticker: str
    company: str
    filing_date: str
    section: str
    score: float
    text: str


class FilingsIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("the index needs at least one chunk")
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self._bm25 = BM25Okapi([tokenize(f"{c.company} {c.text}") for c in chunks])
        self.companies: dict[str, str] = {c.ticker: c.company for c in chunks}
        self.sections: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
        for c in sorted(chunks, key=lambda c: (c.filing_date, c.position)):
            self.sections[(c.ticker, c.section)].append(c)

    def search(
        self, query: str, k: int = 5, ticker: str | None = None, section: str | None = None
    ) -> list[SearchHit]:
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ticker = ticker.upper() if ticker else None
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        hits = []
        for i in ranked:
            chunk = self.chunks[i]
            if scores[i] <= 0:
                break
            if (ticker and chunk.ticker != ticker) or (section and chunk.section != section):
                continue
            hits.append(
                SearchHit(
                    score=round(float(scores[i]), 4), **chunk.model_dump(exclude={"position"})
                )
            )
            if len(hits) == k:
                break
        return hits

    def latest_section(self, ticker: str, section: str) -> list[Chunk]:
        """Chunks of the most recent filing's section, in reading order."""
        chunks = self.sections.get((ticker.upper(), section), [])
        if not chunks:
            return []
        latest = max(c.filing_date for c in chunks)
        return [c for c in chunks if c.filing_date == latest]
