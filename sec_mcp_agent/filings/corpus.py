"""10-K section chunks: the unit the search tool returns and the agent cites."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

SECTION_NAMES = {
    "business": "Item 1. Business",
    "risk_factors": "Item 1A. Risk Factors",
    "mda": "Item 7. Management's Discussion and Analysis",
}


class Chunk(BaseModel):
    """A passage of one 10-K section, with everything needed to cite it."""

    chunk_id: str = Field(description="Stable id: TICKER-YYYY-MM-DD-section-NNN")
    ticker: str
    company: str
    filing_date: str
    section: str = Field(description="business, risk_factors or mda")
    position: int = Field(description="Order of the chunk inside its section, from 0")
    text: str


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_section(text: str, max_chars: int = 1200) -> list[str]:
    """Pack whole paragraphs into chunks of at most `max_chars`.

    Paragraphs are kept intact so a citation quote never spans a cut. A single paragraph
    longer than the limit is split at sentence ends.
    """
    pieces: list[str] = []
    for para in split_paragraphs(text):
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            if current and len(current) + 1 + len(sentence) > max_chars:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def make_chunks(
    ticker: str, company: str, filing_date: str, section: str, text: str, max_chars: int = 1200
) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{ticker}-{filing_date}-{section}-{i:03d}",
            ticker=ticker,
            company=company,
            filing_date=filing_date,
            section=section,
            position=i,
            text=body,
        )
        for i, body in enumerate(chunk_section(text, max_chars))
    ]


def save_chunks(chunks: Iterable[Chunk], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
            n += 1
    return n


def load_chunks(path: str | Path) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
