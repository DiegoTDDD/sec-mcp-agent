"""Download the latest 10-K of each company from SEC EDGAR and write data/chunks.jsonl.

Sections: Business, Risk Factors, MD&A. SEC asks automated clients to identify themselves:
set SEC_IDENTITY="Your Name you@example.com".

    python scripts/build_index.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from edgar import Company, set_identity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sec_mcp_agent.filings.corpus import make_chunks, save_chunks  # noqa: E402

log = logging.getLogger(__name__)
OUT = ROOT / "data" / "chunks.jsonl"
MANIFEST = ROOT / "data" / "manifest.json"
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BAC", "GS",
    "V", "JNJ", "PFE", "UNH", "XOM", "CVX", "KO", "WMT", "NFLX", "BA",
]  # fmt: skip
SECTIONS = {
    "business": lambda tenk: tenk.business,
    "risk_factors": lambda tenk: tenk.risk_factors,
    "mda": lambda tenk: tenk.management_discussion,
}
MIN_SECTION_CHARS = 2_000


def clean(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    identity = os.environ.get("SEC_IDENTITY")
    if not identity:
        raise SystemExit('Set SEC_IDENTITY="Your Name you@example.com" (SEC requires it)')
    set_identity(identity)
    chunks, manifest = [], []
    for ticker in TICKERS:
        try:
            company = Company(ticker)
            filing = company.get_filings(form="10-K").latest()
            tenk = filing.obj()
        except Exception as exc:
            log.warning("%s: skipped (%s)", ticker, exc)
            continue
        name = company.name.title() if company.name.isupper() else company.name
        for section, getter in SECTIONS.items():
            try:
                text = clean(getter(tenk) or "")
            except Exception:
                text = ""
            if len(text) < MIN_SECTION_CHARS:
                log.warning("%s %s: section missing or too short", ticker, section)
                continue
            new = make_chunks(ticker, name, str(filing.filing_date), section, text)
            chunks += new
            manifest.append(
                {
                    "ticker": ticker,
                    "company": name,
                    "filing_date": str(filing.filing_date),
                    "accession": filing.accession_number,
                    "section": section,
                    "chars": len(text),
                    "chunks": len(new),
                }
            )
        log.info("%s: %d chunks so far", ticker, len(chunks))
        time.sleep(0.2)  # well under SEC's limit of 10 requests per second
    if not chunks:
        raise SystemExit("No sections downloaded: check SEC_IDENTITY and network access")
    save_chunks(chunks, OUT)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log.info("Wrote %d chunks from %d sections to %s", len(chunks), len(manifest), OUT)


if __name__ == "__main__":
    main()
