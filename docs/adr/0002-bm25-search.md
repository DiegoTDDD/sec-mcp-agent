# ADR 0002: BM25 search inside the server instead of the hybrid retriever

**Status:** accepted

## Context
sec-filings-rag already has a hybrid retriever (dense + BM25 + reranker, hit@5 0.80 on its
66-question set). Reusing it here would need Qdrant, an embedding model and a reranker to be
running before the MCP server can start.

## Decision
The `sec-filings` server builds a BM25 index (rank-bm25) over paragraph-aligned chunks of the
latest 10-K of 20 companies when it starts, with ticker and section filters.

## Consequences
- The server starts in about a second with no model download and no external service, so it
  works in Claude Desktop on any laptop and in CI.
- The agent compensates for lexical search by searching again with other words, which the
  system prompt asks for; the eval measures whether answers still find the expected evidence.
- Ticker and section filters do most of the precision work: when a company is named the agent
  searches only its filing.
- If the eval shows lexical misses, the retriever behind `search_filings` can be swapped for the
  hybrid one without changing the tool's schema or the agent.
