# ADR 0001: Expose the tools as MCP servers, not as functions inside the agent

**Status:** accepted

## Context
The agent needs two capabilities that already exist in other projects: searching 10-K
filings (sec-filings-rag) and scoring card transactions (realtime-fraud-detection-ml).
Wiring them in as Python functions would tie them to this one agent.

## Decision
Each capability is its own MCP server (`sec-filings`, `fraud-scoring`) built with the official
Python SDK (2.x). The agent is an MCP client: it discovers tools at runtime and routes calls by
tool name. The same servers run unchanged in Claude Desktop (`claude_desktop_config.example.json`)
and will be reused by the multi-agent project (fraud-investigation-agents).

Servers expose all three MCP primitives: tools (search, read, score), resources
(`filing://{ticker}/{section}`, `fraud://model-card`) and a prompt (`risk_review`).

## Consequences
- Tests run every tool through a real MCP client against the server object (in-memory
  transport), so schemas, validation errors and structured output are tested as clients see them.
  CI also starts both servers over stdio, the transport Claude Desktop uses.
- Tool errors (unknown ticker, unknown category) are returned as MCP tool errors with a message
  the model can act on, instead of exceptions that end the run.
- Every tool is annotated read-only; nothing here can change state, which keeps the
  prompt-injection surface small (see the guardrails project for the attack suite).
