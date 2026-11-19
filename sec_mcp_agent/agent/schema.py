"""Structured output: what the model must return and what the agent returns."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CitationDraft(BaseModel):
    """A citation as the model writes it: a chunk id and, ideally, a word-for-word quote."""

    chunk_id: str = Field(description="chunk_id of a passage returned by a tool")
    quote: str = Field(
        default="", description="One sentence copied word for word from that passage, if you can"
    )


class AnswerDraft(BaseModel):
    """The JSON the model is constrained to produce for its final answer."""

    answer: str = Field(description="The answer, in a few sentences")
    citations: list[CitationDraft] = Field(default_factory=list)


class Citation(BaseModel):
    chunk_id: str
    ticker: str
    company: str
    filing_date: str
    section: str
    quote: str
    quote_source: str = Field(
        description="model: the model's quote was found word for word in the passage; "
        "selected: the model's quote was not exact, so the verifier chose the passage sentence "
        "closest to the answer"
    )


class RejectedCitation(BaseModel):
    chunk_id: str
    quote: str
    reason: str


class ToolCallRecord(BaseModel):
    server: str
    tool: str
    arguments: dict[str, Any]
    ok: bool
    seconds: float
    error: str | None = None


class AgentAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    rejected_citations: list[RejectedCitation] = Field(
        default_factory=list,
        description="Citations the model made that are not backed by a retrieved passage",
    )
    fraud_scores: list[dict[str, Any]] = Field(default_factory=list)
    grounded: bool = Field(
        description="True when every citation matched a retrieved passage and an answer built "
        "on filings has at least one",
    )
    repaired: bool = Field(
        default=False, description="The verifier sent citations back once for correction"
    )
    tool_calls: list[ToolCallRecord]
    model: str
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float
