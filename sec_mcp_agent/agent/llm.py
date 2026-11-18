"""The chat-model interface the agent needs, an Ollama implementation and a scripted fake.

The agent only needs one call: given messages and tool specs (and optionally a JSON schema
the reply must follow), return text and/or tool calls. Keeping it this narrow means tests
drive the whole agent loop with a scripted model, and swapping Ollama for another provider
is one class.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass
class ToolCallRequest:
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatReply:
    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatModel(Protocol):
    name: str

    def chat(
        self, messages: list[Message], tools: list[dict] | None = None, schema: dict | None = None
    ) -> ChatReply: ...


class OllamaChat:
    """A local model served by Ollama (free, runs on a laptop or a CI runner)."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        seed: int = 0,
    ) -> None:
        import ollama

        self.name = model or os.environ.get("SEC_MCP_MODEL", "qwen2.5:3b")
        self._client = ollama.Client(host=host or os.environ.get("OLLAMA_HOST"))
        self._options = {"temperature": temperature, "num_ctx": num_ctx, "seed": seed}

    def chat(
        self, messages: list[Message], tools: list[dict] | None = None, schema: dict | None = None
    ) -> ChatReply:
        response = self._client.chat(
            model=self.name,
            messages=messages,
            tools=tools or None,
            format=schema,
            options=self._options,
        )
        message = response.message
        calls = [
            ToolCallRequest(name=c.function.name, arguments=dict(c.function.arguments or {}))
            for c in (message.tool_calls or [])
        ]
        return ChatReply(
            content=message.content or "",
            tool_calls=calls,
            prompt_tokens=response.prompt_eval_count or 0,
            completion_tokens=response.eval_count or 0,
        )


Step = ChatReply | Callable[[list[Message]], ChatReply]


class ScriptedChat:
    """Replays a fixed list of replies; a reply may be a function of the messages so far."""

    def __init__(self, steps: list[Step]) -> None:
        self.name = "scripted"
        self.steps = list(steps)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self, messages: list[Message], tools: list[dict] | None = None, schema: dict | None = None
    ) -> ChatReply:
        self.calls.append(
            {
                "messages": json.loads(json.dumps(messages, default=str)),
                "tools": tools,
                "schema": schema,
            }
        )
        if not self.steps:
            raise AssertionError("the scripted model ran out of replies")
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step
