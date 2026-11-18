# ADR 0005: A local model through Ollama, behind a one-method interface

**Status:** accepted

## Context
The portfolio must cost nothing to run, and the eval has to run in CI on every schedule.

## Decision
The agent depends on a `ChatModel` protocol with a single `chat(messages, tools, schema)` method.
`OllamaChat` serves `qwen2.5:3b` (tool calling and JSON-schema output, fits a CPU runner) at
temperature 0 with a fixed seed. `ScriptedChat` replays fixed replies so tests can drive every
branch of the loop: tool errors, unknown tools, step limits, invalid JSON, fabricated citations.

## Consequences
- Free and reproducible: the eval workflow installs Ollama on a GitHub runner and pulls the model.
- A 3B model makes more tool-use mistakes than a frontier model; the eval reports them instead
  of hiding them, and the citation verifier stops fabricated sources from reaching the answer.
- Switching to another provider is one class implementing `chat`.
