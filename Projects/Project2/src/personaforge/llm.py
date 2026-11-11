"""
llm.py — the thin layer between a persona and a model.

Everything in this package that needs to "think" goes through the `LLM`
protocol below. There are two implementations:

    ClaudeLLM    real Claude calls. Needs ANTHROPIC_API_KEY.
    ScriptedLLM  returns canned strings. No key, no network, no cost.

That split is deliberate and it is the reason this project is testable. The
unit tests in `tests/` inject a ScriptedLLM, so they run in milliseconds for
free, and `main.py --offline` uses one too so you can see the whole app work
before you have configured anything.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

# Claude Opus 5 is the most capable model. While you are iterating you may want
# "claude-haiku-4-5" — a persona conversation makes one call per turn, and those
# add up. Say which model produced your submitted transcripts.
DEFAULT_MODEL = "claude-opus-5"


class LLM(Protocol):
    """Anything that can turn a system prompt plus a transcript into a reply."""

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        ...


# -----------------------------------------------------------------------------
# Tool calling
# -----------------------------------------------------------------------------
# Personas and dialogue only ever need text in, text out, so `LLM.complete` stays
# as it was. The orchestrator needs more: it hands the model a list of tools and
# gets back either prose or a request to call one. That is a second, separate
# protocol so a plain text model (ScriptedLLM) never has to pretend to do it.
#
# Responses are normalised to plain dicts in the Anthropic content-block shape
# ({"type": "text", ...} / {"type": "tool_use", ...}). Dicts are what the API
# accepts back in `messages`, they are trivial to build in a test, and they keep
# the SDK's own classes from leaking out of this module.


@dataclass
class ModelTurn:
    """One model response: its content blocks and why it stopped."""

    content: List[Dict[str, Any]]
    stop_reason: str = "end_turn"

    @property
    def text(self) -> str:
        """All the prose in the response, joined. Empty for a pure tool call."""
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """The tool_use blocks, in the order the model asked for them."""
        return [b for b in self.content if b.get("type") == "tool_use"]


class ToolCallingLLM(Protocol):
    """A model that can be offered tools and may ask to call them."""

    def respond_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> ModelTurn:
        ...


def text_turn(text: str) -> ModelTurn:
    """Build a prose-only response. Mostly for tests and the offline stub."""
    return ModelTurn(content=[{"type": "text", "text": text}], stop_reason="end_turn")


def tool_turn(*calls: tuple, text: str = "") -> ModelTurn:
    """
    Build a response that calls one or more tools, e.g.
    `tool_turn(("create_persona", {"description": "a nurse"}))`.

    Ids are generated so several calls in one turn stay distinguishable, which is
    what lets the orchestrator pair every tool_result with its tool_use.
    """
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    for i, (name, args) in enumerate(calls):
        content.append({"type": "tool_use", "id": f"toolu_{name}_{i}", "name": name, "input": dict(args)})
    return ModelTurn(content=content, stop_reason="tool_use")


@dataclass
class Usage:
    """
    Running token totals for one model client.

    A conversation costs one call per turn and the orchestrator adds one or more
    calls per user request, so "how much did that cost?" is a question you will
    want answered with numbers rather than guesses. Multiply by your model's
    per-million-token prices to get dollars.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response: Any) -> None:
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0

    def __str__(self) -> str:
        return f"{self.calls} calls · {self.input_tokens:,} input tokens · {self.output_tokens:,} output tokens"


# =============================================================================
# ⚠️  NEEDS YOUR OWN API KEY — AND SHIPPED UNVERIFIED
# =============================================================================
# REQUIRES A KEY. ClaudeLLM is every real model call in this package. Without
# ANTHROPIC_API_KEY it raises with instructions rather than failing quietly. Get
# a key at https://console.anthropic.com/ and put it in `.env` (copy
# `.env.example`). Calls are billed to you, and a conversation costs one call
# per turn, so six turns is six calls plus one per persona created.
#
# You do not need a key to work on this project: `uv run pytest` and
# `uv run main.py --offline` exercise the whole package for free.
#
# VERIFIED LIVE. `complete()` has been run against the real API end to end:
# personas are authored, saved, and hold character across a multi-turn
# conversation. One bug surfaced during that run, in persona.py rather than
# here — the model intermittently omits the closing '---' of the frontmatter,
# which the parser used to reject, silently producing a character called
# "Unnamed". The parser now tolerates it.
# =============================================================================


@dataclass
class ClaudeLLM:
    """
    Real Claude. One method, deliberately — this package only ever needs to say
    "here is who you are, here is the conversation so far, what do you say next".

    The client is created lazily so importing this module never requires a key.
    """

    model: str = DEFAULT_MODEL
    usage: Usage = field(default_factory=Usage)
    _client: Any = field(default=None, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                    "or run in offline mode: uv run main.py --offline"
                )
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        """Send one request and return the concatenated text of the reply."""
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        self.usage.add(response)

        # Claude can decline a request; content is empty or partial when it does.
        if response.stop_reason == "refusal":
            return "[declined to respond]"

        return "".join(block.text for block in response.content if block.type == "text").strip()

    def respond_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> ModelTurn:
        """
        One request with tools attached. The caller runs any tools the model asks
        for and sends the results back; this method never loops on its own, so
        the orchestrator stays in charge of how many round trips a request costs.
        """
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        self.usage.add(response)

        if response.stop_reason == "refusal":
            return text_turn("I can't help with that one. Try describing the characters differently.")

        # Keep only the two block types this package understands. Everything is
        # rebuilt as a plain dict so it can go straight back into `messages`.
        content: List[Dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)})
        return ModelTurn(content=content, stop_reason=response.stop_reason or "end_turn")


@dataclass
class ScriptedLLM:
    """
    A fake model that replays a fixed list of replies, then loops.

    Used by the tests and by `--offline`. It records every call it received in
    `.calls`, which makes it easy to assert that a persona's markdown actually
    reached the system prompt.
    """

    replies: List[str] = field(default_factory=lambda: ["(scripted reply)"])
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


@dataclass
class ScriptedToolLLM:
    """
    A fake tool-calling model for tests: replays a fixed list of `ModelTurn`s.

    Unlike ScriptedLLM it does not loop — running out of turns raises, because a
    test that makes more round trips than it scripted is a test that has found a
    bug in the orchestrator's loop. Every request is recorded in `.calls`.
    """

    turns: List[ModelTurn] = field(default_factory=list)
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def respond_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> ModelTurn:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if len(self.calls) > len(self.turns):
            raise AssertionError(f"ScriptedToolLLM ran out of turns after {len(self.turns)} call(s)")
        return self.turns[len(self.calls) - 1]


_DEMO_DIALOGUE = [
    "I understand. Could you tell me more about when the symptoms started?",
    "That has been going on for about three weeks now, mostly in the mornings.",
    "Have you noticed anything that makes it better or worse?",
    "Coffee seems to make it worse, and lying down helps a little.",
    "Thank you, that is useful. I would like to run a couple of tests.",
    "Whatever you think is best, doctor. I just want to feel normal again.",
]

_DEMO_PERSONA = """---
name: {name}
role: {role}
summary: a demo persona generated in offline mode
---

# {name}

## Background
This persona was produced by the offline stub, not by a model. Run without
`--offline` to get a real character written by Claude.

## Personality
- Placeholder
- Placeholder

## How they speak
- In whatever the scripted stub happens to return
"""


@dataclass
class DemoLLM:
    """
    The offline stub used by `--offline`.

    A single ScriptedLLM cannot serve both jobs: asked to author a persona it
    would return a line of dialogue, and you would get a character named
    "Unnamed". So this one looks at the system prompt and answers in the shape
    the caller expects — persona markdown for the author prompt, dialogue for
    everything else.
    """

    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        request = messages[-1]["content"] if messages else ""

        # The persona-author prompt is the only one that asks for frontmatter.
        if "role: <one lowercase word" in system:
            description = request.replace("Write a persona for:", "").strip()
            words = [w for w in description.split() if w.isalpha()]
            role = words[0].lower() if words else "person"
            return _DEMO_PERSONA.format(name=description[:40].title() or "Demo Persona", role=role)

        # The persona-editor prompt: keep the document, record the instruction,
        # so an offline edit visibly changes the file without inventing content.
        if "revise character specifications" in system:
            instruction, _, document = request.partition("\n\nCurrent document:\n\n")
            return document.rstrip() + f"\n\n## Notes\n- {instruction.replace('Instruction:', '').strip()}\n"

        if "stage manager" in system.lower():
            return _OFFLINE_HINT

        return _DEMO_DIALOGUE[(len(self.calls) - 1) % len(_DEMO_DIALOGUE)]

    def respond_with_tools(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> ModelTurn:
        """
        Pretend to be a tool-using model, so `--offline` exercises the real loop.

        This is where the old regular-expression router ended up: no longer
        deciding what the user meant in production, just standing in for the
        model when there is no model. It recognises the three phrasings the
        banner suggests and nothing else — that limit is the point of going live.
        """
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        last = messages[-1] if messages else {"content": ""}

        # A tool just ran: echo its result back as the reply, the way a real
        # model would summarise it.
        if isinstance(last["content"], list):
            results = [b.get("content", "") for b in last["content"] if b.get("type") == "tool_result"]
            return text_turn("\n\n".join(str(r) for r in results) or "Done.")

        text = str(last["content"]).strip().lower()

        create = re.search(r"\b(?:create|make|add|build|generate)\s+(?:a\s+|an\s+|new\s+)*persona\s*(.*)", text)
        if create and create.group(1).strip():
            return tool_turn(("create_persona", {"description": create.group(1).strip()}))

        # "make the doctor more skeptical" / "change maria: she is now 70"
        edit = re.search(r"\b(?:make|edit|change)\s+(?:the\s+)?([a-z0-9-]+)[\s:]+(.+)", text)
        if edit:
            return tool_turn(("edit_persona", {"name": edit.group(1), "instruction": edit.group(2).strip()}))

        if re.search(r"\b(list|show|who)\b.*\bpersona", text) or text in {"list", "ls"}:
            return tool_turn(("list_personas", {}))

        if re.search(r"\b(talk|conversation|converse|chat|discuss|speak|argue)\b", text):
            topic_match = re.search(r"\babout\s+(.*)$", text)
            who_match = re.search(r"\bbetween\s+(.*?)(?:\s+about\b|$)", text)
            who = [w.strip() for w in re.split(r"\s+and\s+|,", who_match.group(1)) if w.strip()] if who_match else []
            return tool_turn((
                "start_conversation",
                {"participants": who, "topic": topic_match.group(1).strip() if topic_match else "an initial consultation"},
            ))

        return text_turn(_OFFLINE_HINT)


_OFFLINE_HINT = (
    "Offline mode — I'm a stub that only understands a few phrasings. Try: "
    "\"create a persona doctor\", \"list personas\", or \"have them talk about the results\". "
    "Run without --offline and Claude reads anything you type."
)


def default_llm(offline: bool = False, model: Optional[str] = None) -> LLM:
    """Pick an implementation. Central so you only change this in one place."""
    if offline:
        return DemoLLM()
    return ClaudeLLM(model=model or DEFAULT_MODEL)
