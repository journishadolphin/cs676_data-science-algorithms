"""
Tests for the conversation loop and the orchestrator's intent routing.

Everything runs against ScriptedLLM, so the whole suite is offline and free.
"""

import pytest

from personaforge import (
    Orchestrator,
    PersonaSpec,
    agent_from_spec,
    run_conversation,
    save_persona,
    save_transcript,
)
from personaforge.llm import ScriptedLLM

PERSONA_MD = """---
name: {name}
role: {role}
summary: a summary
---

# {name}
Body text.
"""


def two_agents(replies=None):
    llm = ScriptedLLM(replies=replies or ["line A", "line B"])
    a = agent_from_spec(PersonaSpec(name="Alice", role="doctor", summary="", body="A"), llm)
    b = agent_from_spec(PersonaSpec(name="Bob", role="patient", summary="", body="B"), llm)
    return a, b, llm


def test_conversation_alternates_speakers():
    a, b, _ = two_agents()
    convo = run_conversation([a, b], topic="the results", turns=4)

    assert [t.speaker for t in convo.turns] == ["Alice", "Bob", "Alice", "Bob"]


def test_conversation_respects_turn_count():
    a, b, _ = two_agents()
    assert len(run_conversation([a, b], topic="x", turns=5).turns) == 5


def test_conversation_needs_two_agents():
    a, _, _ = two_agents()
    with pytest.raises(ValueError):
        run_conversation([a], topic="x", turns=2)


def test_on_turn_fires_per_turn():
    a, b, _ = two_agents()
    seen = []
    run_conversation([a, b], topic="x", turns=3, on_turn=seen.append)

    assert len(seen) == 3
    assert seen[0].speaker == "Alice"


def test_topic_reaches_the_agents():
    """The scene turn must be visible to the first speaker."""
    a, b, llm = two_agents()
    run_conversation([a, b], topic="UNIQUE_TOPIC_MARKER", turns=1)

    first_call = llm.calls[0]["messages"]
    assert any("UNIQUE_TOPIC_MARKER" in m["content"] for m in first_call)


def test_transcript_contains_every_line():
    a, b, _ = two_agents(replies=["hello", "hi there"])
    convo = run_conversation([a, b], topic="greetings", turns=2)
    text = convo.transcript()

    assert "Alice" in text and "Bob" in text
    assert "hello" in text and "hi there" in text


def test_save_transcript_writes_a_file(tmp_path):
    a, b, _ = two_agents()
    convo = run_conversation([a, b], topic="x", turns=2)
    path = save_transcript(convo, tmp_path)

    assert path.exists()
    assert "Alice" in path.read_text(encoding="utf-8")


# -- robustness ---------------------------------------------------------------


class FailsOnCall:
    """A model that answers normally until call `n`, then raises — an API outage mid-scene."""

    def __init__(self, n):
        self.n, self.count = n, 0

    def complete(self, system, messages, max_tokens=2048):
        self.count += 1
        if self.count >= self.n:
            raise ConnectionError("API unavailable")
        return f"line {self.count}"


def test_failed_turn_keeps_the_turns_before_it():
    llm = FailsOnCall(n=4)
    a = agent_from_spec(PersonaSpec(name="Alice", role="doctor", summary="", body="A"), llm)
    b = agent_from_spec(PersonaSpec(name="Bob", role="patient", summary="", body="B"), llm)

    convo = run_conversation([a, b], topic="x", turns=6)

    assert [t.text for t in convo.turns] == ["line 1", "line 2", "line 3"]
    assert "Bob's turn failed" in convo.error
    assert "Stopped early after 3 turn(s)" in convo.transcript()


def test_transcripts_do_not_overwrite_each_other(tmp_path):
    a, b, _ = two_agents()
    first = save_transcript(run_conversation([a, b], topic="the MRI results", turns=2), tmp_path)
    second = save_transcript(run_conversation([a, b], topic="the MRI results", turns=2), tmp_path)

    assert first != second
    assert first.name.startswith("conversation-the-mri-results-")
    assert first.exists() and second.exists()


# Orchestrator tests live in test_orchestrator.py; tool tests in test_tools.py.
