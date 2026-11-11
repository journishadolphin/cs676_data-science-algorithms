"""
Tests for the tool-use loop in the orchestrator.

The routing model is a ScriptedToolLLM: each test scripts exactly which tools
the "model" asks for, then checks that the orchestrator ran them, fed the
results back correctly, and kept its history valid. What these tests cannot
check is whether real Claude would make the same choices — that is a question
about prompt quality, answered by live transcripts, not by unit tests.
"""

import pytest

from personaforge import Orchestrator, PersonaSpec, save_persona
from personaforge.llm import ClaudeLLM, DemoLLM, ScriptedLLM, ScriptedToolLLM, text_turn, tool_turn
from personaforge.orchestrator import MAX_HISTORY_MESSAGES, MAX_TOOL_ROUNDS

PERSONA_MD = """---
name: {name}
role: {role}
summary: a summary
---

# {name}
Body text.
"""


def seed(tmp_path, *people):
    for name, role in people:
        save_persona(PersonaSpec(name=name, role=role, summary="s", body="b"), tmp_path)


def orchestrator(tmp_path, turns, content_replies=None):
    """An orchestrator whose router replays `turns` and whose personas say `content_replies`."""
    router = ScriptedToolLLM(turns=list(turns))
    content = ScriptedLLM(replies=content_replies or ["a line"])
    return Orchestrator(llm=content, persona_dir=tmp_path, router=router), router, content


# -- the loop -----------------------------------------------------------------


def test_plain_chat_makes_one_call_and_runs_no_tools(tmp_path):
    orch, router, content = orchestrator(tmp_path, [text_turn("Who would you like to build?")])
    result = orch.handle("hello, what is this thing?")

    assert result.reply == "Who would you like to build?"
    assert len(router.calls) == 1 and content.calls == []


def test_tool_result_is_sent_back_with_matching_id(tmp_path):
    orch, router, _ = orchestrator(
        tmp_path,
        [tool_turn(("list_personas", {})), text_turn("You have none yet.")],
    )
    orch.handle("who do I have?")

    tool_use = router.calls[1]["messages"][-2]["content"][0]
    tool_result = router.calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == tool_use["id"]
    assert "No personas yet" in tool_result["content"]


def test_one_sentence_three_personas(tmp_path):
    """'Two doctors and a nurse' — the request the regex router could never handle."""
    replies = [PERSONA_MD.format(name=n, role=r) for n, r in
               [("Ana Ruiz", "doctor"), ("Ben Okafor", "doctor"), ("Cleo Park", "nurse")]]
    orch, router, _ = orchestrator(
        tmp_path,
        [
            tool_turn(
                ("create_persona", {"description": "a cautious doctor"}),
                ("create_persona", {"description": "a bold doctor"}),
                ("create_persona", {"description": "a nurse"}),
            ),
            text_turn("Created three personas."),
        ],
        content_replies=replies,
    )
    result = orch.handle("make it two doctors and a nurse")

    assert sorted(p.stem for p in tmp_path.glob("*.md")) == ["ana-ruiz", "ben-okafor", "cleo-park"]
    assert len(router.calls[1]["messages"][-1]["content"]) == 3  # three results, one message
    assert result.reply == "Created three personas."


def test_create_then_converse_in_one_request(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"))
    orch, _, _ = orchestrator(
        tmp_path,
        [
            tool_turn(("create_persona", {"description": "a patient"})),
            tool_turn(("start_conversation", {"participants": ["Alice", "Bob"], "topic": "the chart", "turns": 2})),
            text_turn("They talked."),
        ],
        content_replies=[PERSONA_MD.format(name="Bob Brown", role="patient"), "line"],
    )
    lines = []
    result = orch.handle("add a patient and have them talk about the chart", on_turn=lines.append)

    assert result.conversation is not None and len(lines) == 2
    assert any(e.startswith("create_persona(") for e in result.events)
    assert any(e.startswith("start_conversation(") for e in result.events)


def test_error_result_lets_the_model_ask_a_question(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Dan Diaz", "doctor"), ("Bob Brown", "patient"))
    orch, router, _ = orchestrator(
        tmp_path,
        [
            tool_turn(("start_conversation", {"participants": ["the doctor", "Bob"], "topic": "x"})),
            text_turn("Which doctor — Alice Adams or Dan Diaz?"),
        ],
    )
    result = orch.handle("have the doctor talk to Bob")

    sent_back = router.calls[1]["messages"][-1]["content"][0]
    assert sent_back["is_error"] is True
    assert result.reply.startswith("Which doctor")
    assert result.conversation is None


def test_roster_is_in_the_system_prompt(tmp_path):
    seed(tmp_path, ("Maria Delgado", "patient"))
    orch, router, _ = orchestrator(tmp_path, [text_turn("ok")])
    orch.handle("hi")
    assert "Maria Delgado (patient)" in router.calls[0]["system"]


def test_history_remembers_earlier_requests(tmp_path):
    orch, router, _ = orchestrator(tmp_path, [text_turn("first"), text_turn("second")])
    orch.handle("one")
    orch.handle("two")

    texts = [m["content"] for m in router.calls[1]["messages"] if isinstance(m["content"], str)]
    assert texts == ["one", "two"]


# -- limits and failures ------------------------------------------------------------


def test_runaway_tool_loop_is_capped(tmp_path):
    orch, router, _ = orchestrator(tmp_path, [tool_turn(("list_personas", {}))] * MAX_TOOL_ROUNDS)
    result = orch.handle("list forever")

    assert len(router.calls) == MAX_TOOL_ROUNDS
    assert "stopped after" in result.reply
    assert orch.history[-1]["role"] == "assistant"  # still alternates for the next request


class BrokenRouter:
    """A router whose API call always fails."""

    def respond_with_tools(self, system, messages, tools, max_tokens=1024):
        raise ConnectionError("API unavailable")


def test_api_failure_rolls_back_history_and_reports(tmp_path):
    orch = Orchestrator(llm=ScriptedLLM(), persona_dir=tmp_path, router=BrokenRouter())
    result = orch.handle("create a doctor")

    assert "model call failed" in result.reply and "ConnectionError" in result.reply
    assert orch.history == []


def test_failure_after_a_tool_ran_keeps_its_events(tmp_path):
    """The persona was saved before the follow-up call died; the user should hear about it."""

    class DiesSecondTime(ScriptedToolLLM):
        def respond_with_tools(self, *a, **k):
            if self.calls:
                raise TimeoutError("timed out")
            return super().respond_with_tools(*a, **k)

    router = DiesSecondTime(turns=[tool_turn(("create_persona", {"description": "a nurse"}))])
    content = ScriptedLLM(replies=[PERSONA_MD.format(name="Cleo Park", role="nurse")])
    result = Orchestrator(llm=content, persona_dir=tmp_path, router=router).handle("make a nurse")

    assert (tmp_path / "cleo-park.md").exists()
    assert any("wrote" in e for e in result.events)


def test_history_is_trimmed_at_a_request_boundary(tmp_path):
    orch, _, _ = orchestrator(tmp_path, [text_turn("ok")] * (MAX_HISTORY_MESSAGES + 10))
    for i in range(MAX_HISTORY_MESSAGES):
        orch.handle(f"message {i}")

    assert len(orch.history) <= MAX_HISTORY_MESSAGES
    assert orch.history[0]["role"] == "user" and isinstance(orch.history[0]["content"], str)


def test_text_only_model_is_rejected_with_advice(tmp_path):
    with pytest.raises(TypeError, match="respond_with_tools"):
        Orchestrator(llm=ScriptedLLM(), persona_dir=tmp_path)


# -- offline mode drives the same loop --------------------------------------------------


def test_offline_session_end_to_end(tmp_path):
    """The banner's four lines work with no key, through the real tool loop."""
    orch = Orchestrator(llm=DemoLLM(), persona_dir=tmp_path)

    assert "Created" in orch.handle("create a persona patient with chronic back pain").reply
    assert "Created" in orch.handle("create a persona doctor who is direct and busy").reply
    assert "2 persona(s)" in orch.handle("list personas").reply
    result = orch.handle("have them talk about the test results")

    assert result.conversation is not None and len(result.conversation.turns) == 6


def test_offline_stub_admits_its_limits(tmp_path):
    reply = Orchestrator(llm=DemoLLM(), persona_dir=tmp_path).handle("I fancy a skeptical data scientist").reply
    assert "Offline mode" in reply


# -- the real client, with the network faked out --------------------------------------


class FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeAnthropic:
    """Stands in for `anthropic.Anthropic()` so ClaudeLLM's request/response handling is tested offline."""

    def __init__(self, response):
        self.response, self.requests = response, []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def test_claude_llm_sends_tools_and_normalises_blocks():
    response = FakeBlock(
        stop_reason="tool_use",
        usage=FakeBlock(input_tokens=120, output_tokens=30),
        content=[
            FakeBlock(type="text", text="On it."),
            FakeBlock(type="tool_use", id="toolu_1", name="list_personas", input={}),
        ],
    )
    fake = FakeAnthropic(response)
    llm = ClaudeLLM(model="claude-haiku-4-5", _client=fake)

    turn = llm.respond_with_tools("sys", [{"role": "user", "content": "hi"}], [{"name": "list_personas"}])

    assert fake.requests[0]["tools"] == [{"name": "list_personas"}]
    assert turn.text == "On it."
    assert turn.tool_calls == [{"type": "tool_use", "id": "toolu_1", "name": "list_personas", "input": {}}]
    assert (llm.usage.calls, llm.usage.input_tokens, llm.usage.output_tokens) == (1, 120, 30)
