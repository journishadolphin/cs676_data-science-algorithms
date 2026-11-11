"""
Tests for the Toolbox — the code that runs whatever tool the model picked.

These call the tools directly, with no routing model involved, so a failure
here is a bug in the tool itself rather than in how the model chose it. The
content model is a ScriptedLLM, so nothing touches the network.
"""

from personaforge import TOOL_SCHEMAS, PersonaSpec, Toolbox, list_personas, load_persona, save_persona
from personaforge.llm import DemoLLM, ScriptedLLM

PERSONA_MD = """---
name: {name}
role: {role}
summary: {summary}
---

# {name}
Body text.
"""


def seed(tmp_path, *people):
    """Save (name, role) pairs as personas and return their paths."""
    return [save_persona(PersonaSpec(name=n, role=r, summary="s", body=f"I am {n}."), tmp_path) for n, r in people]


# -- schemas ------------------------------------------------------------------


def test_every_schema_has_a_handler(tmp_path):
    """A schema the toolbox cannot run would surface as 'Unknown tool' mid-session."""
    box = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path)
    for schema in TOOL_SCHEMAS:
        assert "Unknown tool" not in box.run(schema["name"], {}).content


def test_schemas_are_valid_api_shape():
    for schema in TOOL_SCHEMAS:
        assert set(schema) == {"name", "description", "input_schema"}
        assert schema["input_schema"]["type"] == "object"


def test_unknown_tool_is_an_error_not_an_exception(tmp_path):
    outcome = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run("delete_everything", {})
    assert outcome.is_error and "Available tools" in outcome.content


# -- create / list / show -------------------------------------------------------


def test_create_persona_saves_a_file(tmp_path):
    llm = ScriptedLLM(replies=[PERSONA_MD.format(name="Maria Delgado", role="patient", summary="back pain")])
    outcome = Toolbox(llm=llm, persona_dir=tmp_path).run("create_persona", {"description": "a patient"})

    assert not outcome.is_error
    assert "Maria Delgado" in outcome.content
    assert (tmp_path / "maria-delgado.md").exists()
    assert any("wrote" in e for e in outcome.events)


def test_create_persona_without_description_asks_for_one(tmp_path):
    outcome = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run("create_persona", {"description": "  "})
    assert outcome.is_error and "Ask the user" in outcome.content


def test_create_same_name_twice_keeps_both(tmp_path):
    llm = ScriptedLLM(replies=[PERSONA_MD.format(name="Maria Delgado", role="patient", summary="s")])
    box = Toolbox(llm=llm, persona_dir=tmp_path)
    box.run("create_persona", {"description": "a patient"})
    box.run("create_persona", {"description": "another patient"})

    assert sorted(p.name for p in tmp_path.glob("*.md")) == ["maria-delgado-2.md", "maria-delgado.md"]


def test_list_personas_tool(tmp_path):
    seed(tmp_path, ("Maria Delgado", "patient"))
    outcome = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run("list_personas", {})
    assert "Maria Delgado (patient)" in outcome.content


def test_show_persona_returns_the_markdown(tmp_path):
    seed(tmp_path, ("Maria Delgado", "patient"))
    outcome = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run("show_persona", {"name": "maria"})
    assert "I am Maria Delgado." in outcome.content


def test_show_unknown_persona_lists_who_exists(tmp_path):
    seed(tmp_path, ("Maria Delgado", "patient"))
    outcome = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run("show_persona", {"name": "zaphod"})
    assert outcome.is_error and "Available: Maria Delgado (patient)" in outcome.content


# -- edit -------------------------------------------------------------------------


def test_edit_persona_rewrites_the_same_file(tmp_path):
    (path,) = seed(tmp_path, ("Samuel Reyes", "doctor"))
    revised = PERSONA_MD.format(name="Samuel Reyes", role="doctor", summary="now deeply skeptical")
    outcome = Toolbox(llm=ScriptedLLM(replies=[revised]), persona_dir=tmp_path).run(
        "edit_persona", {"name": "the doctor", "instruction": "make him skeptical"}
    )

    assert not outcome.is_error
    assert load_persona(path).summary == "now deeply skeptical"
    assert len(list(tmp_path.glob("*.md"))) == 1  # edited in place, not duplicated


def test_edit_that_returns_garbage_keeps_identity(tmp_path):
    """A model that answers with prose must not rename the persona to 'Unnamed'."""
    (path,) = seed(tmp_path, ("Samuel Reyes", "doctor"))
    box = Toolbox(llm=ScriptedLLM(replies=["Sure! Here is a more skeptical version."]), persona_dir=tmp_path)
    box.run("edit_persona", {"name": "Samuel Reyes", "instruction": "skeptical"})

    spec = load_persona(path)
    assert spec.name == "Samuel Reyes" and spec.role == "doctor"


def test_offline_edit_is_visible_in_the_file(tmp_path):
    (path,) = seed(tmp_path, ("Samuel Reyes", "doctor"))
    Toolbox(llm=DemoLLM(), persona_dir=tmp_path).run(
        "edit_persona", {"name": "doctor", "instruction": "is running forty minutes late"}
    )
    assert "running forty minutes late" in path.read_text(encoding="utf-8")


# -- start_conversation -----------------------------------------------------------


def test_conversation_streams_and_saves(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"))
    seen, started = [], []
    outcome = Toolbox(llm=ScriptedLLM(replies=["a line"]), persona_dir=tmp_path).run(
        "start_conversation",
        {"participants": ["Bob", "the doctor"], "topic": "the chart", "turns": 4},
        on_turn=seen.append,
        on_start=started.append,
    )

    assert outcome.conversation is not None
    assert outcome.conversation.participants == ["Bob Brown", "Alice Adams"]  # order given = speaking order
    assert len(seen) == 4 and started == ["the chart"]
    assert list(tmp_path.glob("conversation-the-chart-*.md"))


def test_three_way_conversation(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"), ("Carol Chen", "nurse"))
    outcome = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run(
        "start_conversation", {"participants": ["Alice", "Bob", "Carol"], "topic": "x", "turns": 3}
    )
    assert [t.speaker for t in outcome.conversation.turns] == ["Alice Adams", "Bob Brown", "Carol Chen"]


def test_unknown_participant_fails_before_any_turn_runs(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"))
    llm = ScriptedLLM()
    outcome = Toolbox(llm=llm, persona_dir=tmp_path).run(
        "start_conversation", {"participants": ["Zaphod", "Bob"], "topic": "lunch"}
    )
    assert outcome.is_error and "No persona matches 'Zaphod'" in outcome.content
    assert llm.calls == []  # no paid turns spent on a doomed conversation


def test_ambiguous_participant_is_reported_with_candidates(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Dan Diaz", "doctor"), ("Bob Brown", "patient"))
    outcome = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run(
        "start_conversation", {"participants": ["the doctor", "Bob"], "topic": "x"}
    )
    assert outcome.is_error
    assert "Alice Adams" in outcome.content and "Dan Diaz" in outcome.content


def test_them_means_the_only_two_personas(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"))
    outcome = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run(
        "start_conversation", {"participants": [], "topic": "x", "turns": 2}
    )
    assert outcome.conversation is not None


def test_them_with_three_personas_asks_who(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"), ("Carol Chen", "nurse"))
    outcome = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run(
        "start_conversation", {"participants": [], "topic": "x"}
    )
    assert outcome.is_error and "Say which personas" in outcome.content


def test_same_persona_twice_is_rejected(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"))
    outcome = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run(
        "start_conversation", {"participants": ["Alice", "alice-adams"], "topic": "x"}
    )
    assert outcome.is_error and "twice" in outcome.content


def test_turn_count_is_bounded(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"))
    box = Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path)
    for bad in (1, 500, "lots"):
        outcome = box.run("start_conversation", {"participants": ["Alice", "Bob"], "topic": "x", "turns": bad})
        assert outcome.is_error and "turns" in outcome.content


def test_transcripts_are_not_listed_as_personas(tmp_path):
    seed(tmp_path, ("Alice Adams", "doctor"), ("Bob Brown", "patient"))
    Toolbox(llm=ScriptedLLM(), persona_dir=tmp_path).run(
        "start_conversation", {"participants": ["Alice", "Bob"], "topic": "x", "turns": 2}
    )
    assert [s.name for s in list_personas(tmp_path)] == ["Alice Adams", "Bob Brown"]
