"""
tools.py — what the orchestrating model is allowed to do, and how it gets done.

The orchestrator does not decide what the user meant; the model does. This
module defines the menu it chooses from (`TOOL_SCHEMAS`, in the JSON-schema
shape the Anthropic Messages API expects) and the `Toolbox` that carries out a
chosen call against the rest of the package.

Two rules shape everything here:

  * A tool never raises. Every outcome — success or failure — comes back as a
    `ToolOutcome` whose text goes to the model as a `tool_result`. A failure is
    marked `is_error` and its message says what to do next ("Available: ..."),
    so the model can recover by asking the user or retrying, instead of the
    whole request falling over.
  * Inputs are validated here, not trusted. The schema tells the model what to
    send; it does not guarantee that it will.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agent import agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import LLM
from .persona import (
    PersonaError,
    edit_persona,
    generate_persona,
    list_personas,
    resolve_persona,
    save_persona,
)

# Conversation length limits. Each turn is one paid model call, so the upper
# bound is a cost guard as much as a sanity check.
MIN_TURNS, MAX_TURNS, DEFAULT_TURNS = 2, 20, 6

# Descriptions are written for the model, not for people: they say when to use
# each tool and what to do when the request is unclear. That prose is the
# routing logic now, so it is worth as much care as code.
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "create_persona",
        "description": (
            "Invent a new character and save it as a markdown persona file. Call once per character; "
            "for 'two doctors and a nurse' call it three times with three distinct descriptions. "
            "Pass along every detail the user gave (role, temperament, situation, name if given). "
            "Do not call this if the user has not said what kind of person they want — ask first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Who to create, e.g. 'a patient with chronic back pain who distrusts doctors'.",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "list_personas",
        "description": "List every saved persona with its name, role, summary and file. Use when the user asks who exists.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "show_persona",
        "description": "Read one persona's full markdown, to answer questions about a character or before editing it.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Name, role, or file name of the persona."}},
            "required": ["name"],
        },
    },
    {
        "name": "edit_persona",
        "description": (
            "Change an existing persona in place, e.g. 'make the doctor more skeptical' or 'she is now 70'. "
            "The file is rewritten and the change applies to every later conversation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name, role, or file name of the persona to change."},
                "instruction": {"type": "string", "description": "The change to make, in plain language."},
            },
            "required": ["name", "instruction"],
        },
    },
    {
        "name": "start_conversation",
        "description": (
            "Run a conversation between two or more existing personas and save the transcript. "
            "Participants are spoken to in the order given, so put whoever should open first. "
            "If the user says 'them' or 'the doctor' and the roster makes it unambiguous, use those personas; "
            "if several personas could match, ask which ones instead of guessing. "
            "Create any missing personas first, then call this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "Names (or unambiguous roles) of the personas who take part, opener first.",
                },
                "topic": {
                    "type": "string",
                    "description": "The situation, phrased as a scene, e.g. 'the doctor delivers the MRI results'.",
                },
                "turns": {
                    "type": "integer",
                    "minimum": MIN_TURNS,
                    "maximum": MAX_TURNS,
                    "description": f"Total lines of dialogue. Default {DEFAULT_TURNS}; use more only if asked.",
                },
            },
            "required": ["participants", "topic"],
        },
    },
]


@dataclass
class ToolOutcome:
    """The result of one tool call, in the shape the orchestrator needs."""

    content: str  # what the model reads back as the tool_result
    is_error: bool = False
    events: List[str] = field(default_factory=list)  # side effects worth showing the user
    conversation: Optional[Conversation] = None


def _error(message: str) -> ToolOutcome:
    return ToolOutcome(content=message, is_error=True)


class Toolbox:
    """
    Executes tool calls against the package.

    `llm` here is the *content* model — the one that writes personas and speaks
    as them — which need not be the same model that routes. The toolbox holds no
    state of its own beyond the persona folder; everything it knows is on disk.
    """

    def __init__(self, llm: LLM, persona_dir: Path) -> None:
        self.llm = llm
        self.persona_dir = Path(persona_dir)

    def run(
        self,
        name: str,
        args: Dict[str, Any],
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> ToolOutcome:
        """Dispatch one call. Unknown tools and unexpected crashes come back as errors, never exceptions."""
        handlers = {
            "create_persona": lambda: self.create_persona(args.get("description", "")),
            "list_personas": self.list_personas,
            "show_persona": lambda: self.show_persona(args.get("name", "")),
            "edit_persona": lambda: self.edit_persona(args.get("name", ""), args.get("instruction", "")),
            "start_conversation": lambda: self.start_conversation(
                args.get("participants") or [], args.get("topic", ""), args.get("turns"), on_turn, on_start
            ),
        }
        handler = handlers.get(name)
        if handler is None:
            return _error(f"Unknown tool '{name}'. Available tools: {', '.join(handlers)}.")
        try:
            return handler()
        except PersonaError as exc:
            return _error(str(exc))
        except Exception as exc:  # noqa: BLE001 — report to the model; let it tell the user
            return _error(f"{name} failed: {type(exc).__name__}: {exc}")

    # -- tools ----------------------------------------------------------------

    def create_persona(self, description: str) -> ToolOutcome:
        """Author a persona with the content model, save it, report where it landed."""
        if not str(description).strip():
            return _error("A description is required. Ask the user what kind of person to create.")
        spec = generate_persona(str(description), self.llm)
        path = save_persona(spec, self.persona_dir)
        return ToolOutcome(
            content=f"Created {spec.name} ({spec.role}) — {spec.summary}. Saved to {path.name}.",
            events=[f"wrote {path}"],
        )

    def list_personas(self) -> ToolOutcome:
        specs = list_personas(self.persona_dir)
        if not specs:
            return ToolOutcome(content="No personas yet.")
        lines = [f"- {s.name} ({s.role}) — {s.summary} [{s.path.name if s.path else s.slug()}]" for s in specs]
        return ToolOutcome(content=f"{len(specs)} persona(s):\n" + "\n".join(lines))

    def show_persona(self, name: str) -> ToolOutcome:
        spec = resolve_persona(str(name), self.persona_dir)
        return ToolOutcome(content=spec.to_markdown())

    def edit_persona(self, name: str, instruction: str) -> ToolOutcome:
        if not str(instruction).strip():
            return _error("An instruction is required. Ask the user what to change.")
        original = resolve_persona(str(name), self.persona_dir)
        revised = edit_persona(original, str(instruction), self.llm)
        path = save_persona(revised, self.persona_dir, overwrite=True)
        return ToolOutcome(
            content=f"Updated {revised.name} ({revised.role}) — {revised.summary}. Rewrote {path.name}.",
            events=[f"rewrote {path}"],
        )

    def start_conversation(
        self,
        participants: List[str],
        topic: str,
        turns: Any = None,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> ToolOutcome:
        """Resolve every participant first, so a typo fails before any paid turn runs."""
        topic = str(topic).strip()
        if not topic:
            return _error("A topic is required. Ask the user what the characters should talk about.")

        try:
            turns = DEFAULT_TURNS if turns is None else int(turns)
        except (TypeError, ValueError):
            return _error(f"turns must be a whole number between {MIN_TURNS} and {MAX_TURNS}.")
        if not MIN_TURNS <= turns <= MAX_TURNS:
            return _error(f"turns must be between {MIN_TURNS} and {MAX_TURNS}; got {turns}.")

        if isinstance(participants, str):
            participants = [participants]
        if not participants:
            # Convenience for "have them talk": with exactly two personas, "them"
            # can only mean one thing. With more, the model must say who.
            existing = list_personas(self.persona_dir)
            if len(existing) != 2:
                names = ", ".join(s.name for s in existing) or "none yet"
                return _error(f"Say which personas should talk. At least two are needed. Available: {names}.")
            specs = existing
        else:
            specs = [resolve_persona(str(p), self.persona_dir) for p in participants]

        if len({s.path for s in specs}) < len(specs):
            return _error("The same persona was listed twice. Each participant must be a different persona.")
        if len(specs) < 2:
            return _error("A conversation needs at least two personas. Create another one first.")

        agents = [agent_from_spec(s, self.llm) for s in specs]
        if on_start is not None:
            on_start(topic)
        conversation = run_conversation(agents, topic=topic, turns=turns, on_turn=on_turn)
        path = save_transcript(conversation, self.persona_dir)

        names = " and ".join(s.name for s in specs)
        summary = f"{len(conversation.turns)} turns between {names}. Transcript saved to {path.name}."
        if conversation.error:
            # Partial transcripts are still saved; the model is told so it can
            # explain what happened rather than claiming success.
            summary += f" The conversation stopped early: {conversation.error}"
        return ToolOutcome(
            content=summary,
            is_error=bool(conversation.error) and not conversation.turns,
            events=[f"wrote {path}"],
            conversation=conversation,
        )
