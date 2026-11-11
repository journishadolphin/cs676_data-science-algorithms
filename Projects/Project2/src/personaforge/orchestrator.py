"""
orchestrator.py — the agent you talk to.

You type "I want a patient and a doctor, and have them argue about the
diagnosis." Something has to turn that into calls into this package. In the
skeleton that something was a stack of regular expressions. Here it is the
model itself: the orchestrator offers Claude the tools in `tools.py`, runs
whichever ones it asks for, sends the results back, and repeats until the model
answers in prose.

    user text ─▶ model ─┬─▶ prose ──────────────────────────▶ reply to user
                        └─▶ tool_use ─▶ Toolbox ─▶ tool_result ─▶ model ─▶ ...

What the regexes could not do and this can: several actions in one sentence
("two doctors and a nurse"), clarifying questions ("which doctor?"), recovery
from a failed tool call, and memory of what was created earlier in the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .conversation import Conversation, Turn
from .llm import LLM, ToolCallingLLM
from .persona import DEFAULT_PERSONA_DIR, list_personas
from .tools import TOOL_SCHEMAS, Toolbox

# The roster is filled in fresh on every request (see `_system_prompt`), so the
# model can resolve "the doctor" without spending a round trip on list_personas.
ORCHESTRATOR_SYSTEM = """You are the stage manager of a persona simulation workshop.

The user builds fictional characters and has them talk to each other. You act on their
requests by calling tools: create personas, list or show them, edit them, and start
conversations between them. Each persona is a markdown file the user can also edit by hand.

How to work:
- Act when the request is clear. "Two doctors and a nurse" means three create_persona calls.
- Ask ONE short question instead of guessing when something essential is missing or
  ambiguous: no description of who to create, or several personas that could be "the doctor".
- Do the steps in order. If the user wants characters who do not exist yet to talk, create
  them first, then start the conversation.
- After tools run, reply in one or two sentences saying what happened. The user already saw
  the dialogue as it streamed, so never repeat or summarise the transcript line by line.
- If a tool returns an error, explain it plainly and say what the user can do next.
- If the user is just chatting, answer briefly and suggest a next step.

Current personas on disk:
{roster}"""

# One user request may legitimately take several round trips (create, create,
# converse, then reply). This caps a confused model from looping forever on
# your bill.
MAX_TOOL_ROUNDS = 8

# History is kept across requests so the model remembers what it did, but it is
# trimmed to this many messages so a long session does not grow without limit.
MAX_HISTORY_MESSAGES = 40


@dataclass
class Result:
    """What the orchestrator did, in a form the UI can render."""

    reply: str
    events: List[str] = field(default_factory=list)
    conversation: Optional[Conversation] = None


class Orchestrator:
    """
    A tool-using agent over the PersonaForge package.

    :param llm:         the content model — writes personas and speaks as them.
    :param persona_dir: where persona and transcript files live.
    :param router:      the model that reads the user's request and picks tools.
                        Defaults to `llm`. Separating them lets you route with a
                        cheap model and write dialogue with a strong one.
    """

    def __init__(
        self,
        llm: LLM,
        persona_dir: Path = DEFAULT_PERSONA_DIR,
        router: Optional[ToolCallingLLM] = None,
    ) -> None:
        router = router if router is not None else llm
        if not hasattr(router, "respond_with_tools"):
            raise TypeError(
                f"{type(router).__name__} cannot call tools. Pass a router that implements "
                "respond_with_tools (ClaudeLLM, DemoLLM, or ScriptedToolLLM in tests)."
            )
        self.llm = llm
        self.router: ToolCallingLLM = router  # type: ignore[assignment]
        self.persona_dir = Path(persona_dir)
        self.toolbox = Toolbox(llm=llm, persona_dir=self.persona_dir)
        self.history: List[Dict[str, Any]] = []

    def _system_prompt(self) -> str:
        """The instructions plus today's roster, read from disk so hand edits show up."""
        specs = list_personas(self.persona_dir)
        roster = "\n".join(f"- {s.name} ({s.role}) — {s.summary}" for s in specs) or "(none yet)"
        return ORCHESTRATOR_SYSTEM.format(roster=roster)

    def handle(
        self,
        text: str,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """
        Act on one user request and return what happened.

        Runs the tool-use loop: ask the model, run any tools it requests, send the
        results back, repeat until it replies in prose or the round cap is hit.
        `on_turn` and `on_start` are passed through to conversations so dialogue
        streams to the UI as it is produced.

        If the model call fails part-way, history is rolled back to before this
        request, so the next request starts from a valid state — but anything a
        tool already finished (a saved persona, a saved transcript) is kept and
        reported in `events`.
        """
        checkpoint = len(self.history)
        self.history.append({"role": "user", "content": text})
        events: List[str] = []
        conversation: Optional[Conversation] = None

        try:
            for _ in range(MAX_TOOL_ROUNDS):
                turn = self.router.respond_with_tools(self._system_prompt(), self.history, TOOL_SCHEMAS)
                # An empty assistant message is rejected by the API on the next
                # call, so a blank reply is recorded as a placeholder.
                self.history.append({"role": "assistant", "content": turn.content or [{"type": "text", "text": "…"}]})

                if not turn.tool_calls:
                    self._trim_history()
                    return Result(reply=turn.text or "Done.", events=events, conversation=conversation)

                # Every tool_use must be answered by a tool_result with the same
                # id, all in the single user message that follows.
                results = []
                for call in turn.tool_calls:
                    events.append(_describe_call(call))
                    outcome = self.toolbox.run(call["name"], call.get("input") or {}, on_turn, on_start)
                    events.extend(outcome.events)
                    conversation = outcome.conversation or conversation
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": outcome.content,
                        "is_error": outcome.is_error,
                    })
                self.history.append({"role": "user", "content": results})

            # Out of rounds. Close the exchange with an assistant message so the
            # history still alternates correctly for the next request.
            reply = f"I stopped after {MAX_TOOL_ROUNDS} tool steps without finishing. Try a smaller request."
            self.history.append({"role": "assistant", "content": reply})
            self._trim_history()
            return Result(reply=reply, events=events, conversation=conversation)

        except Exception as exc:  # noqa: BLE001 — a failed model call should not corrupt the session
            del self.history[checkpoint:]
            return Result(
                reply=(
                    f"The model call failed ({type(exc).__name__}: {exc}). "
                    "This request was rolled back; any files already written are kept. Try again."
                ),
                events=events,
                conversation=conversation,
            )

    def _trim_history(self) -> None:
        """
        Drop the oldest exchanges once history passes MAX_HISTORY_MESSAGES.

        Cuts only at a plain user message (a fresh request), never between a
        tool_use and its tool_result — the API rejects an orphaned result. The
        roster in the system prompt means forgetting old turns never makes the
        model forget who exists.
        """
        if len(self.history) <= MAX_HISTORY_MESSAGES:
            return
        for i in range(len(self.history) - MAX_HISTORY_MESSAGES, len(self.history)):
            message = self.history[i]
            if message["role"] == "user" and isinstance(message["content"], str):
                del self.history[:i]
                return


def _describe_call(call: Dict[str, Any]) -> str:
    """A short, human-readable form of a tool call for the UI, e.g. `create_persona(a nurse)`."""
    values = [", ".join(map(str, v)) if isinstance(v, list) else str(v) for v in (call.get("input") or {}).values()]
    args = ", ".join(v for v in values if v)
    if len(args) > 70:
        args = args[:67] + "..."
    return f"{call['name']}({args})"


# =============================================================================
# STATUS — the skeleton's YOUR TASK list for this module
# =============================================================================
#
#  1. Regex router replaced with tool use.                              DONE
#  2. Clarifying questions: the system prompt tells the model to ask,
#     and tools return errors that say "ask the user ..." when an
#     essential input is missing.                                      DONE
#  3. Pattern order no longer decides meaning; "list them, then have
#     them talk" is two tool calls in sequence.                        DONE
#  4. No silent "first two personas": an ambiguous reference raises
#     AmbiguousPersona, which reaches the model as an error to ask on.  DONE
#  5. `turns` is a tool argument, bounded 2–20.                        DONE
#  6. History is trimmed at request boundaries, and the roster is
#     injected into the system prompt on every request.                DONE
#
#  Open: the roster changes the system prompt every time a persona is
#  created, which defeats prompt caching. Worth measuring before fixing.
