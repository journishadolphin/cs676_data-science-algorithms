"""
conversation.py — put two or more agents in a room and let them talk.

The loop is deliberately dumb: agents take strict turns in the order given, for
a fixed number of turns, and then it stops. No agent decides who speaks next and
no agent can interrupt. Making that smarter is one of the more interesting
things you can do with this project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from .agent import PersonaAgent


@dataclass
class Turn:
    """One line of dialogue."""

    speaker: str
    text: str


@dataclass
class Conversation:
    """A completed dialogue, plus the setup that produced it."""

    topic: str
    participants: List[str]
    turns: List[Turn] = field(default_factory=list)
    # Set when a turn failed part-way. The turns before it are kept, so a model
    # error on turn five costs you turn five, not the four before it.
    error: Optional[str] = None

    def transcript(self) -> str:
        """Render as plain text, suitable for saving or printing."""
        header = f"# Conversation: {self.topic}\n\nParticipants: {', '.join(self.participants)}\n\n"
        body = "\n\n".join(f"**{t.speaker}:** {t.text}" for t in self.turns)
        if self.error:
            body += f"\n\n> Stopped early after {len(self.turns)} turn(s): {self.error}"
        return header + body


def run_conversation(
    agents: List[PersonaAgent],
    topic: str,
    turns: int = 6,
    on_turn: Optional[Callable[[Turn], None]] = None,
) -> Conversation:
    """
    Run a round-robin conversation between the given agents.

    :param agents:  two or more PersonaAgents. They speak in list order.
    :param topic:   seeds the opening; each agent sees it as the situation.
    :param turns:   total lines of dialogue, split across the agents.
    :param on_turn: called after each line, so a UI can print as it goes rather
                    than waiting for the whole conversation to finish.
    """
    if len(agents) < 2:
        raise ValueError("a conversation needs at least two agents")

    conversation = Conversation(topic=topic, participants=[a.name for a in agents])

    # The topic is injected as a scene-setting turn from a narrator. Every agent
    # sees it as context but nobody has to answer it directly.
    history: List[Turn] = [Turn(speaker="Scene", text=topic)]

    for i in range(turns):
        speaker = agents[i % len(agents)]
        try:
            text = speaker.respond(history)
        except Exception as exc:  # noqa: BLE001 — any model failure ends the scene, not the program
            conversation.error = f"{speaker.name}'s turn failed ({type(exc).__name__}: {exc})"
            break
        turn = Turn(speaker=speaker.name, text=text)

        history.append(turn)
        conversation.turns.append(turn)

        if on_turn is not None:
            on_turn(turn)

    return conversation


def save_transcript(conversation: Conversation, directory, filename: Optional[str] = None) -> Path:
    """
    Write the transcript to disk next to the personas that produced it.

    Without an explicit filename each conversation gets its own file, named for
    its topic and the time it was saved, e.g.
    `conversation-the-mri-results-20261110-142233.md`, so a new run never
    overwrites the last one. The `conversation` prefix is how `list_personas`
    tells transcripts apart from personas in the same folder.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if filename is None:
        topic = re.sub(r"[^a-z0-9]+", "-", conversation.topic.lower()).strip("-")[:40] or "untitled"
        filename = f"conversation-{topic}-{datetime.now():%Y%m%d-%H%M%S}.md"
        # Two runs in the same second (tests, mostly) must not collide either.
        stem, n = filename[:-3], 2
        while (directory / filename).exists():
            filename = f"{stem}-{n}.md"
            n += 1
    path = directory / filename
    path.write_text(conversation.transcript(), encoding="utf-8")
    return path


# =============================================================================
# STATUS — the skeleton's YOUR TASK list for this module
# =============================================================================
#
#  1. Turn-taking is still strict round-robin.                               OPEN
#  2. Conversations still end after N turns (N is now set per request).      OPEN
#  3. The "Scene" pseudo-speaker is still how the topic is injected.         OPEN
#  4. No observer / judge agent yet.                                         OPEN
#  5. Each transcript gets its own file: conversation-<topic>-<time>.md.     DONE
#  6. A failed turn ends the conversation but keeps every earlier turn,
#     and the saved transcript notes where and why it stopped.               DONE
