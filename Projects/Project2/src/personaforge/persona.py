"""
persona.py — a persona is a markdown file on disk.

That is the whole idea of this package. There is no persona database and no
in-memory registry. A persona is a `.md` file in `temp/`, and anything that can
read that file can wake the persona up. You can open one in a text editor,
change a sentence, and the agent behaves differently on the next run.

File format — frontmatter for the machine, markdown for the model:

    ---
    name: Maria Delgado
    role: patient
    summary: 58-year-old with poorly controlled type 2 diabetes
    ---

    # Maria Delgado

    ## Background
    ...

The frontmatter parser here is about fifteen lines and handles exactly the
three keys above. That is intentional — see YOUR TASK at the bottom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .llm import LLM

# Where persona files live. Relative to wherever you launched the app.
DEFAULT_PERSONA_DIR = Path("temp")


@dataclass
class PersonaSpec:
    """One persona: the metadata we index on, plus the markdown that is its mind."""

    name: str
    role: str
    summary: str
    body: str  # markdown; becomes the agent's system prompt
    # Where this spec was loaded from, if anywhere. Lets an edit write back to the
    # same file instead of guessing from the slug. Not part of equality: two
    # specs with the same content are the same persona wherever they live.
    path: Optional[Path] = field(default=None, compare=False, repr=False)

    def slug(self) -> str:
        """Filesystem-safe stem, e.g. 'Maria Delgado' -> 'maria-delgado'."""
        s = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return s or "unnamed"

    def to_markdown(self) -> str:
        """Serialize to the frontmatter + body format shown above."""
        return (
            "---\n"
            f"name: {self.name}\n"
            f"role: {self.role}\n"
            f"summary: {self.summary}\n"
            "---\n\n"
            f"{self.body.strip()}\n"
        )

    @classmethod
    def from_markdown(cls, text: str) -> "PersonaSpec":
        """
        Parse a persona file back into a spec.

        Missing frontmatter is tolerated: we fall back to sensible defaults and
        treat the whole file as the body, so a hand-written persona still loads.
        """
        meta = {"name": "Unnamed", "role": "unknown", "summary": ""}
        text = text.strip()

        # Models sometimes wrap the whole document in a code fence.
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
            text = text.strip()

        body = text

        if text.startswith("---"):
            rest = text[3:].lstrip("\n")
            closing = re.search(r"^---\s*$", rest, re.MULTILINE)
            if closing:
                # Well-formed: frontmatter is everything up to the closing fence.
                front, body = rest[:closing.start()], rest[closing.end():]
            else:
                # The closing '---' is missing, which the model does drop from
                # time to time. Rather than throw the whole document away, read
                # `key: value` lines from the top and treat the first line that
                # is not one as the start of the body.
                front_lines, body_lines, in_front = [], [], True
                for line in rest.splitlines():
                    if in_front and re.match(r"^[A-Za-z_][A-Za-z0-9_ -]*:", line):
                        front_lines.append(line)
                    elif in_front and not line.strip():
                        continue            # blank line inside the header block
                    else:
                        in_front = False
                        body_lines.append(line)
                front, body = "\n".join(front_lines), "\n".join(body_lines)

            for line in front.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip().lower()
                    if key in meta:
                        meta[key] = value.strip()

        return cls(name=meta["name"], role=meta["role"], summary=meta["summary"], body=body.strip())


class PersonaError(Exception):
    """Base class for persona lookup problems. The message says what to do next."""


class PersonaNotFound(PersonaError):
    """No saved persona matches the name the caller gave."""


class AmbiguousPersona(PersonaError):
    """
    More than one persona matches. `candidates` lists them so the caller (or the
    orchestrating model) can ask the user which one they meant rather than
    silently picking the first.
    """

    def __init__(self, needle: str, candidates: List[PersonaSpec]) -> None:
        self.candidates = candidates
        names = ", ".join(f"{c.name} ({c.role}, {c.path.name if c.path else c.slug()})" for c in candidates)
        super().__init__(f"'{needle}' matches more than one persona: {names}. Use a full name or file name.")


def save_persona(spec: PersonaSpec, directory: Path = DEFAULT_PERSONA_DIR, overwrite: bool = False) -> Path:
    """
    Write the persona to `<directory>/<slug>.md`, creating the folder if needed.

    A second persona with the same name gets `<slug>-2.md` rather than replacing
    the first; creating two patients called Maria should leave you with two
    patients. Pass `overwrite=True` (as edits do) to replace the file in place.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if overwrite and spec.path is not None:
        path = Path(spec.path)
    else:
        path = directory / f"{spec.slug()}.md"
        n = 2
        while path.exists() and not overwrite:
            path = directory / f"{spec.slug()}-{n}.md"
            n += 1

    path.write_text(spec.to_markdown(), encoding="utf-8")
    spec.path = path
    return path


def load_persona(path: Path) -> PersonaSpec:
    """Read one persona file from disk, remembering where it came from."""
    spec = PersonaSpec.from_markdown(Path(path).read_text(encoding="utf-8"))
    spec.path = Path(path)
    return spec


def list_personas(directory: Path = DEFAULT_PERSONA_DIR) -> List[PersonaSpec]:
    """Every persona currently saved, sorted by filename. Empty if none yet."""
    directory = Path(directory)
    if not directory.exists():
        return []
    # Transcripts share the folder with personas, so skip them by prefix.
    return [load_persona(p) for p in sorted(directory.glob("*.md")) if not p.name.startswith("conversation")]


def find_persona(name: str, directory: Path = DEFAULT_PERSONA_DIR) -> Optional[PersonaSpec]:
    """
    Look a persona up by name, slug, or role — case-insensitively.

    Users type "the doctor" or "maria", not exact filenames, so we accept a
    loose match. This is crude; see YOUR TASK.
    """
    needle = name.strip().lower()
    for spec in list_personas(directory):
        if needle in (spec.name.lower(), spec.slug(), spec.role.lower()):
            return spec
    for spec in list_personas(directory):
        if needle in spec.name.lower() or needle in spec.role.lower():
            return spec
    return None


def _words(text: str) -> List[str]:
    """Lowercase alphanumeric tokens: 'Dr. Samuel Reyes' -> ['dr', 'samuel', 'reyes']."""
    return re.findall(r"[a-z0-9]+", text.lower())


def resolve_persona(name: str, directory: Path = DEFAULT_PERSONA_DIR) -> PersonaSpec:
    """
    Find exactly one persona for a loose reference like "the doctor" or "reyes".

    Matching runs from strictest to loosest and stops at the first tier with any
    hits: exact file stem, exact name, exact role, then whole-word match against
    name and role. Whole words rather than substrings, so "doc" never matches
    "Dorothy". If the winning tier has more than one hit, that is an error — the
    caller should ask, not guess.

    :raises PersonaNotFound:  nothing matched; the message lists who exists.
    :raises AmbiguousPersona: several matched; `.candidates` lists them.
    """
    specs = list_personas(directory)
    needle = name.strip().lower()
    # "the doctor" and "dr reyes" are how people talk; drop filler words.
    words = [w for w in _words(needle) if w not in {"the", "a", "an", "dr", "mr", "mrs", "ms"}] or _words(needle)

    tiers = [
        lambda s: s.path is not None and s.path.stem == needle,
        lambda s: s.name.lower() == needle,
        lambda s: s.role.lower() == needle or (len(words) == 1 and s.role.lower() == words[0]),
        lambda s: bool(words) and all(w in _words(s.name) + _words(s.role) for w in words),
    ]
    for matches in tiers:
        hits = [s for s in specs if matches(s)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise AmbiguousPersona(name, hits)

    available = ", ".join(f"{s.name} ({s.role})" for s in specs) or "none yet"
    raise PersonaNotFound(f"No persona matches '{name}'. Available: {available}.")


# -----------------------------------------------------------------------------
# Generating a persona with the model
# -----------------------------------------------------------------------------

_AUTHOR_SYSTEM = """You write character specifications for a multi-agent simulation.

Given a short description, write a persona as a markdown document. Output ONLY the
markdown — no preamble, no code fences.

Use exactly this structure:

---
name: <a specific, realistic full name>
role: <one lowercase word, e.g. patient, doctor, teacher, customer>
summary: <one line, under 15 words>
---

# <name>

## Background
Two or three sentences. Be concrete and specific — age, situation, history.

## Personality
Three or four bullet points about temperament and outlook.

## How they speak
Two or three bullets on voice: vocabulary, sentence length, verbal habits.

## What they want
One or two sentences on their goal in a conversation.

## What they avoid
One or two sentences on what they will not say or do.

Write a specific individual, not a type. Give them a quirk. Avoid stereotypes,
and do not make them a spokesperson for a demographic."""


def generate_persona(description: str, llm: LLM) -> PersonaSpec:
    """
    Ask the model to author a persona from a one-line description.

    :param description: e.g. "a patient with type 2 diabetes who distrusts doctors"
    :param llm:         any LLM implementation (real or scripted)
    :return:            the parsed PersonaSpec — not yet saved to disk
    """
    markdown = llm.complete(
        system=_AUTHOR_SYSTEM,
        messages=[{"role": "user", "content": f"Write a persona for: {description}"}],
        max_tokens=2048,
    )

    spec = PersonaSpec.from_markdown(markdown)

    # If the model ignored the frontmatter format we still want something usable,
    # so fall back to the raw description rather than failing the whole command.
    if spec.name == "Unnamed":
        spec.name = description[:40].strip().title() or "Unnamed"
        spec.summary = description.strip()

    return spec


_EDITOR_SYSTEM = """You revise character specifications for a multi-agent simulation.

You will receive an existing persona document and an instruction describing a change.
Return the COMPLETE revised document — frontmatter and every section — with the change
applied and everything else kept as close to the original as possible. Keep the same
name unless the instruction explicitly renames the character. Output ONLY the markdown:
no preamble, no code fences, no commentary on what you changed."""


def edit_persona(spec: PersonaSpec, instruction: str, llm: LLM) -> PersonaSpec:
    """
    Apply a natural-language change ("make her more skeptical") to a persona.

    The model rewrites the whole document rather than patching it, because a
    change in temperament usually touches several sections at once. If the
    rewrite comes back without a usable name, the original name is kept: an edit
    must never turn Maria into "Unnamed". Returns a new spec; the caller saves it.
    """
    markdown = llm.complete(
        system=_EDITOR_SYSTEM,
        messages=[{"role": "user", "content": f"Instruction: {instruction}\n\nCurrent document:\n\n{spec.to_markdown()}"}],
        max_tokens=2048,
    )
    revised = PersonaSpec.from_markdown(markdown)

    # Be generous with the model's formatting but strict about identity: a
    # missing field falls back to the original, and an empty body means the
    # rewrite failed, so the original body survives.
    return PersonaSpec(
        name=spec.name if revised.name == "Unnamed" else revised.name,
        role=spec.role if revised.role == "unknown" else revised.role,
        summary=revised.summary or spec.summary,
        body=revised.body or spec.body,
        path=spec.path,
    )


# =============================================================================
# STATUS — the skeleton's YOUR TASK list for this module
# =============================================================================
#
#  1. Loose matching / indistinguishable personas: `resolve_persona` matches
#     whole words, strictest tier first, and raises AmbiguousPersona instead
#     of guessing. `find_persona` keeps its old behaviour for compatibility.  DONE
#  2. `save_persona` no longer overwrites: a same-name persona gets -2, -3.  DONE
#  3. The frontmatter parser still handles three fixed keys and no quoting.   OPEN
#  4. Generated output is still not validated (one-word role, sections).
#     `edit_persona` does refuse to lose the name, role or body.             PARTIAL
#  5. Personas can be edited from inside the app (`edit_persona` tool).       DONE
#  6. Personas do not change during a conversation.                          OPEN
