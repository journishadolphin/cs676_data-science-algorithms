# PersonaForge — Design Notes

**Author:** Nisha (Journisha Dolphin) · CS676 Project 2 · Version 0.2.0

> Working draft for the Checkpoint 2 technical report. Sections marked **TODO (live)**
> need numbers or transcripts from a run with a real API key.

## 1. The extension: tool use replaces the regex router

### What changed

| Before (skeleton 0.1.0) | After (0.2.0) |
|---|---|
| `Orchestrator.handle()` tried a stack of `re.search` patterns; first match won | `handle()` runs a tool-use loop: the model picks tools, the orchestrator runs them and returns `tool_result`s until the model replies in prose |
| One action per sentence | Several tool calls per turn, several rounds per request ("create a patient, then have her talk to the doctor") |
| No clarifying questions | Tools return errors that say "ask the user …"; the system prompt tells the model to ask one question when something is ambiguous |
| `_converse` silently used the first two personas | `resolve_persona` raises `AmbiguousPersona` with the candidates; the model sees that and asks |
| `turns` hard-coded to 6 | `turns` is a tool argument, bounded to 2–20 |
| History grew without limit, only used for chat | History is shared by every request, trimmed at request boundaries; the persona roster is put into the system prompt on every request |

New and changed modules:

- `tools.py` (new): `TOOL_SCHEMAS` plus a `Toolbox` that runs them. **Tools never raise** — every outcome becomes a `tool_result`, with `is_error: true` and a next-step hint on failure.
- `llm.py`: a second protocol, `ToolCallingLLM.respond_with_tools()`, kept separate from `LLM.complete()`, so text-only fakes don't have to pretend they can call tools. Responses come back as plain dicts (`ModelTurn`), the same shape the API accepts back. `ScriptedToolLLM` is the test fake. `Usage` counts calls and tokens.
- `orchestrator.py`: the loop, a round cap, history trimming, rollback on failure.
- `persona.py`: `resolve_persona` (whole-word matching, strictest match first, ambiguity is an error), `edit_persona`, and `save_persona` no longer overwrites a persona that has the same name.
- `conversation.py`: if a turn fails, the turns before it are kept; every transcript gets its own file name.

### Tool schemas

Five tools (full JSON in `src/personaforge/tools.py`):

| Tool | Inputs | Why it exists |
|---|---|---|
| `create_persona` | `description` | Called once per character, so "two doctors and a nurse" means three calls |
| `list_personas` | — | Explicit "who do I have?" |
| `show_persona` | `name` | Lets the model answer "what does Maria want?" from the file itself |
| `edit_persona` | `name`, `instruction` | "Make the doctor more skeptical" without opening an editor (skeleton YOUR TASK persona #5) |
| `start_conversation` | `participants[]` (2+), `topic`, `turns` (2–20) | Participants are listed in speaking order; turns are capped to limit cost |

The tool **descriptions** now do the routing logic. They say when to use each tool *and when to stop and ask instead*.

### What the model does with an ambiguous request

The handling has two layers:

1. **Prompt layer.** The system prompt says to ask one short question when an essential input is missing. The roster in the prompt lets the model see on its own that there are two doctors.
2. **Tool layer (the backstop).** If the model guesses anyway, `resolve_persona` raises `AmbiguousPersona`. The model gets back `'the doctor' matches more than one persona: Ana Ruiz (doctor, ana-ruiz.md), Ben Okafor (…)` as an error and turns it into a question. This path is tested in `test_error_result_lets_the_model_ask_a_question`.

A participant name that doesn't exist fails **before any paid dialogue turn runs** (`test_unknown_participant_fails_before_any_turn_runs`).

**TODO (live):** paste a real exchange showing each layer at work.

### Alternatives I rejected

- **Keep the regexes and add an LLM fallback.** It's cheaper per request. But the order of the patterns still decides what a sentence means, and multi-step requests still fail. Rejected.
- **Ask the model for one JSON "intent" object.** One call, simple to parse. But it can't chain steps (create, then converse) or react to a failed lookup, and parsing the JSON becomes a new failure mode (the README's missing-`---` bug, again). Tool use gives structured arguments *and* a loop.
- **One "do anything" tool with a free-text command.** That just moves the parsing problem into my own code.
- **Let tools raise exceptions.** One bad name would end the whole request. Returning errors to the model lets it recover in the same request.

### Trade-offs I accepted

- **Cost:** every request now costs at least one router call, and a tool request costs at least two (the call, then the summary). The `--router-model` flag lets a cheap model route while a strong one writes dialogue.
- **Prompt caching:** the roster in the system prompt changes whenever a persona is created, which breaks prompt-cache hits. I accepted that to save the `list_personas` round trip; it's worth measuring.
- **Offline mode:** `DemoLLM.respond_with_tools` still uses regexes. They now only stand in for the model when there is no API key, and never decide anything in live mode. Offline mode therefore runs the *real* tool loop.

## 2. Transcripts (annotated)

**TODO (live):** at least one conversation that went well and one that went badly, with notes on the difference. Saved transcripts land in `temp/conversation-<topic>-<timestamp>.md`.

## 3. Failure analysis and cost

**TODO (live):** run a fixed session, then `/usage`, and record:

| Scenario | Router calls | Content calls | Input tokens | Output tokens | $ (at current price list) |
|---|---|---|---|---|---|
| create 2 personas + 6-turn conversation | | | | | |
| "two doctors and a nurse, have them argue" (3 creates + 6 turns) | | | | | |

Expected lower bound: creating a persona = 1 content call + ≥2 router calls; an N-turn conversation = N content calls, each re-sending the full transcript, so input tokens grow roughly quadratically with N.

Known weak spots to check live: the model narrating instead of speaking (stage directions), agents agreeing too quickly, and the router repeating the transcript in its summary despite the instruction not to.

## 4. Tests: what they verify and what they don't

76 tests, all offline, run in about 0.1s. The new ones are in `tests/test_orchestrator.py`, `tests/test_tools.py`, and parts of `test_persona.py` and `test_conversation.py`.

**They verify:** every `tool_use` gets back a `tool_result` with the same id; several tool calls in one turn; multi-round chains; errors reaching the model as `is_error`; the round cap; history rollback when the API fails; trimming that never cuts between a `tool_use` and its result; that `ClaudeLLM` sends `tools=` and normalizes its response (using a fake Anthropic client); ambiguity and whole-word matching; edits that keep the persona's identity; failed turns keeping the transcript so far.

**They do not verify:** whether real Claude picks the right tool, asks good questions, or stays in character. Unit tests can't check that; transcripts can.
