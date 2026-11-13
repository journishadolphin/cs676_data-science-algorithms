# PersonaForge — Design Notes

**Author:** Nisha (Journisha Dolphin) · CS676 Project 2 · Version 0.2.0

> Checkpoint 2 technical report. All transcripts, numbers and quotes in sections 1–3 come
> from one live session on 2026-11-12 using `claude-haiku-4-5` for both routing and content
> (`uv run main.py --model claude-haiku-4-5`). The raw files are in
> [`docs/transcripts/`](transcripts/).

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

**Live example (prompt layer).** The roster held two doctors, Dr. Marcus Webb (doctor) and
Dr. Marcus Chen (resident), when I typed *"have the doctor talk to the patient about whether
she needs surgery"*. The router called no tool. It replied:

> Which doctor should lead—Dr. Webb (the experienced, warm internist) or Dr. Chen (the cautious
> resident)? Or do you want them both there together?

That cost one router call and no dialogue turns. The tool-layer backstop was not needed in
this run.

The same request also shows a miss. I wrote "she", but the only patient (Marcus Chen) is male.
The router did not point that out.

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

### The session

Every input I typed (the full console output is in
[`live-session-console.txt`](transcripts/live-session-console.txt)):

1. `I need a patient with chronic back pain who distrusts doctors and an overbooked, blunt internist`
   The router made two `create_persona` calls in one turn, producing
   [Marcus Chen](transcripts/marcus-chen.md) (patient) and Dr. Marcus Webb (doctor).
2. `have them talk about the MRI results`
   `start_conversation` ran with the default 6 turns:
   [transcript A](transcripts/conversation-the-doctor-delivers-the-mri-results-20261112-191534.md).
3. `make the doctor warmer, then have them talk again about the MRI results for 8 turns`
   One request, two steps: `edit_persona` rewrote
   [Dr. Marcus Webb](transcripts/dr-marcus-webb.md), then `start_conversation` ran with `turns: 8`:
   [transcript B](transcripts/conversation-the-doctor-delivers-the-mri-results-20261112-191609.md).
4. `add another doctor, a cautious first-year resident`
   Produced [Dr. Marcus Chen](transcripts/dr-marcus-chen.md) (resident).
5. `have the doctor talk to the patient about whether she needs surgery`
   The router asked a clarifying question (quoted in section 1).

On the routing side, every request mapped to the intended tools. In requests 1 and 3, a single
sentence produced several actions, which the regex router could not do.

### Transcript A: went well

The doctor opens bluntly ("your MRI came back clean… nine times out of ten, this resolves").
The patient pushes back with evidence rather than hostility, exactly as his persona file says
("he'll listen, but only if you give him evidence, not authority").

The doctor then concedes: "You're right. Eight years changes the conversation. I wasn't
listening to that part." Several of the patient's details come straight from his file: part-time
bookkeeping from home, online stretching routines, cannabis, three surgeries with one making
things worse. All six lines are pure dialogue, with no stage directions.

**Where it broke:**
- **Wrong fact.** The patient says he has managed the pain for "eight years". His file says he
  was injured at 32 and is 47, which is fifteen years.
- **Cut off mid-question.** The conversation stops on the patient's open question ("What am I
  looking for, and how do I know if it's working…?"). A fixed turn count ends it whether or
  not the question was answered.
- **Wrong summary.** The stage manager described the result as "they just clashed over the MRI
  findings", but the two actually converged. The router only receives a one-line tool result
  ("6 turns between … Transcript saved to …"), not the dialogue, so it made that description up.

### Transcript B: went badly

After the "warmer" edit, both agents broke the rule "Reply with dialogue only. No stage
directions, no asterisks, no narration." The doctor's first line begins "I pull up a chair and
sit down at eye level with you instead of standing", with the speech itself in quotation marks.
Nearly every later turn, from both characters, mixes first-person narration with quoted speech:
"I nod slowly", "I lean back slightly", "I tap my pen against my knee".

The conversation also contradicts transcript A:
- **Different MRI result.** The MRI is now "a disc herniation at L4-L5… pressing on the nerve
  root", where A said it was "clean".
- **Different PT history.** The patient has done physical therapy "three times", where in A it
  was twice.

Agents have no memory between conversations (`agent.py` YOUR TASK 1), and nothing records
facts established in one conversation, so each run invents its own medical findings. Like A,
B ends on an unanswered question, here about what happens if nothing changes in six weeks.

**What differed between A and B:** the edited doctor file and 8 turns instead of 6. Same model,
same topic, same patient file. The edited doctor file contains no stage directions, so the file
alone does not explain the narration. With one run of each, I can't tell whether the edit
caused it or it was run-to-run variation. Settling that would take several runs of each version
with everything else held fixed.

### Persona authoring

All three generated characters are named Marcus. The patient (Marcus Chen) and the resident
(Dr. Marcus Chen) share a first and last name, and the patient and the internist are both 47.
Each `create_persona` call runs the author prompt alone and never sees the roster, so it can't
avoid names that already exist.

`resolve_persona` keeps lookups correct here: the exact name "Marcus Chen" matches the patient,
while "Chen" alone raises `AmbiguousPersona`. But the duplicate names are confusing to read.

The "warmer" edit worked, but more broadly than asked. The summary changed from blunt and
overbooked to "A caring, compassionate internist…", and "blunt" disappeared. "Perpetually behind
schedule" survived in the background section.

## 3. Failure analysis and cost

### Measured cost of the session above

`/usage` reported: **28 calls · 38,347 input tokens · 5,535 output tokens.**

With one model for both roles, the counter does not split router calls from content calls.
The split below is counted from the session itself: each persona created or edited, and each
dialogue turn, is exactly one content call.

| Part of the session | Calls |
|---|---|
| Persona authoring (3 creates) and editing (1 edit) | 4 content |
| Transcript A (6 turns) + transcript B (8 turns) | 14 content |
| Routing (5 requests, including the final summaries) | 10 router (28 − 18) |
| **Total** | **28** |

At Anthropic's list price for Claude Haiku 4.5 ($1 per million input tokens, $5 per million
output tokens), the whole session cost about **$0.038 + $0.028 ≈ $0.07**. Routing added 10 calls
on top of 18 content calls, roughly 56% more calls than the content alone. Those router calls
are the price of replacing the regexes. `--router-model` exists to make them cheaper, but
here one model did both jobs.

I did not log tokens per call, so this run cannot show how input tokens grow from turn to turn.

### Where the system fails (observed above)

| Failure | Where | Likely cause |
|---|---|---|
| Stage directions and narration | Transcript B, both agents | Not established (see section 2) |
| Contradictory facts across conversations | MRI clean (A) vs herniation (B) | No memory; nothing shared between runs |
| Contradicts own persona file | "eight years" vs fifteen in the file | No check of dialogue against the persona |
| Conversation ends mid-question | A and B | Fixed turn count, no notion of "finished" |
| Inaccurate summary from the stage manager | After transcript A | Router sees only a one-line tool result |
| Duplicate names across personas | Marcus ×3 | Author prompt doesn't see the roster |
| Wrong pronoun not flagged | Request 5 | Not handled by the prompt or tools |

### Next fixes, in order of payoff for the effort
1. Pass the roster's names to the author prompt, to stop name collisions.
2. Include the last few lines of the transcript in the `start_conversation` tool result, so
   the router's summary is grounded in what was said.
3. Add an observer pass that flags narration and contradictions with the persona file
   (`conversation.py` YOUR TASK 4).

## 4. Tests: what they verify and what they don't

76 tests, all offline, run in about 0.1s. The new ones are in `tests/test_orchestrator.py`, `tests/test_tools.py`, and parts of `test_persona.py` and `test_conversation.py`.

**They verify:** every `tool_use` gets back a `tool_result` with the same id; several tool calls in one turn; multi-round chains; errors reaching the model as `is_error`; the round cap; history rollback when the API fails; trimming that never cuts between a `tool_use` and its result; that `ClaudeLLM` sends `tools=` and normalizes its response (using a fake Anthropic client); ambiguity and whole-word matching; edits that keep the persona's identity; failed turns keeping the transcript so far.

**They do not verify:** whether real Claude picks the right tool, asks good questions, or stays in character. Unit tests can't check that; transcripts can. Section 2 is that evidence: correct routing in all five requests, and character breaks in transcript B that every unit test would have passed.
