"""
PersonaForge — an agent-to-agent persona simulation package.

The idea in four lines:

    spec  = generate_persona("a patient with chronic back pain", llm)
    path  = save_persona(spec)              # writes temp/<name>.md
    agent = agent_from_spec(spec, llm)      # markdown + model = a mind
    run_conversation([doctor, patient], topic="the test results")

A persona is a markdown file. An agent is that file plus a model. A conversation
is agents taking turns. The `Orchestrator` puts a tool-using model in front of
all of it, so a user can ask for any of the above in plain language:

    Orchestrator(llm).handle("make two nurses and have them argue about the rota")
"""

from .agent import PersonaAgent, agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import (
    LLM,
    ClaudeLLM,
    ModelTurn,
    ScriptedLLM,
    ScriptedToolLLM,
    ToolCallingLLM,
    Usage,
    default_llm,
    text_turn,
    tool_turn,
)
from .orchestrator import Orchestrator, Result
from .persona import (
    AmbiguousPersona,
    PersonaError,
    PersonaNotFound,
    PersonaSpec,
    edit_persona,
    find_persona,
    generate_persona,
    list_personas,
    load_persona,
    resolve_persona,
    save_persona,
)
from .tools import TOOL_SCHEMAS, Toolbox, ToolOutcome

# 0.2.0: the orchestrator routes with tool use instead of regular expressions.
__version__ = "0.2.0"

__all__ = [
    "PersonaAgent",
    "agent_from_spec",
    "Conversation",
    "Turn",
    "run_conversation",
    "save_transcript",
    "LLM",
    "ToolCallingLLM",
    "ClaudeLLM",
    "ScriptedLLM",
    "ScriptedToolLLM",
    "ModelTurn",
    "Usage",
    "text_turn",
    "tool_turn",
    "default_llm",
    "Orchestrator",
    "Result",
    "TOOL_SCHEMAS",
    "Toolbox",
    "ToolOutcome",
    "PersonaSpec",
    "PersonaError",
    "PersonaNotFound",
    "AmbiguousPersona",
    "edit_persona",
    "find_persona",
    "resolve_persona",
    "generate_persona",
    "list_personas",
    "load_persona",
    "save_persona",
]
