from __future__ import annotations

"""
Simulated interviewee using Stanford's genagents library.

Two usage modes:
  1. Load a pre-built agent from disk (GSS demographic agents from the genagents repo):
       agent = load_agent("eval/agents/gss_agent_001")

  2. Build a custom agent from a demographics dict + memory strings:
       agent = create_agent(
           demographics={"first_name": "Maria", "age": 24, ...},
           memories=["She is 24 weeks pregnant with her first child.",
                     "She stopped exercising after the first trimester due to nausea."],
       )

Then get a response each turn:
       text = respond(agent, transcript)
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# genagents path + settings bootstrap
# ---------------------------------------------------------------------------
# genagents is a git submodule at <project_root>/genagents/.
# Its Python package lives at genagents/genagents/ and it also requires
# genagents/simulation_engine/ to be importable.  Both resolve once we add
# the submodule root to sys.path.
_GENAGENTS_ROOT = Path(__file__).parent.parent / "genagents"
if not _GENAGENTS_ROOT.exists():
    raise ImportError(
        "genagents submodule not found. Initialise it with:\n"
        "  git submodule update --init\n"
    )
if str(_GENAGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_GENAGENTS_ROOT))

_SETTINGS_PATH = _GENAGENTS_ROOT / "simulation_engine" / "settings.py"
_api_key = os.environ.get("OPENAI_API_KEY", "")
_llm_vers = os.environ.get("GENAGENTS_LLM_VERS", "gpt-4o-mini")
_SETTINGS_PATH.write_text(f"""\
from pathlib import Path

OPENAI_API_KEY = {_api_key!r}
KEY_OWNER = "agentic-interviewer"
DEBUG = False
MAX_CHUNK_SIZE = 4
LLM_VERS = {_llm_vers!r}
BASE_DIR = str(Path(__file__).resolve().parent.parent)
POPULATIONS_DIR = f"{{BASE_DIR}}/agent_bank/populations"
LLM_PROMPT_DIR = f"{{BASE_DIR}}/simulation_engine/prompt_template"
""")

try:
    from genagents.genagents import GenerativeAgent
except ImportError as e:
    raise ImportError(
        "Could not import genagents. Make sure:\n"
        "  1. git submodule update --init\n"
        "  2. pip install -r requirements.txt\n"
        "  3. OPENAI_API_KEY is set in your environment\n"
    ) from e


def load_agent(agent_folder: str | Path) -> GenerativeAgent:
    """Load a pre-built genagents agent from a saved folder (scratch.json, nodes.json, etc.)."""
    return GenerativeAgent(agent_folder=str(agent_folder))


def reset_agent_memory(agent: GenerativeAgent) -> None:
    """
    Clear the agent's memory stream (nodes + embeddings) in-place.
    scratch.json (demographics) is preserved.
    Call agent.save() afterwards to persist the reset to disk.
    """
    agent.memory_stream.seq_nodes = []
    agent.memory_stream.id_to_node = {}
    agent.memory_stream.embeddings = {}


def create_agent(demographics: dict, memories: list[str]) -> GenerativeAgent:
    """
    Build a new agent from scratch.

    demographics: dict of key-value attributes stored in agent scratch memory.
                  Common keys: first_name, last_name, age, occupation, background.
    memories:     List of factual/experiential statements seeded into the agent's
                  memory stream (use present-tense or past-tense prose).
    """
    agent = GenerativeAgent()
    agent.update_scratch(demographics)
    for memory in memories:
        agent.remember(memory)
    return agent


def respond(agent: GenerativeAgent, transcript: list[dict]) -> str:
    """
    Generate the agent's next response given the conversation so far,
    then append the exchange to the agent's memory stream.

    transcript: list of {"role": "interviewer"|"interviewee", "content": str}
                as stored in InterviewState.transcript.
                The last entry must be the current interviewer turn.

    Within a run: curr_dialogue provides full context for consistency.
    Across runs: accumulated memories surface via semantic retrieval, simulating
    realistic recall of prior interviews (with natural fading of less relevant detail).
    Call agent.save() after each run to persist memories across protocol runs.
    """
    agent_name = _get_name(agent)
    curr_dialogue = []
    for turn in transcript:
        speaker = "Interviewer" if turn["role"] == "interviewer" else agent_name
        curr_dialogue.append((speaker, turn["content"]))

    response = agent.utterance(curr_dialogue=curr_dialogue, context="")

    last_question = transcript[-1]["content"] if transcript else ""
    agent.remember(f"Interviewer: {last_question}\n\n{agent_name}: {response}\n\n")

    return response


def _get_name(agent: GenerativeAgent) -> str:
    """Return a display name for the agent, falling back to 'Participant'."""
    scratch = getattr(agent, "scratch", {})
    if isinstance(scratch, dict):
        first = scratch.get("first_name", "")
        last = scratch.get("last_name", "")
        name = f"{first} {last}".strip()
        return name if name else "Participant"
    # scratch may be an object
    try:
        return agent.get_fullname()
    except Exception:
        return "Participant"
