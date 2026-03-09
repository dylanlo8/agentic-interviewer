from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when the script is run directly (python eval/simulate.py)
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env BEFORE any project imports so OPENAI_API_KEY is available when
# eval.interviewee generates genagents/simulation_engine/settings.py
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

"""
Run a single simulated interview and save the transcript to eval/results/.

Usage:
    python eval/simulate.py \\
        --protocol notebooks/protocol.json \\
        --agent eval/agents/gss_agent_001 \\
        --interviewer agentic \\
        --minutes-per-turn 2.0

The --interviewer flag selects which interviewer drives the conversation:
    agentic     — the multi-agent system (default)
    scripted    — scripted baseline (no probing)
    single_llm  — single GPT-4o-mini baseline
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

from ai_interviewer.llm import LLMConfig
from ai_interviewer.protocol import load_protocol
from ai_interviewer.runner import run_interview_headless
from eval.interviewee import load_agent, respond

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def _agentic_run(protocol, cfg, agent, minutes_per_turn) -> list[dict]:
    """Run interview with the multi-agent system."""
    state = run_interview_headless(
        protocol=protocol,
        cfg=cfg,
        get_response=lambda q: _agent_respond(agent, q, state_ref),
        minutes_per_turn=minutes_per_turn,
    )
    return state.transcript


def _agent_respond(agent, question: str, state_ref: list) -> str:
    """Helper used by the agentic runner to pass the live transcript to the agent."""
    # state_ref[0] is set to the InterviewState by a wrapper below
    transcript = state_ref[0].transcript if state_ref else []
    # The interviewer turn hasn't been appended to transcript yet at call time;
    # append it manually so the agent sees the full exchange.
    ctx = transcript + [{"role": "interviewer", "content": question}]
    return respond(agent, ctx)


def run_simulation(
    protocol_path: Path,
    agent_folder: Path,
    interviewer_id: str = "agentic",
    minutes_per_turn: float = 2.0,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Path:
    """
    Run one simulated interview and write the transcript JSON to eval/results/.
    Returns the path of the saved transcript file.
    """
    protocol = load_protocol(protocol_path)
    agent = load_agent(agent_folder)
    agent_id = Path(agent_folder).name

    cfg = LLMConfig(
        topic_eval_model=os.environ.get("TOPIC_EVAL_MODEL", model),
        topic_eval_base_url=os.environ.get("TOPIC_EVAL_BASE_URL") or None,
        followup_model=os.environ.get("FOLLOWUP_MODEL", model),
        active_listening_model=os.environ.get("ACTIVE_LISTENING_MODEL", model),
        summariser_model=os.environ.get("SUMMARISER_MODEL", model),
        summariser_base_url=os.environ.get("SUMMARISER_BASE_URL") or None,
        temperature=temperature,
    )

    logger.info("Starting simulation: interviewer=%s | agent=%s | protocol=%s",
                interviewer_id, agent_id, protocol.protocol_name)

    if interviewer_id == "agentic":
        transcript = _run_agentic(protocol, cfg, agent, minutes_per_turn)
    elif interviewer_id == "scripted":
        from eval.baselines.scripted import run_scripted
        transcript = run_scripted(protocol, agent, minutes_per_turn)
    elif interviewer_id == "single_llm":
        from eval.baselines.single_llm import run_single_llm
        transcript = run_single_llm(protocol, agent, minutes_per_turn, model, temperature)
    else:
        raise ValueError(f"Unknown interviewer: {interviewer_id!r}. Choose: agentic, scripted, single_llm")

    result = {
        "interviewer_id": interviewer_id,
        "agent_id": agent_id,
        "protocol_name": protocol.protocol_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "minutes_per_turn": minutes_per_turn,
        "turns": sum(1 for t in transcript if t["role"] == "interviewee"),
        "transcript": transcript,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{interviewer_id}_{agent_id}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Transcript saved → %s", out_path)
    return out_path


def _run_agentic(protocol, cfg, agent, minutes_per_turn: float) -> list[dict]:
    """
    Run the multi-agent interviewer. Uses a mutable list as a state reference
    so the get_response callback can access the live transcript.
    """
    # We need the InterviewState reference inside get_response, but run_interview_headless
    # creates state internally. We use a closure over a mutable container.
    state_box: list = []

    # Monkey-patch: wrap run_interview_headless to capture the state object.
    # Simpler approach: replicate the state init here so we can build the transcript context.
    # The cleanest solution: keep a running copy of the transcript in the closure.
    running_transcript: list[dict] = []

    def get_response(interviewer_turn: str) -> str:
        ctx = running_transcript + [{"role": "interviewer", "content": interviewer_turn}]
        answer = respond(agent, ctx)
        # After get_response returns, the headless runner appends both turns to its internal
        # state.transcript. Mirror that here so subsequent calls see the full history.
        running_transcript.append({"role": "interviewer", "content": interviewer_turn})
        running_transcript.append({"role": "interviewee", "content": answer})
        return answer

    state = run_interview_headless(
        protocol=protocol,
        cfg=cfg,
        get_response=get_response,
        minutes_per_turn=minutes_per_turn,
    )
    return state.transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single simulated interview")
    parser.add_argument("--protocol", required=True, help="Path to protocol JSON")
    parser.add_argument("--agent", required=True, help="Path to genagents agent folder")
    parser.add_argument("--interviewer", default="agentic",
                        choices=["agentic", "scripted", "single_llm"],
                        help="Which interviewer to use (default: agentic)")
    parser.add_argument("--minutes-per-turn", type=float, default=2.0,
                        help="Simulated minutes credited per interviewee turn (default: 2.0)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model for the interviewer agents (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    out = run_simulation(
        protocol_path=Path(args.protocol),
        agent_folder=Path(args.agent),
        interviewer_id=args.interviewer,
        minutes_per_turn=args.minutes_per_turn,
        model=args.model,
        temperature=args.temperature,
    )
    print(f"\nTranscript saved to: {out}")


if __name__ == "__main__":
    main()
