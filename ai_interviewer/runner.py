from __future__ import annotations

import logging
import textwrap
import time
from concurrent.futures import Future
from typing import Callable, Optional

from ai_interviewer.agents.active_listening import generate_turn
from ai_interviewer.agents.orchestrator import decide_action
from ai_interviewer.agents.followup import generate_probe
from ai_interviewer.agents.topic_evaluator import evaluate_momentum_async
from ai_interviewer.llm import LLMConfig
from ai_interviewer.protocol import Protocol
from ai_interviewer.state import InterviewState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared by both live and headless runners)
# ---------------------------------------------------------------------------

def _init_state(protocol: Protocol) -> InterviewState:
    return InterviewState(
        total_min=protocol.total_minutes,
        wrapup_min=protocol.wrapup_minutes,
        max_followups_per_thread=protocol.max_followups_per_thread,
    )


def _get_turn_text(action: str, state: InterviewState, protocol: Protocol) -> str:
    """Return the raw question/statement for the given action (before active listening wrap)."""
    topics = protocol.topics

    if action == "TRANSITION_TOPIC":
        next_idx = state.current_topic_idx + 1
        if next_idx >= len(topics):
            return "We've covered everything I wanted to explore with you today."
        return topics[next_idx].guiding_question

    if action == "WRAP_UP":
        return "Thank you so much for sharing all of this with me. Is there anything else you'd like to add before we close?"

    return ""  # PROBE — caller supplies probe_question directly


def _apply_action(action: str, state: InterviewState, protocol: Protocol, probe_result: Optional[dict]) -> None:
    """Mutate state to reflect the action just taken."""
    topics = protocol.topics

    if action == "PROBE":
        state.followups_in_thread += 1
        if probe_result:
            state.open_loops.extend(probe_result.get("open_loops", []))

    elif action == "TRANSITION_TOPIC":
        prev_topic = topics[min(state.current_topic_idx, len(topics) - 1)]
        state.current_topic_idx += 1
        state.topic_time_used = 0.0
        state.followups_in_thread = 0
        state.open_loops = []
        if state.current_topic_idx < len(topics):
            next_topic = topics[state.current_topic_idx]
            logger.info(
                "[TopicSwitch] '%s' → '%s'",
                prev_topic.topic_title,
                next_topic.topic_title,
            )
        else:
            logger.info("[TopicSwitch] '%s' → (all topics complete)", prev_topic.topic_title)


def _consume_momentum(future: Optional[Future], state: InterviewState) -> None:
    """Read the async Topic Evaluator result if available; default to True on timeout/error."""
    if future is None:
        return
    try:
        state.topic_momentum = future.result(timeout=1.0)
    except Exception:
        state.topic_momentum = True  # Safe default: keep probing


# ---------------------------------------------------------------------------
# Headless runner — no terminal I/O; used by the evaluation pipeline
# ---------------------------------------------------------------------------

def run_interview_headless(
    protocol: Protocol,
    cfg: LLMConfig,
    get_response: Callable[[str], str],
    minutes_per_turn: float = 2.0,
) -> InterviewState:
    """
    Run a complete interview without any terminal I/O.

    get_response(interviewer_turn: str) -> str
        Called each turn to obtain the interviewee's response.
        In simulation mode this calls a genagents agent or baseline LLM.

    minutes_per_turn
        Simulated time credited to elapsed_min and topic_time_used per turn.
        Use a value that reflects realistic conversation pace (default: 2.0 min).

    Returns the final InterviewState (with full transcript).
    """
    state = _init_state(protocol)
    topics = protocol.topics
    momentum_future: Optional[Future] = None

    interviewer_turn = generate_turn(
        topics[0].guiding_question, state, topics, cfg, is_first_turn=True
    )

    while True:
        response = get_response(interviewer_turn)
        if not response:
            continue

        # ── Update timing ─────────────────────────────────────────────────────
        state.elapsed_min += minutes_per_turn
        state.topic_time_used += minutes_per_turn

        topic = topics[min(state.current_topic_idx, len(topics) - 1)]
        logger.info(
            "[Turn] elapsed=%.1f min | topic='%s' (%.1f/%.1f min used)",
            state.elapsed_min,
            topic.topic_title,
            state.topic_time_used,
            topic.budget_minutes,
        )

        # ── Update transcript ─────────────────────────────────────────────────
        state.transcript.append({"role": "interviewer", "content": interviewer_turn})
        state.transcript.append({"role": "interviewee", "content": response})

        # ── Consume previous momentum signal ──────────────────────────────────
        _consume_momentum(momentum_future, state)
        if momentum_future is not None:
            logger.info("[TopicEvaluator] consumed — continue_probing=%s", state.topic_momentum)

        # ── Orchestrator: select action ───────────────────────────────────────
        action = decide_action(state, topics)
        logger.info("[Orchestrator] action=%s", action)

        # ── WRAP_UP: record closing turn and exit ─────────────────────────────
        if action == "WRAP_UP":
            core = _get_turn_text("WRAP_UP", state, protocol)
            interviewer_turn = generate_turn(core, state, topics, cfg)
            state.transcript.append({"role": "interviewer", "content": interviewer_turn})
            break

        # ── Kick off async Topic Evaluator for next turn ──────────────────────
        momentum_future = evaluate_momentum_async(state, topics, cfg)

        # ── Generate core content ─────────────────────────────────────────────
        probe_result: Optional[dict] = None

        if action == "PROBE":
            probe_result = generate_probe(state, topics, cfg)
            core = probe_result["probe_question"]
        else:
            core = _get_turn_text(action, state, protocol)

        # ── Apply state changes ───────────────────────────────────────────────
        _apply_action(action, state, protocol, probe_result)

        # ── Check if all topics done after transition ─────────────────────────
        if state.current_topic_idx >= len(topics):
            core = _get_turn_text("WRAP_UP", state, protocol)

        # ── Active Listening wraps the final turn ─────────────────────────────
        interviewer_turn = generate_turn(core, state, topics, cfg)

    return state


# ---------------------------------------------------------------------------
# Live runner — terminal I/O with wall-clock timing
# ---------------------------------------------------------------------------

def run_interview(protocol: Protocol, cfg: LLMConfig) -> None:
    state = _init_state(protocol)
    topics = protocol.topics
    momentum_future: Optional[Future] = None

    W = 60
    print(f"\n{'=' * W}")
    print(f"  {protocol.protocol_name}")
    print(f"{'=' * W}")

    if protocol.description:
        for line in textwrap.wrap(protocol.description, width=W - 2):
            print(f"  {line}")
        print()

    topic_titles = "\n".join(f"  {i + 1}. {t.topic_title}" for i, t in enumerate(topics))
    print(f"  This conversation will cover {len(topics)} topics:")
    print(topic_titles)
    print(f"\n  Expected duration: ~{int(protocol.total_minutes)} minutes")
    print(f"{'=' * W}\n")

    interviewer_turn = generate_turn(
        topics[0].guiding_question, state, topics, cfg, is_first_turn=True
    )

    turn_start = time.time()

    while True:
        print(f"\nINTERVIEWER: {interviewer_turn}\n")

        try:
            response = input("YOU: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Interview ended by user]")
            break

        if not response:
            continue

        elapsed_turn = (time.time() - turn_start) / 60.0
        state.elapsed_min += elapsed_turn
        state.topic_time_used += elapsed_turn
        turn_start = time.time()

        topic = topics[min(state.current_topic_idx, len(topics) - 1)]
        logger.info(
            "[Turn] elapsed=%.1f min | turn=%.1f min | topic='%s' (%.1f/%.1f min used)",
            state.elapsed_min,
            elapsed_turn,
            topic.topic_title,
            state.topic_time_used,
            topic.budget_minutes,
        )

        state.transcript.append({"role": "interviewer", "content": interviewer_turn})
        state.transcript.append({"role": "interviewee", "content": response})

        _consume_momentum(momentum_future, state)
        if momentum_future is not None:
            logger.info("[TopicEvaluator] consumed — continue_probing=%s", state.topic_momentum)

        action = decide_action(state, topics)
        logger.info("[Orchestrator] action=%s", action)

        if action == "WRAP_UP":
            core = _get_turn_text("WRAP_UP", state, protocol)
            interviewer_turn = generate_turn(core, state, topics, cfg)
            print(f"\nINTERVIEWER: {interviewer_turn}\n")
            break

        momentum_future = evaluate_momentum_async(state, topics, cfg)

        probe_result: Optional[dict] = None

        if action == "PROBE":
            probe_result = generate_probe(state, topics, cfg)
            core = probe_result["probe_question"]
        else:
            core = _get_turn_text(action, state, protocol)

        _apply_action(action, state, protocol, probe_result)

        if state.current_topic_idx >= len(topics):
            core = _get_turn_text("WRAP_UP", state, protocol)

        interviewer_turn = generate_turn(core, state, topics, cfg)

    _print_summary(state)


def _print_summary(state: InterviewState) -> None:
    interviewee_turns = sum(1 for t in state.transcript if t["role"] == "interviewee")
    print(f"\n{'=' * 60}")
    print(f"  Session complete")
    print(f"  Duration : {state.elapsed_min:.1f} min")
    print(f"  Turns    : {interviewee_turns}")
    if state.open_loops:
        print(f"  Open loops: {', '.join(state.open_loops)}")
    print(f"{'=' * 60}\n")
