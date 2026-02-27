from __future__ import annotations

import logging
import time
from concurrent.futures import Future
from typing import Optional

from ai_interviewer.agents.active_listening import generate_turn
from ai_interviewer.agents.orchestrator import decide_action
from ai_interviewer.agents.socratic import generate_probe
from ai_interviewer.agents.topic_evaluator import evaluate_momentum_async
from ai_interviewer.llm import LLMConfig
from ai_interviewer.protocol import Protocol
from ai_interviewer.state import InterviewState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
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
# Main Interview loop
# ---------------------------------------------------------------------------

def run_interview(protocol: Protocol, cfg: LLMConfig) -> None:
    state = _init_state(protocol)
    topics = protocol.topics
    momentum_future: Optional[Future] = None

    print(f"\n{'=' * 60}")
    print(f"  {protocol.protocol_name}")
    print(f"  {protocol.total_minutes} min · {len(topics)} topics")
    print(f"{'=' * 60}\n")

    # First turn: guiding question of topic 0, no active-listening wrap (no prior response yet)
    interviewer_turn = generate_turn(
        topics[0].guiding_question, state, topics, cfg, is_first_turn=True
    )

    turn_start = time.time()

    while True:
        # ── Display interviewer turn ─────────────────────────────────────────
        print(f"\nINTERVIEWER: {interviewer_turn}\n")

        # ── Get interviewee response ─────────────────────────────────────────
        try:
            response = input("YOU: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Interview ended by user]")
            break

        if not response:
            continue

        # ── Update timing ────────────────────────────────────────────────────
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

        # ── Update transcript ────────────────────────────────────────────────
        state.transcript.append({"role": "interviewer", "content": interviewer_turn})
        state.transcript.append({"role": "interviewee", "content": response})

        # ── Consume previous momentum signal ─────────────────────────────────
        prev_momentum = state.topic_momentum
        _consume_momentum(momentum_future, state)
        if momentum_future is not None:
            logger.info("[TopicEvaluator] consumed — continue_probing=%s", state.topic_momentum)

        # ── Orchestrator: select action ───────────────────────────────────────
        action = decide_action(state, topics)
        logger.info("[Orchestrator] action=%s", action)

        # ── WRAP_UP: deliver closing and exit ────────────────────────────────
        if action == "WRAP_UP":
            core = _get_turn_text("WRAP_UP", state, protocol)
            interviewer_turn = generate_turn(core, state, topics, cfg)
            print(f"\nINTERVIEWER: {interviewer_turn}\n")
            break

        # ── Kick off async Topic Evaluator for next turn ─────────────────────
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
