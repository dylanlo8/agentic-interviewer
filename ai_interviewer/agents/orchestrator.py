from __future__ import annotations

from ai_interviewer.state import InterviewState


def decide_action(state: InterviewState, topics: list) -> str:
    """Deterministic router — no LLM. Priority order matters."""

    # 1. Total time up — only force wrap-up if not already on the last topic,
    #    since the last topic is typically a closing/reflection topic that should
    #    run to its own budget rather than be cut off by the wrapup guard.
    on_last_topic = state.current_topic_idx >= len(topics) - 1
    if state.elapsed_min >= state.total_min - state.wrapup_min and not on_last_topic:
        return "WRAP_UP"

    # 2. All topics exhausted
    if state.current_topic_idx >= len(topics):
        return "WRAP_UP"

    topic = topics[state.current_topic_idx]

    # 3. Topic budget exhausted
    if state.topic_time_used >= topic.budget_minutes:
        return "TRANSITION_TOPIC"

    # 4. Too many consecutive follow-ups
    if state.followups_in_thread >= state.max_followups_per_thread:
        return "TRANSITION_TOPIC"

    # 5. Topic Evaluator signals low momentum (only meaningful after at least one probe)
    if not state.topic_momentum and state.followups_in_thread > 0:
        return "TRANSITION_TOPIC"

    # 6. Default: probe
    return "PROBE"
