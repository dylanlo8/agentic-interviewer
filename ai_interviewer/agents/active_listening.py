from __future__ import annotations

from ai_interviewer.llm import LLMConfig, _invoke_json
from ai_interviewer.prompts import ACTIVE_LISTENING_SYSTEM, build_context
from ai_interviewer.state import InterviewState


def generate_turn(
    core_content: str,
    state: InterviewState,
    topics: list,
    cfg: LLMConfig,
    is_first_turn: bool = False,
) -> str:
    """
    Wrap core_content with an active-listening prefix and return the final interviewer turn.

    On the very first turn there is no prior interviewee response to reflect on,
    so core_content is returned as-is.
    """
    if is_first_turn:
        return core_content

    context = build_context(state, topics)
    result = _invoke_json(
        cfg.active_listening_model,
        cfg.temperature,
        ACTIVE_LISTENING_SYSTEM,
        f"Construct the complete interviewer turn.\n\nCore content:\n{core_content}\n\n{context}",
    )
    return result.get("interviewer_turn", core_content)
