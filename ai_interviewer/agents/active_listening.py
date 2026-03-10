from __future__ import annotations

import re

from ai_interviewer.llm import LLMConfig, _invoke_json
from ai_interviewer.prompts import ACTIVE_LISTENING_SYSTEM, build_context
from ai_interviewer.state import InterviewState


def _strip_trailing_questions(text: str) -> str:
    """Remove any sentences ending with '?' from the end of the prefix.

    Guards against the LLM ignoring the 'never ask a question in the prefix'
    rule, which would otherwise cause the core question to appear twice.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    clean = [s for s in sentences if not s.strip().endswith("?")]
    return " ".join(clean).strip()


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
        f"Decide whether an acknowledgement prefix is needed before the following question, then write one only if it is.\n\nNext question:\n{core_content}\n\n{context}",
    )
    if not result.get("use_prefix", False):
        return core_content
    prefix = _strip_trailing_questions(result.get("prefix", "").strip())
    return f"{prefix} {core_content}" if prefix else core_content
