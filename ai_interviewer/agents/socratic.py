from __future__ import annotations

from ai_interviewer.llm import LLMConfig, _invoke_json
from ai_interviewer.prompts import SOCRATIC_SYSTEM, build_context
from ai_interviewer.state import InterviewState


def generate_probe(state: InterviewState, topics: list, cfg: LLMConfig) -> dict:
    """
    Generate a follow-up probe for the interviewee's latest response.

    Returns a dict with keys:
      - probe_question (str): the single follow-up question to ask
      - open_loops (list[str]): threads raised but not finished
      - rationale (str): why this question was chosen
    """
    context = build_context(state, topics)
    result = _invoke_json(
        cfg.socratic_model,
        cfg.temperature,
        SOCRATIC_SYSTEM,
        f"Generate a follow-up probe.\n\n{context}",
    )
    # Ensure expected keys exist
    result.setdefault("probe_question", "Can you tell me more about that?")
    result.setdefault("open_loops", [])
    result.setdefault("rationale", "")
    return result
