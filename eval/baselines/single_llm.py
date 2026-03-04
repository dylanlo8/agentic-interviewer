from __future__ import annotations

"""
Single-LLM baseline interviewer.

One GPT-4o-mini call per turn decides the next interviewer utterance.
No multi-agent architecture, no hard-rule gates, no separate active listening pass.
This baseline tests whether a simple prompted LLM can match the multi-agent system.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ai_interviewer.protocol import Protocol
from eval.interviewee import respond  # also bootstraps the genagents sys.path

logger = logging.getLogger(__name__)

SINGLE_LLM_SYSTEM = """You are a qualitative research interviewer conducting a structured interview.

You will receive the interview protocol (topics and objectives) and the conversation so far.
Your task is to generate the next interviewer turn — one concise utterance that:
- Acknowledges what the participant just said (brief active listening prefix)
- Either asks a probing follow-up OR transitions to the next topic when coverage is sufficient
- Never asks multiple questions at once
- Uses open-ended, non-leading language

When most topic objectives are covered or the conversation is stalling, move to the next topic.
When all topics are done, close the interview warmly.

Return ONLY the interviewer's next utterance as plain text — no JSON, no labels."""


def run_single_llm(
    protocol: Protocol,
    agent: GenerativeAgent,
    minutes_per_turn: float = 2.0,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> list[dict]:
    """
    Run the single-LLM baseline against a genagents agent.
    Returns the full transcript as a list of {"role", "content"} dicts.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    transcript: list[dict] = []
    elapsed = 0.0
    max_time = protocol.total_minutes - protocol.wrapup_minutes

    # First turn: opening question
    opening = protocol.topics[0].guiding_question
    logger.info("[SingleLLM] opening topic='%s'", protocol.topics[0].topic_title)

    while True:
        # Build context for the LLM
        user_message = _build_user_message(protocol, transcript, elapsed, max_time)

        if not transcript:
            # First turn — use opening question directly
            interviewer_turn = opening
        else:
            messages = [
                SystemMessage(content=SINGLE_LLM_SYSTEM),
                HumanMessage(content=user_message),
            ]
            interviewer_turn = llm.invoke(messages).content.strip()

        logger.info("[SingleLLM] elapsed=%.1f | turn=%s...", elapsed, interviewer_turn[:60])

        # Get agent response
        ctx = transcript + [{"role": "interviewer", "content": interviewer_turn}]
        response = respond(agent, ctx)

        transcript.append({"role": "interviewer", "content": interviewer_turn})
        transcript.append({"role": "interviewee", "content": response})

        elapsed += minutes_per_turn

        # Stop when time is up
        if elapsed >= protocol.total_minutes:
            break

    # Final wrap-up turn (no agent response needed after this)
    closing_ctx = _build_user_message(protocol, transcript, elapsed, max_time, closing=True)
    messages = [
        SystemMessage(content=SINGLE_LLM_SYSTEM),
        HumanMessage(content=closing_ctx),
    ]
    closing_turn = llm.invoke(messages).content.strip()
    transcript.append({"role": "interviewer", "content": closing_turn})

    return transcript


def _build_user_message(
    protocol: Protocol,
    transcript: list[dict],
    elapsed: float,
    max_time: float,
    closing: bool = False,
) -> str:
    protocol_summary = _format_protocol(protocol)
    transcript_text = _format_transcript(transcript) if transcript else "(no turns yet)"
    time_remaining = max(0.0, protocol.total_minutes - elapsed)

    instruction = (
        "The interview time is up. Generate a warm closing statement."
        if closing
        else f"Time remaining: {time_remaining:.1f} min. Generate the next interviewer turn."
    )

    return (
        f"PROTOCOL:\n{protocol_summary}\n\n"
        f"TRANSCRIPT SO FAR:\n{transcript_text}\n\n"
        f"{instruction}"
    )


def _format_protocol(protocol: Protocol) -> str:
    lines = [f"{protocol.protocol_name}"]
    for i, topic in enumerate(protocol.topics, 1):
        lines.append(f"\nTopic {i}: {topic.topic_title} (budget: {topic.budget_minutes} min)")
        lines.append(f"  Guiding question: {topic.guiding_question}")
        if topic.objectives:
            for obj in topic.objectives:
                lines.append(f"  - {obj}")
    return "\n".join(lines)


def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for turn in transcript[-8:]:  # Last 8 turns for context efficiency
        role = "INTERVIEWER" if turn["role"] == "interviewer" else "PARTICIPANT"
        lines.append(f"{role}: {turn['content']}")
    return "\n\n".join(lines)
