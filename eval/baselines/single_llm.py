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

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are a professional qualitative research interviewer, trained in evidence-based \
interviewing techniques. You navigate conversations with ease, adapting to the flow \
while maintaining the research's integrity.

CORE GUIDELINES:
- Ask exactly ONE question per turn — never overload the participant with multiple questions.
- Ask open-ended, non-leading questions. Do not suggest answers, associations, or ideas \
for how participants could respond.
- If the participant gives a short or vague answer, ask a follow-up to understand further: \
e.g. "Why is that?", "Could you expand on that?", "Can you give me an example?"
- If a participant gives a surprising or unclear answer, ask a neutral clarifying question \
— do not judge or evaluate their answer.
- Do not take a position on whether answers are right or wrong.
- From time to time, briefly restate in one or two sentences what was just said \
(using the participant's own words), then check whether you understood correctly \
before moving on.
- Maintain a casual, conversational tone — pleasant, neutral, and professional. \
Neither overly cold nor overly familiar.
- Assure the participant that their data is handled with care and privacy is respected \
if they seem hesitant about sensitive topics. Avoid direct questions on intimate topics.
- When it makes sense, connect your question to something the participant just said.
- Move to the next topic only when its objectives are sufficiently covered or the \
conversation is stalling. Do not rush through topics.
- When all topics are done, close the interview warmly.

INTERVIEW OUTLINE:
{outline}

Return ONLY the interviewer's next utterance as plain text — no JSON, no labels.\
"""

USER_TEMPLATE = """\
TRANSCRIPT:
{transcript}

{instruction}\
"""

INSTRUCTION_CONTINUE = "Time remaining: {time_remaining:.1f} min. Generate the next interviewer turn."
INSTRUCTION_CLOSING = "The interview time is up. Generate a warm closing statement."


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_single_llm(
    protocol: Protocol,
    agent,
    minutes_per_turn: float = 2.0,
    model: str = "gpt-5-mini",
    temperature: float = 0.2,
) -> list[dict]:
    """
    Run the single-LLM baseline against a genagents agent.
    Returns the full transcript as a list of {"role", "content"} dicts.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    transcript: list[dict] = []
    elapsed = 0.0

    system_message = SystemMessage(content=SYSTEM_TEMPLATE.format(outline=_format_outline(protocol)))

    opening = protocol.topics[0].guiding_question
    logger.info("[SingleLLM] opening topic='%s'", protocol.topics[0].topic_title)

    while True:
        if not transcript:
            interviewer_turn = opening
        else:
            user_message = HumanMessage(content=_build_user_message(transcript, elapsed, protocol.total_minutes))
            interviewer_turn = llm.invoke([system_message, user_message]).content.strip()

        logger.info("[SingleLLM] elapsed=%.1f | turn=%s...", elapsed, interviewer_turn[:60])

        ctx = transcript + [{"role": "interviewer", "content": interviewer_turn}]
        response = respond(agent, ctx)

        transcript.append({"role": "interviewer", "content": interviewer_turn})
        transcript.append({"role": "interviewee", "content": response})

        elapsed += minutes_per_turn

        if elapsed >= protocol.total_minutes:
            break

    # Final wrap-up turn
    closing_message = HumanMessage(content=_build_user_message(transcript, elapsed, protocol.total_minutes, closing=True))
    closing_turn = llm.invoke([system_message, closing_message]).content.strip()
    transcript.append({"role": "interviewer", "content": closing_turn})

    return transcript


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_message(
    transcript: list[dict],
    elapsed: float,
    total_minutes: float,
    closing: bool = False,
) -> str:
    time_remaining = max(0.0, total_minutes - elapsed)
    instruction = (
        INSTRUCTION_CLOSING
        if closing
        else INSTRUCTION_CONTINUE.format(time_remaining=time_remaining)
    )
    return USER_TEMPLATE.format(transcript=_format_transcript(transcript), instruction=instruction)


def _format_outline(protocol: Protocol) -> str:
    lines = [f"{protocol.protocol_name} ({protocol.total_minutes} min total)"]
    for i, topic in enumerate(protocol.topics, 1):
        lines.append(f"\nTopic {i}: {topic.topic_title} (budget: {topic.budget_minutes} min)")
        lines.append(f"  Guiding question: {topic.guiding_question}")
        if topic.objectives:
            for obj in topic.objectives:
                lines.append(f"  - {obj}")
    return "\n".join(lines)


def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for turn in transcript:
        role = "INTERVIEWER" if turn["role"] == "interviewer" else "PARTICIPANT"
        lines.append(f"{role}: {turn['content']}")
    return "\n\n".join(lines)
