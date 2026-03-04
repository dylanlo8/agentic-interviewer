from __future__ import annotations

"""
Scripted baseline interviewer.

Asks each topic's guiding question in order with no probing or active listening.
Immediately transitions after each interviewee response.
This is the simplest possible baseline — a pure protocol reader.
"""

import logging

from ai_interviewer.protocol import Protocol
from eval.interviewee import respond  # also bootstraps the genagents sys.path

logger = logging.getLogger(__name__)


def run_scripted(
    protocol: Protocol,
    agent: GenerativeAgent,
    minutes_per_turn: float = 2.0,
) -> list[dict]:
    """
    Run the scripted baseline against a genagents agent.
    Returns the full transcript as a list of {"role", "content"} dicts.
    """
    transcript: list[dict] = []
    topics = protocol.topics
    elapsed = 0.0
    max_time = protocol.total_minutes - protocol.wrapup_minutes

    for topic in topics:
        if elapsed >= max_time:
            break

        question = topic.guiding_question
        logger.info("[Scripted] topic='%s' | question=%s", topic.topic_title, question[:60])

        ctx = transcript + [{"role": "interviewer", "content": question}]
        response = respond(agent, ctx)

        transcript.append({"role": "interviewer", "content": question})
        transcript.append({"role": "interviewee", "content": response})

        elapsed += minutes_per_turn

    # Wrap-up
    closing = "Thank you so much for sharing all of this with me. Is there anything else you'd like to add before we close?"
    ctx = transcript + [{"role": "interviewer", "content": closing}]
    final_response = respond(agent, ctx)
    transcript.append({"role": "interviewer", "content": closing})
    transcript.append({"role": "interviewee", "content": final_response})
    transcript.append({"role": "interviewer", "content": "Thank you for your time."})

    return transcript
