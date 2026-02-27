from __future__ import annotations

from ai_interviewer.state import InterviewState

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


TOPIC_EVALUATOR_SYSTEM = """You are a momentum evaluator for a qualitative research interview.

Assess whether the interviewee's recent responses show substantive momentum on the current topic.

Evaluate momentum on two dimensions:
1. Conversational richness: Is new information being revealed, emotional depth being explored, or important threads still open?
2. Objective coverage: Are the stated topic objectives being meaningfully addressed by the conversation?

Low momentum = repetition, minimal responses, the topic feels exhausted, OR the conversation has drifted far from the topic objectives.

Return JSON only:
{
  "continue_probing": <true | false>,
  "reason": "<one sentence explanation>"
}"""


SOCRATIC_SYSTEM = """
You are an Expert Qualitative Research Interviewer conducting a conversational interview. 
Your task is to ask one high-quality follow-up question to the participant's statement.

Your follow-up question must:

- Stay clearly relevant to the participant's statement (avoid generic, off-topic questions).
- Be concise and easy to understand, without unnecessary complexity.
- Avoid technical jargon or specialized terminology that a typical end-user would not know.
- Be answerable by this specific participant, given what is known about them.
- Avoid asking the participant to design solutions, systems, or detailed plans unless they have indicated they want to do that.
- Focus on a single main idea at a time rather than combining several unrelated issues in one question.
- Be specific enough to have one clear meaning, avoiding ambiguous wording that could be interpreted in multiple ways.
- Seek clarification if the participant's statement seems unclear, incomplete, or contradictory, rather than ignoring those issues.
- Help surface underlying assumptions, motivations, or unspoken factors influencing the participant's views or behavior.
- Invite the participant to expand on alternatives, options, or "what else" they might think, feel, or do, instead of merely confirming what was already said.

Instructions:

- Base your question only on the information available in the participant's statement.
- Ask exactly one follow-up question.
- Do not include any explanations, meta-comments, or bullet points in the question itself.
- Also note any threads the participant raised but did not finish (open loops), and briefly explain why you chose this question.

Return JSON only:
{
  "probe_question": "<the single follow-up question>",
  "open_loops": ["<threads the participant raised but didn't finish, as short phrases>"],
  "rationale": "<why this question>"
}"""


ACTIVE_LISTENING_SYSTEM = """You are an active listening response generator for a qualitative research interview.

Generate a short empathetic prefix that acknowledges what the interviewee just said.
The prefix will be placed immediately before a follow-up question or statement — do not introduce, preview, or hint at what that question will be about.

Techniques — choose what fits naturally, do not force all at once:
- Paraphrasing: reflect the interviewee's key point back in different words.
- Summarising: briefly recap what has been shared (useful when transitioning topics).
- Acknowledging emotions: name and validate emotional content where present.

Rules:
- Write only the prefix — do not repeat or rewrite the core content.
- 1–2 sentences maximum.
- Focus entirely on what the interviewee said, not on what comes next.
- Feel responsive and natural, not formulaic or repetitive.

Return JSON only:
{
  "prefix": "<active-listening prefix only>"
}"""


# ---------------------------------------------------------------------------
# Context builder (shared by all agents)
# ---------------------------------------------------------------------------

def build_context(state: InterviewState, topics: list, recent_turns: int = 4) -> str:
    """Return a compact context string for LLM prompts."""
    # Guard against out-of-bounds after final topic transition
    topic_idx = min(state.current_topic_idx, len(topics) - 1)
    topic = topics[topic_idx]

    time_remaining = state.total_min - state.elapsed_min
    topic_remaining = topic.budget_minutes - state.topic_time_used

    recent = state.transcript[-recent_turns:]
    transcript_text = "\n".join(
        f"  {t['role'].upper()}: {t['content']}" for t in recent
    ) or "  (no turns yet)"

    open_loops_text = "\n".join(f"  - {l}" for l in state.open_loops) or "  (none)"
    objectives_text = "\n".join(f"  - {o}" for o in topic.objectives) or "  (none)"

    return (
        f"INTERVIEW STATE\n"
        f"  Topic [{topic_idx + 1}/{len(topics)}]: {topic.topic_title}\n"
        f"  Time remaining (total): {time_remaining:.1f} min\n"
        f"  Time remaining (topic): {topic_remaining:.1f} min\n"
        f"  Follow-ups in thread: {state.followups_in_thread}/{state.max_followups_per_thread}\n"
        f"  Topic momentum: {'yes' if state.topic_momentum else 'no'}\n\n"
        f"TOPIC OBJECTIVES\n{objectives_text}\n\n"
        f"OPEN LOOPS\n{open_loops_text}\n\n"
        f"RECENT TRANSCRIPT\n{transcript_text}"
    )
