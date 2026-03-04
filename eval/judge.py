from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""
Transcript evaluator — scores a saved transcript JSON against the protocol.

Usage:
    python eval/judge.py eval/results/<transcript>.json \\
        --protocol notebooks/protocol.json \\
        --model gpt-4o

Scores each transcript on five dimensions (1–5) and writes a scores JSON
file alongside the transcript in eval/results/.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from ai_interviewer.llm import _invoke_json
from ai_interviewer.protocol import Protocol, load_protocol

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge system prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an expert evaluator of qualitative research interviews.

You will be given a complete interview transcript and the protocol used to conduct it,
including the topic objectives that the interviewer was trying to address.

Score the interview on each of the five dimensions below. For each dimension give:
  - "score": an integer from 1 to 5
  - "rationale": one to two sentences explaining the score

Scoring scale:
  1 = Very poor   2 = Poor   3 = Acceptable   4 = Good   5 = Excellent

Dimensions:

1. topic_coverage
   Did the conversation meaningfully address the stated objectives for each topic?
   Consider whether all topics were reached and whether each objective was explored.

2. response_depth
   Did the interviewee provide rich, specific, concrete answers?
   A score of 5 means detailed personal examples and emotional nuance were present.
   A score of 1 means responses were brief, vague, or evasive throughout.

3. question_quality
   Were the interviewer's questions open-ended, focused on one idea at a time,
   non-leading, and designed to elicit deeper reflection?
   Penalise closed yes/no questions, leading questions, or multi-part questions.

4. active_listening
   Did the interviewer demonstrate active listening — paraphrasing, reflecting
   emotions, acknowledging what was said — before asking the next question?
   Penalise formulaic affirmations ("That's interesting", "Thank you for sharing").

5. pacing
   Was time distributed well across topics? Were topics rushed or padded?
   A score of 5 means each topic received appropriate depth relative to its importance.
   A score of 1 means the interview was clearly imbalanced (one topic dominating, others skipped).

Return ONLY valid JSON in this exact format:
{
  "scores": {
    "topic_coverage":   {"score": <1-5>, "rationale": "<string>"},
    "response_depth":   {"score": <1-5>, "rationale": "<string>"},
    "question_quality": {"score": <1-5>, "rationale": "<string>"},
    "active_listening": {"score": <1-5>, "rationale": "<string>"},
    "pacing":           {"score": <1-5>, "rationale": "<string>"}
  },
  "summary": "<one paragraph narrative assessment of the overall interview quality>"
}"""


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------

def judge_transcript(
    transcript_path: Path,
    protocol: Protocol,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> dict:
    """
    Score a saved transcript JSON against the protocol.

    Returns the scores dict (also written to disk as a sibling _scores.json file).
    """
    transcript_data = json.loads(transcript_path.read_text())
    transcript = transcript_data["transcript"]
    interviewer_id = transcript_data.get("interviewer_id", "unknown")
    agent_id = transcript_data.get("agent_id", "unknown")

    # Build the user message: protocol overview + full transcript
    protocol_text = _format_protocol(protocol)
    transcript_text = _format_transcript(transcript)

    user_message = (
        f"PROTOCOL:\n{protocol_text}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        f"Please score this interview."
    )

    logger.info("Judging transcript: interviewer=%s | agent=%s | turns=%d",
                interviewer_id, agent_id, transcript_data.get("turns", "?"))

    raw = _invoke_json(model, temperature, JUDGE_SYSTEM, user_message)

    # Compute overall mean score
    scores = raw.get("scores", {})
    score_values = [v["score"] for v in scores.values() if isinstance(v, dict) and "score" in v]
    overall = round(sum(score_values) / len(score_values), 2) if score_values else None

    result = {
        "interviewer_id": interviewer_id,
        "agent_id": agent_id,
        "protocol_name": protocol.protocol_name,
        "transcript_file": transcript_path.name,
        "scores": scores,
        "overall": overall,
        "summary": raw.get("summary", ""),
    }

    # Write scores file alongside the transcript
    scores_path = transcript_path.with_name(transcript_path.stem + "_scores.json")
    scores_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Scores saved → %s  (overall=%.2f)", scores_path, overall or 0)

    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_protocol(protocol: Protocol) -> str:
    lines = [f"Protocol: {protocol.protocol_name} (v{protocol.version})"]
    for i, topic in enumerate(protocol.topics, 1):
        lines.append(f"\nTopic {i}: {topic.topic_title} (budget: {topic.budget_minutes} min)")
        lines.append(f"  Guiding question: {topic.guiding_question}")
        if topic.objectives:
            lines.append("  Objectives:")
            for obj in topic.objectives:
                lines.append(f"    - {obj}")
    return "\n".join(lines)


def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for turn in transcript:
        role = "INTERVIEWER" if turn["role"] == "interviewer" else "PARTICIPANT"
        lines.append(f"{role}: {turn['content']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Score a simulated interview transcript")
    parser.add_argument("transcript", help="Path to transcript JSON file")
    parser.add_argument("--protocol", required=True, help="Path to protocol JSON")
    parser.add_argument("--model", default="gpt-4o",
                        help="Judge LLM model (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    protocol = load_protocol(Path(args.protocol))
    result = judge_transcript(
        transcript_path=Path(args.transcript),
        protocol=protocol,
        model=args.model,
        temperature=args.temperature,
    )

    print(f"\nOverall score: {result['overall']}/5.0")
    print(f"\nSummary: {result['summary']}\n")
    for dim, val in result["scores"].items():
        print(f"  {dim:<20} {val['score']}/5  — {val['rationale']}")


if __name__ == "__main__":
    main()
