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

Scores each transcript on six discriminating dimensions (1–10) and writes a scores JSON
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

JUDGE_SYSTEM = """You are a rigorous evaluator of qualitative research interviews.

You will be given a complete interview transcript and the protocol used to conduct it.

Your task is NOT to assess conversational fluency or warmth in the abstract.
Your task is to assess the interviewer's RESEARCH EFFECTIVENESS — specifically, how well
the interviewer extracts deep, specific, previously-unknown information from the participant.

Score the interview on each of the six dimensions below. For each dimension give:
  - "score": an integer from 1 to 10
  - "rationale": two to three sentences citing specific evidence — quote exact turns or
    paraphrase specific exchanges. Generic rationale is not acceptable.

IMPORTANT CALIBRATION RULES:
- Scores of 9-10 require near-flawless performance. Reserve them for genuinely exceptional work.
- A score of 5-6 means adequate but with clear gaps. Most interviews land here.
- A score of 7-8 means good with identifiable strengths and minor weaknesses.
- You MUST use the full 1-10 range. Do not cluster around 5-7 out of caution.
- You MUST cite specific exchanges (quote content or reference turn patterns) in every rationale.
- Do NOT reward fluency, warmth, or topic completion alone. These are necessary but not sufficient.

Scoring scale (higher = better for all dimensions):
  1-2 = Very poor   3-4 = Poor   5-6 = Acceptable   7-8 = Good   9-10 = Excellent (rare)

Each dimension provides behavioural anchors at 2, 4, 6, 8, and 10. Use odd numbers when
the interview falls between two anchors.

---

Dimensions:

1. probe_tailoring  [higher is better]
   Are the interviewer's follow-up questions visibly grounded in what the participant
   JUST said -- referencing specific words, names, situations, or contradictions from
   that turn -- or are they generic probes that could apply to any interview?

   A generic probe ("Can you tell me more about that?", "How did that make you feel?")
   is NOT tailored even if it follows a rich answer. A tailored probe quotes, mirrors,
   or directly extends a specific element of the participant's last response.

   Penalise: probes that ignore what was just said; probes that restate the prior question
   in new words; interviewer pivoting to a new topic without acknowledging specific content.
     2 = Probes throughout are interchangeable; nothing references specific participant content.
     4 = Occasional surface-level reference (repeats a word the participant used) but no genuine extension.
     6 = Many probes reference specific content; several are still generic or only loosely connected.
     8 = Most probes clearly respond to specific participant content; rare generic moment.
    10 = Every follow-up is unmistakably anchored in specific prior participant content -- zero generic probes.

2. ambiguity_resolution  [higher is better]
   When the participant gives a vague, metaphorical, or non-specific answer, does the
   interviewer diagnose the vagueness and request concrete behavioural evidence?
   Or does the interviewer accept the vague answer, paraphrase it, and move on?

   "Concrete behavioural evidence" means asking the participant to name a specific
   instance, give a number, describe an action, or identify a real example -- not
   just asking them to elaborate.

   Penalise: accepting vague answers without requesting specificity; paraphrasing vague
   answers as though they were complete; pivoting to a new topic after a non-answer.
     2 = Vague answers consistently accepted and paraphrased; never challenged for specificity.
     4 = Rare attempts to clarify but mostly surface-level ("what do you mean?"); concrete probes absent.
     6 = Vague answers sometimes challenged with specific requests; inconsistent across the interview.
     8 = Most vague answers are followed by a concrete request (instance, number, example); rare lapse.
    10 = Every vague or metaphorical answer triggers a diagnostic follow-up requesting behavioural specificity.

3. insight_density  [higher is better]
   How many distinct, previously-unknown facts, dimensions, or perspectives does the
   interviewer surface per topic -- relative to the number of turns used?

   Count "insight units": a concrete piece of information the participant has not mentioned
   before (a specific person, event, number, reason, emotion, contradiction, or mechanism).
   Compare insight units to the total interviewer turns to get a ratio.

   Penalise: turns that simply confirm or paraphrase already-stated content; closing sequences
   that consume turns without producing new information; repetitive logistics questions.
     2 = Almost no new information per turn; conversation circulates around 2-3 already-stated facts.
     4 = Low yield; most turns confirm or rephrase; new insight appears rarely (< 1 unit per 5 turns).
     6 = Moderate yield; new dimensions surface in roughly half the exchanges.
     8 = High yield; most exchanges (> 2 in 3) surface a genuinely new dimension or specific fact.
    10 = Near-every turn produces a new insight unit; no turns wasted on confirmation or logistics.

4. paraphrase_fidelity  [higher is better]
   When the interviewer paraphrases or summarises the participant's response, does the
   paraphrase faithfully reflect what was said -- or does it introduce interpretive framing,
   emotional labels, or conclusions the participant did not express?

   Penalise: substituting interviewer's interpretation for participant's own words; attributing
   emotions the participant did not name; softening or amplifying stated experiences;
   any paraphrase that would cause a careful listener to say "that's not quite what they said."
     2 = Paraphrases consistently introduce new framing or emotion not present in participant's words.
     4 = Frequent interpretive additions; participant's actual language rarely reflected back accurately.
     6 = Most paraphrases are accurate; occasional interpretive gloss or subtle reframing.
     8 = Paraphrases are nearly always faithful; rare minor addition that does not mislead.
    10 = Every paraphrase is a precise, neutral mirror of the participant's own words and stated emotions.

5. sustained_engagement  [higher is better]
   Does the interviewer maintain substantive research engagement throughout the full
   interview -- or does engagement drop off, with the interviewer pivoting early to
   closing logistics ("Is there anything else?", "We're almost out of time") while
   protocol objectives remain unmet?

   "Closing logistics" = any turn whose primary function is to signal the end of the
   interview, check for omissions generically, or wrap up, rather than probe a specific topic.
   Count how many turns in the final 30% of the interview are logistical vs. substantive.

   Penalise: premature pivot to wrap-up language before protocol objectives are met;
   repeating "Is there anything else?" or equivalent more than once; logistical turns that
   crowd out remaining protocol topics.
     2 = Interviewer signals wrap-up or repeats closing questions before half the interview is done.
     4 = Closing pivot happens noticeably early; protocol topics remain unaddressed; "anything else?" repeated 3+ times.
     6 = Closing begins appropriately but one or two premature closures or repeated closing questions occur.
     8 = Sustained substantive engagement through most of the interview; closing is brief and purposeful.
    10 = Full substantive engagement until all protocol objectives met; closing is a single, clean sequence.

6. adaptive_sequencing  [higher is better]
   Does the interviewer adapt the order and content of protocol questions based on what
   the participant has already revealed -- or does the interviewer cycle through scripted
   questions mechanically regardless of prior responses?

   Signs of adaptive sequencing: skipping a scripted question because its answer was
   already volunteered; reordering topics because a participant naturally led there;
   merging two topics because the participant covered both in one answer.

   Signs of mechanical cycling: asking a scripted question whose answer was already
   given; restarting a topic the participant fully addressed unprompted; ignoring participant-
   led transitions and forcing the scripted order anyway.
     2 = Every scripted question asked in order regardless of participant's prior answers; no adaptation.
     4 = Rare skips or reorders; mostly mechanical cycling through the protocol.
     6 = Some adaptation visible; interviewer skips or reorders in obvious cases but misses subtler opportunities.
     8 = Consistent adaptation; scripted questions used only when they add genuine new ground.
    10 = Protocol used as a guide, not a script; sequencing is fully responsive to what the participant has revealed.

---

Return ONLY valid JSON in this exact format:
{
  "scores": {
    "probe_tailoring":        {"score": <1-10>, "rationale": "<string>"},
    "ambiguity_resolution":   {"score": <1-10>, "rationale": "<string>"},
    "insight_density":        {"score": <1-10>, "rationale": "<string>"},
    "paraphrase_fidelity":    {"score": <1-10>, "rationale": "<string>"},
    "sustained_engagement":   {"score": <1-10>, "rationale": "<string>"},
    "adaptive_sequencing":    {"score": <1-10>, "rationale": "<string>"}
  },
  "summary": "<one paragraph narrative assessment focused on research effectiveness>"
}"""


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------

def judge_transcript(
    transcript_path: Path,
    protocol: Protocol,
    model: str = "gpt-4.1",
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

    # Normalise: LLMs occasionally flatten {"score": N, "rationale": "..."} to a plain string.
    # Coerce any non-dict value so downstream code can always assume dict shape.
    scores = raw.get("scores", {})
    for dim, val in list(scores.items()):
        if not isinstance(val, dict):
            scores[dim] = {"score": int(val) if str(val).isdigit() else 0, "rationale": str(val)}
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
    logger.info("Scores saved -> %s  (overall=%.2f)", scores_path, overall or 0)

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
    parser = argparse.ArgumentParser(
        description=(
            "Score one or more simulated interview transcripts.\n\n"
            "By default, transcripts that already have a _scores.json sibling are skipped.\n"
            "Pass --overwrite to re-judge them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("transcript", nargs="?", metavar="TRANSCRIPT",
                       help="Path to a single transcript JSON file")
    group.add_argument("--dir", metavar="DIR",
                       help="Directory -- judge all transcript JSON files inside it")
    group.add_argument("--transcripts", nargs="+", metavar="FILE",
                       help="Explicit list of transcript JSON files to judge")
    parser.add_argument("--protocol", required=True, help="Path to protocol JSON")
    parser.add_argument("--model", default="gpt-4o",
                        help="Judge LLM model (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-judge even if a _scores.json already exists (default: skip)")
    args = parser.parse_args()

    protocol = load_protocol(Path(args.protocol))

    # Resolve candidate transcript paths
    if args.transcript:
        candidates = [Path(args.transcript)]
    elif args.dir:
        candidates = sorted(
            p for p in Path(args.dir).glob("*.json")
            if not p.stem.endswith("_scores") and not p.stem.startswith("benchmark_")
        )
    else:
        candidates = [Path(p) for p in args.transcripts]

    if not candidates:
        print("No transcript files found.")
        return

    # Skip already-scored unless --overwrite
    if args.overwrite:
        to_judge = candidates
    else:
        to_judge = [p for p in candidates
                    if not p.with_name(p.stem + "_scores.json").exists()]
        skipped = len(candidates) - len(to_judge)
        if skipped:
            logger.info("Skipping %d already-scored transcript(s) -- use --overwrite to re-judge.", skipped)

    if not to_judge:
        print("All transcripts already scored. Use --overwrite to re-judge.")
        return

    single = len(to_judge) == 1 and args.transcript
    for transcript_path in to_judge:
        result = judge_transcript(
            transcript_path=transcript_path,
            protocol=protocol,
            model=args.model,
            temperature=args.temperature,
        )
        print(f"\n-- {transcript_path.name}  (overall: {result['overall']}/10.0)")
        if single:
            print(f"\nSummary: {result['summary']}")
        for dim, val in result["scores"].items():
            if isinstance(val, dict):
                print(f"  {dim:<24} {val['score']}/10  -- {val['rationale']}")
            else:
                print(f"  {dim:<24} {val}")


if __name__ == "__main__":
    main()
