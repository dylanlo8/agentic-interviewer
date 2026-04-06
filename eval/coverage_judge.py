from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""
Protocol coverage evaluator — scores transcripts on how well the interviewer
covered the protocol's topics and objectives.

Unlike judge.py (which strips the protocol to topic titles), this judge receives
the FULL protocol (guiding questions + objectives) and evaluates structural coverage.

Usage:
    python eval/coverage_judge.py eval/results/<transcript>.json \\
        --protocol sample_protocols/protocol_costofliving.json \\
        --model claude-sonnet-4-6

    python eval/coverage_judge.py \\
        --dir eval/results/protocol_costofliving/ \\
        --protocol sample_protocols/protocol_costofliving.json \\
        --model claude-sonnet-4-6 \\
        --batch

Writes a _coverage.json file alongside each transcript, and a combined
batch_coverage/batch_coverage_<ts>.json when run with --batch.
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from ai_interviewer.llm import _invoke_json
from ai_interviewer.protocol import Protocol, load_protocol

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coverage judge system prompt
# ---------------------------------------------------------------------------

COVERAGE_SYSTEM = """You are a rigorous evaluator of qualitative research interviews.

You will be given:
1. A FULL PROTOCOL — including each topic's title, guiding question, and stated objectives.
2. A FULL TRANSCRIPT of the interview.

Your task is to assess how well the INTERVIEWER covered the protocol's topics and objectives.
This is a structural evaluation, not a response-quality evaluation.

Score the interview on each of the three dimensions below. For each dimension give:
  - "score": an integer from 1 to 10
  - "rationale": two to three sentences citing specific evidence — name specific topics,
    objectives, or transcript moments. Generic rationale is not acceptable.

IMPORTANT CALIBRATION RULES:
- Scores of 9-10 require near-flawless performance. Reserve them for genuinely exceptional work.
- A score of 5-6 means adequate but with clear gaps. Most interviews land here.
- A score of 7-8 means good with identifiable strengths and minor weaknesses.
- You MUST use the full 1-10 range. Do not cluster around 5-7 out of caution.
- You MUST cite specific topics or objectives by name in every rationale.

Scoring scale (higher = better for all dimensions):
  1-3 = Poor   4-6 = Acceptable   7-8 = Good   9-10 = Excellent (rare)

Each dimension provides behavioural anchors at 2, 4, 6, 8, and 10.
Use odd numbers when the interview falls between two anchors.

---

Dimensions:

1. topic_breadth_coverage  [higher is better]
   Did the interviewer meaningfully open and explore every protocol topic,
   or were one or more topics skipped, rushed through, or never opened?

   A topic is "meaningfully opened" if the interviewer asked the guiding question
   (or a faithful paraphrase) AND followed up at least once within that topic.
   A topic opened with only the guiding question and no follow-up scores partial credit.

   Penalise: topics entirely absent from the transcript; topics where only the guiding
   question was asked with no follow-up; abrupt termination before the final topic(s);
   disproportionate time crowding out later topics.
     2 = Two or more topics entirely absent; interview ends before the closing topic is reached.
     4 = All topics technically appear but one or two receive only a single guiding question with no follow-up.
     6 = All topics opened and followed up; one topic visibly truncated relative to its scope.
     8 = All topics opened and substantively explored; minor time imbalance only.
    10 = Every topic opened, probed, and brought to a natural close; coverage is thorough and proportional.

2. objective_saturation  [higher is better]
   Across all topics, to what degree do the combined interviewer questions and
   participant responses address the stated objectives?

   An objective is "addressed" if the participant provided substantive content
   (not just a passing mention) on that objective, whether prompted or volunteered.
   Assess holistically across all topics.

   Penalise: objectives with no substantive content from either side; objectives
   acknowledged only at a surface level ("yes, I've thought about that"); systematic
   omission of a particular objective type across multiple topics.
     2 = Fewer than half of all objectives across all topics receive any substantive attention.
     4 = Roughly half of objectives addressed; the other half untouched or only superficially mentioned.
     6 = Most objectives partially addressed; a few notable gaps remain, especially on complex objectives.
     8 = Nearly all objectives meaningfully addressed; minor gaps on one or two secondary objectives only.
    10 = Every stated objective is addressed with substantive content from either the interviewer or participant.

3. objective_balance  [higher is better]
   Within each topic, does the interviewer distribute attention across all stated
   objectives, or does it fixate on one while ignoring the others?

   This is distinct from objective_saturation: a topic can have half its objectives
   saturated while all turns focus on just one of them — that is a balance failure
   even if saturation is moderate.

   Penalise: any topic where one objective dominates and at least one other receives
   zero turns; re-probing an already-covered objective when others remain uncovered;
   the interviewer chasing a participant-volunteered thread without pivoting to uncovered objectives.
     2 = Persistent fixation in almost every topic; one dominant objective and one or more completely ignored.
     4 = Fixation occurs in several topics; some effort to broaden but the pattern is visible.
     6 = Most topics distribute attention across objectives; one or two topics show clear imbalance.
     8 = Nearly all topics show distributed attention; minor emphasis variation only.
    10 = Every topic's objectives receive proportional attention; no objective crowded out within any topic.

---

Return ONLY valid JSON in this exact format:
{
  "scores": {
    "topic_breadth_coverage": {"score": <1-10>, "rationale": "<string>"},
    "objective_saturation":   {"score": <1-10>, "rationale": "<string>"},
    "objective_balance":      {"score": <1-10>, "rationale": "<string>"}
  },
  "summary": "<one paragraph narrative assessment focused on protocol coverage>"
}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_protocol_full(protocol: Protocol) -> str:
    """Return the full protocol with guiding questions and objectives."""
    lines = [f"Protocol: {protocol.protocol_name}"]
    for i, topic in enumerate(protocol.topics, 1):
        lines.append(f"\nTopic {i}: {topic.topic_title}")
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


def _parse_coverage_json(content: str) -> dict:
    """Parse JSON from LLM response, stripping fences and comments."""
    def _strip_fences(s: str) -> str:
        return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s.strip(), flags=re.MULTILINE)
    def _strip_comments(s: str) -> str:
        return re.sub(r"^\s*//[^\n]*\n?", "", s, flags=re.MULTILINE)

    stripped = _strip_comments(_strip_fences(content))
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            return json.loads(match.group())
    raise ValueError(f"Could not parse JSON from LLM response:\n{content}")


def _build_result(transcript_data: dict, protocol: Protocol, raw: dict, transcript_file: str) -> dict:
    scores = raw.get("scores", {})
    for dim, val in list(scores.items()):
        if not isinstance(val, dict):
            scores[dim] = {"score": int(val) if str(val).isdigit() else 0, "rationale": str(val)}
    score_values = [v["score"] for v in scores.values() if isinstance(v, dict) and "score" in v]
    overall = round(sum(score_values) / len(score_values), 2) if score_values else None
    return {
        "interviewer_id": transcript_data.get("interviewer_id", "unknown"),
        "agent_id": transcript_data.get("agent_id", "unknown"),
        "protocol_name": protocol.protocol_name,
        "transcript_file": transcript_file,
        "scores": scores,
        "overall": overall,
        "summary": raw.get("summary", ""),
    }


# ---------------------------------------------------------------------------
# Single-transcript scoring
# ---------------------------------------------------------------------------

def score_coverage(
    transcript_path: Path,
    protocol: Protocol,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
) -> dict:
    """
    Score a single transcript on the three coverage dimensions.
    Writes a _coverage.json sibling file and returns the result dict.
    """
    transcript_data = json.loads(transcript_path.read_text())
    transcript = transcript_data["transcript"]

    protocol_text = _format_protocol_full(protocol)
    transcript_text = _format_transcript(transcript)

    user_message = (
        f"FULL PROTOCOL:\n{protocol_text}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n\n"
        f"Please score this interview on protocol coverage."
    )

    logger.info("Scoring coverage: interviewer=%s | agent=%s | turns=%d",
                transcript_data.get("interviewer_id", "?"),
                transcript_data.get("agent_id", "?"),
                transcript_data.get("turns", "?"))

    raw = _invoke_json(model, temperature, COVERAGE_SYSTEM, user_message)
    result = _build_result(transcript_data, protocol, raw, transcript_path.name)

    coverage_path = transcript_path.with_name(transcript_path.stem + "_coverage.json")
    coverage_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Coverage scores saved -> %s  (overall=%.2f)", coverage_path, result["overall"] or 0)

    return result


# ---------------------------------------------------------------------------
# Batch scoring via Anthropic Message Batches API
# ---------------------------------------------------------------------------

def score_coverage_batch(
    transcript_paths: list[Path],
    protocol: Protocol,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
    poll_interval: int = 30,
) -> tuple[list[dict], Path | None]:
    """
    Score multiple transcripts in one Anthropic Message Batch (~50% cheaper).
    Writes individual _coverage.json files and a combined batch_coverage/ file.
    Returns (results, combined_path).
    """
    client = anthropic.Anthropic()
    protocol_text = _format_protocol_full(protocol)

    requests: list[anthropic.types.message_create_params.Request] = []
    id_to_path: dict[str, Path] = {}

    for path in transcript_paths:
        transcript_data = json.loads(path.read_text())
        transcript_text = _format_transcript(transcript_data["transcript"])
        user_message = (
            f"FULL PROTOCOL:\n{protocol_text}\n\n"
            f"TRANSCRIPT:\n{transcript_text}\n\n"
            f"Please score this interview on protocol coverage."
        )
        custom_id = path.stem
        id_to_path[custom_id] = path
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": 2048,
                "temperature": temperature,
                "system": COVERAGE_SYSTEM,
                "messages": [{"role": "user", "content": user_message}],
            },
        })

    logger.info("Submitting coverage batch of %d transcript(s) (model=%s)", len(requests), model)
    batch = client.messages.batches.create(requests=requests)
    logger.info("Batch created: id=%s", batch.id)

    while batch.processing_status != "ended":
        logger.info("Batch status: %s — waiting %ds…", batch.processing_status, poll_interval)
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch.id)

    logger.info("Batch complete. Processing results…")

    result_map: dict[str, dict] = {}

    for item in client.messages.batches.results(batch.id):
        path = id_to_path[item.custom_id]
        transcript_data = json.loads(path.read_text())

        if item.result.type == "error":
            logger.error("Batch item %s failed: %s", item.custom_id, item.result.error)
            continue

        content = item.result.message.content[0].text
        raw = _parse_coverage_json(content)
        result = _build_result(transcript_data, protocol, raw, path.name)

        coverage_path = path.with_name(path.stem + "_coverage.json")
        coverage_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info("Coverage saved -> %s  (overall=%.2f)", path.name, result["overall"] or 0)

        result_map[item.custom_id] = result

    # Return in original order
    results = [result_map[p.stem] for p in transcript_paths if p.stem in result_map]

    # Write combined file in batch_coverage/ subfolder
    combined_path: Path | None = None
    if results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = transcript_paths[0].parent / "batch_coverage"
        batch_dir.mkdir(exist_ok=True)
        combined_path = batch_dir / f"batch_coverage_{ts}.json"
        combined_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info("Combined coverage saved -> %s", combined_path)

    return results, combined_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(
        description=(
            "Score transcripts on protocol coverage (topic breadth, objective saturation, objective balance).\n\n"
            "Uses the FULL protocol (guiding questions + objectives).\n"
            "Pass --batch to submit all transcripts as one Anthropic Message Batch."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("transcript", nargs="?", metavar="TRANSCRIPT",
                       help="Path to a single transcript JSON file")
    group.add_argument("--dir", metavar="DIR",
                       help="Directory — score all transcript JSON files inside it")
    group.add_argument("--transcripts", nargs="+", metavar="FILE",
                       help="Explicit list of transcript JSON files to score")
    parser.add_argument("--protocol", required=True, help="Path to protocol JSON")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Judge LLM model (default: claude-sonnet-4-6)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-score even if a _coverage.json already exists (default: skip)")
    parser.add_argument("--batch", action="store_true",
                        help="Submit all transcripts as one Anthropic Message Batch")
    args = parser.parse_args()

    protocol = load_protocol(Path(args.protocol))

    # Resolve candidates
    if args.transcript:
        candidates = [Path(args.transcript)]
    elif args.dir:
        candidates = sorted(
            p for p in Path(args.dir).glob("*.json")
            if not p.stem.endswith("_scores")
            and not p.stem.endswith("_coverage")
            and not p.stem.startswith("benchmark_")
        )
    else:
        candidates = [Path(p) for p in args.transcripts]

    if not candidates:
        print("No transcript files found.")
        return

    # Batch mode always scores all candidates (results go to _coverage.json / batch_coverage/, never overwrite _scores.json)
    if args.batch:
        to_score = candidates
    elif args.overwrite:
        to_score = candidates
    else:
        to_score = [p for p in candidates if not p.with_name(p.stem + "_coverage.json").exists()]
        skipped = len(candidates) - len(to_score)
        if skipped:
            logger.info("Skipping %d already-scored transcript(s) — use --overwrite to re-score.", skipped)

    if not to_score:
        print("All transcripts already scored. Use --overwrite to re-score.")
        return

    if args.batch:
        results, combined_path = score_coverage_batch(
            transcript_paths=to_score,
            protocol=protocol,
            model=args.model,
            temperature=args.temperature,
        )
        for result in results:
            print(f"\n-- {result['transcript_file']}  (overall: {result['overall']}/10.0)")
            for dim, val in result["scores"].items():
                if isinstance(val, dict):
                    print(f"  {dim:<28} {val['score']}/10  -- {val['rationale'][:120]}")
        if combined_path:
            print(f"\nCombined coverage scores: {combined_path}")
        return

    single = len(to_score) == 1 and args.transcript
    for transcript_path in to_score:
        result = score_coverage(
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
                print(f"  {dim:<28} {val['score']}/10  -- {val['rationale']}")
            else:
                print(f"  {dim:<28} {val}")


if __name__ == "__main__":
    main()
