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
import time
from pathlib import Path

import anthropic
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

Score the interview on each of the eight dimensions below. For each dimension give:
  - "score": an integer from 1 to 10
  - "rationale": two to three sentences citing specific evidence — quote exact turns or
    paraphrase specific exchanges. Generic rationale is not acceptable.

IMPORTANT CALIBRATION RULES:
- Scores of 9-10 require near-flawless performance. Reserve them for genuinely exceptional work.
- A score of 5-6 means adequate but with clear gaps. Most interviews land here.
- A score of 7-8 means good with identifiable strengths and minor weaknesses.
- You MUST use the full 1-10 range. Do not cluster around 5-7 out of caution.
- You MUST cite specific exchanges (quote content or reference turn patterns) in every rationale.

Scoring scale (higher = better for all dimensions):
  1-3 = Poor   4-6 = Acceptable   7-8 = Good   9-10 = Excellent (rare)

---

Dimensions (1–6 assess the PARTICIPANT's responses; 7–8 assess the INTERVIEWER's conduct):

1. relevance  [higher is better]
   Does the participant's response address what was actually asked, or does it drift,
   deflect, or answer a different question entirely?

   Score based on the proportion of participant turns that directly address the question posed.
     2 = Responses are consistently off-topic or tangential; rarely addresses the question asked.
     4 = Responses partially address questions but frequently drift or deflect.
     6 = Most responses are relevant; occasional tangents that the interviewer does not redirect.
     8 = Responses are consistently relevant; rare minor drift.
    10 = Every response directly and fully addresses the question asked.

2. specificity  [higher is better]
   Do the participant's responses contain specific concepts and detailed examples, or
   only generic descriptions?

   Distinguish three levels: (a) generic description only ("it was hard"), (b) specific
   concepts without examples ("my rent increased significantly"), (c) specific concepts
   with detailed examples ("my rent went from $900 to $1,150 in one renewal cycle").
     2 = Responses consist almost entirely of generic descriptions; no specific concepts or examples.
     4 = Some specific concepts appear but are rarely supported by concrete examples.
     6 = Mix of specific concepts and examples; generic descriptions still frequent.
     8 = Most responses contain specific concepts with detailed examples; rare generic turn.
    10 = Every response is grounded in specific concepts with detailed, concrete examples throughout.

3. clarity  [higher is better]
   Are the participant's responses easy to follow and unambiguous, or are they vague,
   contradictory, or hard to interpret?

     2 = Responses are consistently unclear, contradictory, or impossible to interpret without guessing.
     4 = Frequent ambiguity; meaning often requires inference; ideas poorly organised.
     6 = Most responses are understandable; occasional ambiguous passage.
     8 = Responses are clear and well-organised; rare ambiguity.
    10 = Every response is precise, logically ordered, and immediately interpretable.

4. informativeness  [higher is better]
   How much novel, non-obvious information do the participant's responses contain?

   Assess how surprising or unexpected the content is relative to what a generic person
   might say about this topic. A response that repeats common knowledge or stock phrases
   scores low; one that reveals personal specifics, unexpected perspectives, contradictions,
   or mechanisms scores high.
     2 = Responses consist almost entirely of predictable, generic information; no novel content.
     4 = Occasional specific detail amid mostly expected content.
     6 = Mix of novel and expected content; several turns add genuine new information.
     8 = Most responses contain non-obvious information; rare predictable turn.
    10 = Responses are consistently surprising and information-dense; nearly every turn adds novel content.

5. cognitive_empathy  [higher is better]
   Do the participant's responses surface the "why" behind their beliefs and experiences —
   explaining motivations, reasoning, or values — or do they stay at the level of
   what happened without explaining why?

     2 = Responses describe events or positions but never explain the reasoning or motivation behind them.
     4 = Rationale appears rarely; most responses are descriptive without explanation.
     6 = Rationale present in some responses; inconsistent across the interview.
     8 = Most responses include the participant's reasoning or motivation; rare descriptive-only turn.
    10 = Every response connects stated beliefs or experiences to an explicit rationale or motivation.

6. palpability  [higher is better]
   Do the participant's responses ground abstract beliefs or opinions in concrete personal
   evidence — lived experiences or hypothetical examples — or do they rely on
   abstractions and generalisations?

     2 = Responses are almost entirely abstract; no personal or hypothetical examples offered.
     4 = Rare concrete example; most responses stay at the level of generalisations.
     6 = Personal examples present but inconsistent; some turns remain abstract.
     8 = Most responses are grounded in concrete personal or hypothetical examples; rare abstraction.
    10 = Every response is anchored in a specific lived or hypothetical example; no unsupported generalisations.

7. self_awareness  [higher is better]
   Does the interviewer avoid expressing bias, assumptions, or leading framings? Does the
   participant engage naturally, without treating the interviewer as a bot or script?

   Penalise: leading questions, loaded phrasing, expressions of surprise or approval, yes/no
   framings, presuppositions embedded in questions, or any participant turn that signals
   awareness of interacting with an automated system.
     2 = Multiple instances of expressed bias, assumptions, or loaded framing; participant disengaged or robotic.
     4 = Several leading or assumptive questions; occasional participant response suggests scripted interaction.
     6 = Some neutral lapses; mostly non-directive but with noticeable moments of framing.
     8 = Interviewer is largely neutral; one or two minor framings that do not materially affect responses.
    10 = Fully neutral throughout; no bias, assumptions, or leading language; participant engages naturally.

8. follow_up  [higher is better]
   Does the conversation flow naturally, with the interviewer collecting data that responds
   to what arose during the interview itself? Or is the flow unnatural, repetitive, or broken?

   Penalise: asking the same question twice; abrupt topic switches that ignore what was just
   said; formulaic transitions that feel scripted; broken or incoherent exchanges.
     2 = Conversation is unnatural throughout — repetitive, broken, or entirely scripted in feel.
     4 = Frequent unnatural moments; flow is disrupted more often than not.
     6 = Generally natural with several unnatural moments; occasional repetition or abrupt pivot.
     8 = Flow is mostly natural and responsive; rare scripted or repetitive moment.
    10 = Conversation flows entirely naturally; every follow-up arises organically from what preceded it.

---

Return ONLY valid JSON in this exact format:
{
  "scores": {
    "relevance":         {"score": <1-10>, "rationale": "<string>"},
    "specificity":       {"score": <1-10>, "rationale": "<string>"},
    "clarity":           {"score": <1-10>, "rationale": "<string>"},
    "informativeness":   {"score": <1-10>, "rationale": "<string>"},
    "cognitive_empathy": {"score": <1-10>, "rationale": "<string>"},
    "palpability":       {"score": <1-10>, "rationale": "<string>"},
    "self_awareness":    {"score": <1-10>, "rationale": "<string>"},
    "follow_up":         {"score": <1-10>, "rationale": "<string>"}
  },
  "summary": "<one paragraph narrative assessment>"
}"""

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

def _format_protocol(protocol: Protocol) -> str:
    """Return only the protocol name and topic titles for domain context.
    Guiding questions, objectives, and time budgets are omitted to avoid
    anchoring the judge to scripted coverage rather than exchange quality."""
    lines = [f"Protocol: {protocol.protocol_name}"]
    lines.append("Topics covered:")
    for i, topic in enumerate(protocol.topics, 1):
        lines.append(f"  {i}. {topic.topic_title}")
    return "\n".join(lines)


def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for turn in transcript:
        role = "INTERVIEWER" if turn["role"] == "interviewer" else "PARTICIPANT"
        lines.append(f"{role}: {turn['content']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Batch judging via Anthropic Message Batches API
# ---------------------------------------------------------------------------

def judge_transcripts_batch(
    transcript_paths: list[Path],
    protocol: Protocol,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.0,
    poll_interval: int = 30,
) -> list[dict]:
    """
    Score multiple transcripts in one Anthropic Message Batch (async, ~50% cheaper).

    Submits all requests at once, polls until the batch is complete, then parses
    and writes a _scores.json file alongside each transcript.

    Returns a list of result dicts in the same order as transcript_paths.
    """
    client = anthropic.Anthropic()

    protocol_text = _format_protocol(protocol)

    # Build one request per transcript. custom_id maps back to the file path.
    requests: list[anthropic.types.message_create_params.Request] = []
    id_to_path: dict[str, Path] = {}
    for path in transcript_paths:
        transcript_data = json.loads(path.read_text())
        transcript_text = _format_transcript(transcript_data["transcript"])
        user_message = (
            f"PROTOCOL:\n{protocol_text}\n\n"
            f"TRANSCRIPT:\n{transcript_text}\n\n"
            f"Please score this interview."
        )
        custom_id = path.stem  # unique within a results dir
        id_to_path[custom_id] = path
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": 2048,
                "temperature": temperature,
                "system": JUDGE_SYSTEM,
                "messages": [{"role": "user", "content": user_message}],
            },
        })

    logger.info("Submitting batch of %d transcript(s) to Anthropic (model=%s)", len(requests), model)
    batch = client.messages.batches.create(requests=requests)
    logger.info("Batch created: id=%s", batch.id)

    # Poll until processing is complete
    while batch.processing_status != "ended":
        logger.info("Batch status: %s — waiting %ds…", batch.processing_status, poll_interval)
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch.id)

    logger.info("Batch complete. Processing results…")

    results: list[dict] = []
    result_map: dict[str, dict] = {}

    for item in client.messages.batches.results(batch.id):
        path = id_to_path[item.custom_id]
        transcript_data = json.loads(path.read_text())

        if item.result.type == "error":
            logger.error("Batch item %s failed: %s", item.custom_id, item.result.error)
            continue

        content = item.result.message.content[0].text

        # Reuse the same JSON cleaning logic as _invoke_json
        import re
        def _strip_fences(s: str) -> str:
            return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s.strip(), flags=re.MULTILINE)
        def _strip_comments(s: str) -> str:
            return re.sub(r"^\s*//[^\n]*\n?", "", s, flags=re.MULTILINE)

        stripped = _strip_comments(_strip_fences(content))
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            raw = json.loads(match.group()) if match else {}

        scores = raw.get("scores", {})
        for dim, val in list(scores.items()):
            if not isinstance(val, dict):
                scores[dim] = {"score": int(val) if str(val).isdigit() else 0, "rationale": str(val)}
        score_values = [v["score"] for v in scores.values() if isinstance(v, dict) and "score" in v]
        overall = round(sum(score_values) / len(score_values), 2) if score_values else None

        result = {
            "interviewer_id": transcript_data.get("interviewer_id", "unknown"),
            "agent_id": transcript_data.get("agent_id", "unknown"),
            "protocol_name": protocol.protocol_name,
            "transcript_file": path.name,
            "scores": scores,
            "overall": overall,
            "summary": raw.get("summary", ""),
        }

        logger.info("Scored %s  (overall=%.2f)", path.name, overall or 0)
        result_map[item.custom_id] = result

    # Return in original order
    for path in transcript_paths:
        if path.stem in result_map:
            results.append(result_map[path.stem])

    # Write combined scores file in a dedicated batch_scores/ subfolder
    combined_path: Path | None = None
    if results:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = transcript_paths[0].parent / "batch_scores"
        batch_dir.mkdir(exist_ok=True)
        combined_path = batch_dir / f"batch_scores_{ts}.json"
        combined_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info("Combined scores saved -> %s", combined_path)

    return results, combined_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(
        description=(
            "Score one or more simulated interview transcripts.\n\n"
            "By default, transcripts that already have a _scores.json sibling are skipped.\n"
            "Pass --overwrite to re-judge them.\n"
            "Pass --batch to submit all transcripts as a single Anthropic Message Batch (requires a claude-* model)."
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
    parser.add_argument("--batch", action="store_true",
                        help="Submit all transcripts as one Anthropic Message Batch (requires a claude-* model)")
    args = parser.parse_args()

    protocol = load_protocol(Path(args.protocol))

    # Resolve candidate transcript paths
    if args.transcript:
        candidates = [Path(args.transcript)]
    elif args.dir:
        candidates = sorted(
            p for p in Path(args.dir).glob("*.json")
            if not p.stem.endswith("_scores") and not p.stem.endswith("_coverage") and not p.stem.startswith("benchmark_")
        )
    else:
        candidates = [Path(p) for p in args.transcripts]

    if not candidates:
        print("No transcript files found.")
        return

    # Batch mode always judges all transcripts (results go to batch_scores/, never overwrite existing scores)
    if args.batch:
        to_judge = candidates
    elif args.overwrite:
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

    if args.batch:
        results, combined_path = judge_transcripts_batch(
            transcript_paths=to_judge,
            protocol=protocol,
            model=args.model,
            temperature=args.temperature,
        )
        for result in results:
            print(f"\n-- {result['transcript_file']}  (overall: {result['overall']}/10.0)")
            for dim, val in result["scores"].items():
                if isinstance(val, dict):
                    print(f"  {dim:<24} {val['score']}/10  -- {val['rationale']}")
                else:
                    print(f"  {dim:<24} {val}")
        if combined_path:
            print(f"\nCombined scores: {combined_path}")
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
