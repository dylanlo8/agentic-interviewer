from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

"""
Benchmark runner — runs all interviewers × all agents and produces a comparison report.

Usage:
    python eval/benchmark.py \\
        --protocol notebooks/protocol.json \\
        --agents eval/agents/ \\
        --interviewers agentic scripted single_llm \\
        --minutes-per-turn 2.0 \\
        --judge-model gpt-4o

Results are saved to eval/results/benchmark_<timestamp>.json
A summary table is printed to the terminal.
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

DIMENSIONS = ["topic_coverage", "response_depth", "question_quality", "active_listening", "pacing"]


def run_benchmark(
    protocol_path: Path,
    agents_dir: Path,
    interviewer_ids: list[str],
    minutes_per_turn: float = 2.0,
    interviewer_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    temperature: float = 0.2,
) -> Path:
    """
    Run all combinations of interviewers × agents, judge each transcript,
    and write a benchmark report. Returns the path to the report JSON.
    """
    from ai_interviewer.protocol import load_protocol
    from eval.judge import judge_transcript
    from eval.simulate import run_simulation

    protocol = load_protocol(protocol_path)
    agent_folders = sorted(p for p in agents_dir.iterdir() if p.is_dir())

    if not agent_folders:
        raise ValueError(f"No agent folders found in {agents_dir}")

    logger.info("Benchmark: %d interviewers × %d agents = %d simulations",
                len(interviewer_ids), len(agent_folders), len(interviewer_ids) * len(agent_folders))

    all_scores: list[dict] = []

    for interviewer_id in interviewer_ids:
        for agent_folder in agent_folders:
            logger.info("─── %s × %s ───", interviewer_id, agent_folder.name)

            # Run simulation → transcript JSON
            transcript_path = run_simulation(
                protocol_path=protocol_path,
                agent_folder=agent_folder,
                interviewer_id=interviewer_id,
                minutes_per_turn=minutes_per_turn,
                model=interviewer_model,
                temperature=temperature,
            )

            # Judge transcript → scores JSON
            scores = judge_transcript(
                transcript_path=transcript_path,
                protocol=protocol,
                model=judge_model,
                temperature=0.0,
            )
            all_scores.append(scores)

    # Aggregate and save report
    report = _build_report(interviewer_ids, all_scores, protocol.protocol_name)
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"benchmark_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Benchmark report saved → %s", report_path)

    _print_summary_table(report)
    return report_path


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _build_report(interviewer_ids: list[str], all_scores: list[dict], protocol_name: str) -> dict:
    """Aggregate per-run scores into means per interviewer."""
    # Group by interviewer
    by_interviewer: dict[str, list[dict]] = {iid: [] for iid in interviewer_ids}
    for s in all_scores:
        iid = s.get("interviewer_id", "unknown")
        if iid in by_interviewer:
            by_interviewer[iid].append(s)

    aggregated = {}
    for iid, runs in by_interviewer.items():
        if not runs:
            continue
        dim_means = {}
        for dim in DIMENSIONS:
            values = [r["scores"][dim]["score"] for r in runs if dim in r.get("scores", {})]
            dim_means[dim] = round(sum(values) / len(values), 2) if values else None
        overalls = [r["overall"] for r in runs if r.get("overall") is not None]
        aggregated[iid] = {
            "runs": len(runs),
            "dimension_means": dim_means,
            "overall_mean": round(sum(overalls) / len(overalls), 2) if overalls else None,
        }

    return {
        "protocol_name": protocol_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interviewers": aggregated,
        "individual_runs": all_scores,
    }


def _print_summary_table(report: dict) -> None:
    header_dims = ["Coverage", "Depth", "Q-Quality", "Active-L", "Pacing", "MEAN"]
    col_w = 10
    name_w = 16

    separator = "-" * (name_w + col_w * len(header_dims) + 2)
    header = f"{'Interviewer':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in header_dims)

    print(f"\n{'=' * len(separator)}")
    print(f"  Benchmark: {report['protocol_name']}")
    print(f"{'=' * len(separator)}")
    print(header)
    print(separator)

    for iid, agg in report["interviewers"].items():
        dims = agg["dimension_means"]
        row_vals = [
            dims.get("topic_coverage"),
            dims.get("response_depth"),
            dims.get("question_quality"),
            dims.get("active_listening"),
            dims.get("pacing"),
            agg.get("overall_mean"),
        ]
        row = f"{iid:<{name_w}}" + "".join(
            f"{(f'{v:.1f}' if v is not None else '-'):>{col_w}}" for v in row_vals
        )
        print(row)

    print(separator)
    print(f"  (n={list(report['interviewers'].values())[0]['runs']} agents per interviewer)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmark: interviewers × agents")
    parser.add_argument("--protocol", required=True, help="Path to protocol JSON")
    parser.add_argument("--agents", required=True, help="Directory containing agent folders")
    parser.add_argument("--interviewers", nargs="+",
                        default=["agentic", "scripted", "single_llm"],
                        choices=["agentic", "scripted", "single_llm"],
                        help="Interviewer systems to benchmark (default: all three)")
    parser.add_argument("--minutes-per-turn", type=float, default=2.0,
                        help="Simulated minutes per interviewee turn (default: 2.0)")
    parser.add_argument("--interviewer-model", default="gpt-4o-mini",
                        help="LLM model for agentic/single_llm interviewers (default: gpt-4o-mini)")
    parser.add_argument("--judge-model", default="gpt-4o",
                        help="LLM model for the transcript judge (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    report_path = run_benchmark(
        protocol_path=Path(args.protocol),
        agents_dir=Path(args.agents),
        interviewer_ids=args.interviewers,
        minutes_per_turn=args.minutes_per_turn,
        interviewer_model=args.interviewer_model,
        judge_model=args.judge_model,
        temperature=args.temperature,
    )
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
