from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

"""
Benchmark runner — samples agents from a population, randomises interviewer order
per agent, and runs all simulations. Transcripts are saved to:

    eval/results/<protocol_name>/<interviewer_id>_<agent_id>_<timestamp>.json

No judging is performed here. To score transcripts run:

    python eval/judge.py --dir eval/results/<protocol_name>/ --protocol <protocol.json>

Usage:
    python eval/benchmark.py \\
        --protocol sample_protocols/protocol.json \\
        --population genagents/agent_bank/populations/gss_agents \\
        --n-agents 5 \\
        --interviewers agentic single_llm \\
        --minutes-per-turn 2.0
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Agent sampling
# ---------------------------------------------------------------------------

def sample_agents(population_dir: Path, n: int, seed: int | None = None) -> list[Path]:
    """Randomly sample up to n agent folders from a population directory."""
    all_folders = [p for p in population_dir.iterdir() if p.is_dir()]
    if not all_folders:
        raise ValueError(f"No agent folders found in {population_dir}")
    rng = random.Random(seed)
    sampled = rng.sample(all_folders, min(n, len(all_folders)))
    logger.info("Sampled %d agents from %s", len(sampled), population_dir)
    return sampled


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    protocol_path: Path,
    population_dir: Path,
    interviewer_ids: list[str],
    n_agents: int = 5,
    minutes_per_turn: float = 1.0,
    interviewer_model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    seed: int | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Sample agents, run each through all interviewers in randomised order,
    and save transcripts organised by protocol name.
    Returns the list of transcript paths created.
    """
    from ai_interviewer.protocol import load_protocol
    from eval.simulate import run_simulation

    protocol = load_protocol(protocol_path)
    agent_folders = sample_agents(population_dir, n_agents, seed=seed)

    # For each agent, randomly shuffle interviewer order to counterbalance
    # order effects (agent memory persists across runs within a benchmark).
    rng = random.Random(seed)
    agent_assignments: list[dict] = []
    for folder in agent_folders:
        order = rng.sample(interviewer_ids, len(interviewer_ids))
        agent_assignments.append({"folder": folder, "interviewer_order": order})

    total = sum(len(a["interviewer_order"]) for a in agent_assignments)
    logger.info("Benchmark: %d agents × %d interviewers = %d simulations",
                len(agent_folders), len(interviewer_ids), total)

    transcript_paths: list[Path] = []

    for assignment in agent_assignments:
        agent_folder = assignment["folder"]
        for interviewer_id in assignment["interviewer_order"]:
            logger.info("─── %s × %s ───", interviewer_id, agent_folder.name)

            transcript_path = run_simulation(
                protocol_path=protocol_path,
                agent_folder=agent_folder,
                interviewer_id=interviewer_id,
                minutes_per_turn=minutes_per_turn,
                model=interviewer_model,
                temperature=temperature,
                output_dir=output_dir,
            )
            transcript_paths.append(transcript_path)

    return transcript_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full benchmark: interviewers × sampled agents. Saves transcripts only — no judging."
    )
    parser.add_argument("--protocol", required=True, help="Path to protocol JSON")
    parser.add_argument("--population", required=True,
                        help="Directory of genagents agent folders to sample from")
    parser.add_argument("--n-agents", type=int, default=5,
                        help="Number of agents to randomly sample (default: 5)")
    parser.add_argument("--interviewers", nargs="+",
                        default=["agentic", "single_llm"],
                        choices=["agentic", "single_llm"],
                        help="Interviewer systems to benchmark (default: all)")
    parser.add_argument("--minutes-per-turn", type=float, default=2.0,
                        help="Simulated minutes per interviewee turn (default: 2.0)")
    parser.add_argument("--interviewer-model", default="gpt-4o-mini",
                        help="LLM model for agentic/single_llm interviewers (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible agent sampling and order assignment")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save results (default: eval/results/)")
    args = parser.parse_args()

    transcript_paths = run_benchmark(
        protocol_path=Path(args.protocol),
        population_dir=Path(args.population),
        interviewer_ids=args.interviewers,
        n_agents=args.n_agents,
        minutes_per_turn=args.minutes_per_turn,
        interviewer_model=args.interviewer_model,
        temperature=args.temperature,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    print(f"\n{len(transcript_paths)} transcript(s) saved:")
    for p in transcript_paths:
        print(f"  {p}")
    if transcript_paths:
        print(f"\nTo judge, run:")
        print(f"  python eval/judge.py --dir {transcript_paths[0].parent} --protocol {args.protocol}")


if __name__ == "__main__":
    main()
