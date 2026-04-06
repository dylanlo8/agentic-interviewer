from __future__ import annotations

"""
Seed an existing genagents agent with protocol-grounded memories.

Takes a population of agents and one or more protocols, generates realistic
biographical memory statements per agent per protocol, and saves them back
to disk. Run this once before benchmarking.

Usage:
    python eval/seed_agent.py \\
        --population genagents/agent_bank/populations/gss_agents \\
        --protocols sample_protocols/protocol_costofliving.json \\
        --n-agents 5 \\
        --seed 42

The script samples agents from the population, generates memories grounded
in each agent's demographics and each protocol's topic objectives, seeds the
agents via agent.remember(), and saves them back to their folders.
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from ai_interviewer.llm import _invoke_json, _invoke_raw
from ai_interviewer.protocol import Protocol, load_protocol
from eval.interviewee import load_agent, reset_agent_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SEED_SYSTEM = """\
You generate realistic, specific biographical memory statements for a fictional survey participant.

Rules:
- Each statement must be concrete and factual — include specific numbers, prices, places, or situations
- Statements must be consistent with the participant's demographics
- Write in third-person: "{name} pays $1,200/month rent in Little Rock..."
- One distinct fact per statement — keep each short and scannable
- Cover the topic objectives provided — generate 3 statements per topic
- Do NOT repeat facts across statements

Return a SINGLE valid JSON object with ALL memories in one flat array — do not split by topic, \
no comments: {"memories": ["...", "...", ...]}\
"""

SEED_USER = """\
PARTICIPANT DEMOGRAPHICS:
{demographics}

INTERVIEW PROTOCOL: {protocol_name}
{topics}

There are {n_topics} topics listed above. Generate exactly 3 concrete memory statements per topic \
({total_memories} statements total). Cover every topic — do not stop early.\
"""

# ---------------------------------------------------------------------------
# Memory generation
# ---------------------------------------------------------------------------

def _format_topics(protocol: Protocol) -> str:
    lines = []
    for topic in protocol.topics:
        lines.append(f"\nTopic: {topic.topic_title}")
        lines.append(f"  Guiding question: {topic.guiding_question}")
        for obj in (topic.objectives or []):
            lines.append(f"  Objective: {obj}")
    return "\n".join(lines)


def generate_seed_memories(
    demographics: dict,
    protocol: Protocol,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """
    Call an LLM to generate protocol-grounded memory statements for a participant.
    Returns a list of concrete memory strings ready to pass to agent.remember().
    """
    name = f"{demographics.get('first_name', '')} {demographics.get('last_name', '')}".strip() or "The participant"

    n_topics = len(protocol.topics)
    user_content = SEED_USER.format(
        demographics=json.dumps(demographics, indent=2),
        protocol_name=protocol.protocol_name,
        topics=_format_topics(protocol),
        n_topics=n_topics,
        total_memories=n_topics * 3,
    )
    system_content = SEED_SYSTEM.replace("{name}", name)

    try:
        parsed = _invoke_json(model, 0.8, system_content, user_content)
        return parsed["memories"]
    except (ValueError, KeyError):
        pass

    # Fallback: model returned multiple {"memories": [...]} objects — merge them
    raw = _invoke_raw(model, 0.8, system_content, user_content)
    all_memories = []
    for match in re.finditer(r'"memories"\s*:\s*(\[.*?\])', raw, re.DOTALL):
        all_memories.extend(json.loads(match.group(1)))
    if all_memories:
        return all_memories
    raise ValueError(f"Could not extract memories from LLM response:\n{raw}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def seed_population(
    population_dir: Path,
    protocols: list[Protocol],
    n_agents: int = 5,
    model: str = "gpt-4o-mini",
    seed: int | None = None,
) -> list[str]:
    """
    Sample n_agents from population_dir, generate memories for each protocol,
    seed and save each agent. Returns list of seeded agent IDs.
    """
    all_folders = [p for p in population_dir.iterdir() if p.is_dir()]
    if not all_folders:
        raise ValueError(f"No agent folders found in {population_dir}")

    rng = random.Random(seed)
    sampled = rng.sample(all_folders, min(n_agents, len(all_folders)))
    logger.info("Sampled %d agents from %s", len(sampled), population_dir)

    seeded_ids = []
    for folder in sampled:
        agent = load_agent(folder)
        demographics = agent.scratch if isinstance(agent.scratch, dict) else {}
        agent_name = f"{demographics.get('first_name', folder.name)}"

        # Clear agent memories each generation
        reset_agent_memory(agent)
        logger.info("Memory reset: agent=%s", agent_name)

        for protocol in protocols:
            logger.info("Generating memories: agent=%s | protocol=%s", agent_name, protocol.protocol_name)
            memories = generate_seed_memories(demographics, protocol, model=model)
            for memory in memories:
                # Add memory + Construct embeddings
                agent.remember(memory)
            logger.info("  Seeded %d memories", len(memories))

        agent.save(str(folder))
        logger.info("Saved agent → %s", folder)
        seeded_ids.append(folder.name)

    return seeded_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed genagents with protocol-grounded memories")
    parser.add_argument("--population", required=True,
                        help="Directory of genagents agent folders")
    parser.add_argument("--protocols", nargs="+", required=True,
                        help="One or more protocol JSON paths")
    parser.add_argument("--n-agents", type=int, default=5,
                        help="Number of agents to sample and seed (default: 5)")
    parser.add_argument("--model", default="gpt-5-mini",
                        help="LLM model for memory generation (default: gpt-4o-mini)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible agent sampling")
    args = parser.parse_args()

    protocols = [load_protocol(Path(p)) for p in args.protocols]

    seeded = seed_population(
        population_dir=Path(args.population),
        protocols=protocols,
        n_agents=args.n_agents,
        model=args.model,
        seed=args.seed,
    )

    print(f"\nSeeded {len(seeded)} agents:")
    for agent_id in seeded:
        print(f"  {agent_id}")


if __name__ == "__main__":
    main()
