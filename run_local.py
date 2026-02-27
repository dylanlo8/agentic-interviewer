from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from ai_interviewer.llm import LLMConfig
from ai_interviewer.protocol import load_protocol
from ai_interviewer.runner import run_interview


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    protocol_path = Path(os.environ.get("AI_PROTOCOL_PATH", "notebooks/protocol.json"))
    protocol = load_protocol(protocol_path)

    cfg = LLMConfig(
        topic_eval_model=os.environ.get("TOPIC_EVAL_MODEL", "gpt-4o-mini"),
        socratic_model=os.environ.get("SOCRATIC_MODEL", "gpt-4o-mini"),
        active_listening_model=os.environ.get("ACTIVE_LISTENING_MODEL", "gpt-4o-mini"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
    )

    run_interview(protocol, cfg)


if __name__ == "__main__":
    main()
