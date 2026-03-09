from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor

from ai_interviewer.llm import LLMConfig, _invoke_json
from ai_interviewer.prompts import SUMMARISER_SYSTEM
from ai_interviewer.state import InterviewState

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1)

_RECENT_TURNS = 4  # Must match the default in build_context()


def summarise_async(state: InterviewState, cfg: LLMConfig) -> Future[str]:
    """
    Kick off a non-blocking summary update.
    Returns a Future[str] — call .result() on the next turn to consume the updated summary.
    Falls back to the prior summary on timeout or error.
    """
    older_turns = state.transcript[:-_RECENT_TURNS] if len(state.transcript) > _RECENT_TURNS else []
    snapshot = {
        "older_turns": older_turns,
        "prior_summary": state.conversation_summary,
    }
    logger.info("[Summariser] dispatched async call — %d older turns to summarise", len(older_turns))
    return _executor.submit(_summarise, snapshot, cfg)


def _summarise(snapshot: dict, cfg: LLMConfig) -> str:
    t0 = time.time()

    older_turns = snapshot["older_turns"]
    prior_summary = snapshot["prior_summary"]

    transcript_text = "\n".join(
        f"  {t['role'].upper()}: {t['content']}" for t in older_turns
    ) or "  (none)"

    user_message = ""
    if prior_summary:
        user_message += f"PRIOR SUMMARY\n  {prior_summary}\n\n"
    user_message += f"TRANSCRIPT (older turns, excluding the most recent {_RECENT_TURNS})\n{transcript_text}"

    result = _invoke_json(
        cfg.summariser_model,
        cfg.temperature,
        SUMMARISER_SYSTEM,
        user_message,
        base_url=cfg.summariser_base_url,
    )
    summary = result.get("summary", prior_summary or "")
    elapsed = time.time() - t0
    logger.info("[Summariser] updated summary (%.2fs): %s", elapsed, summary[:80])
    return summary
