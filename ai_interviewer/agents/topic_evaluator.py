from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor

from ai_interviewer.llm import LLMConfig, _invoke_json
from ai_interviewer.prompts import TOPIC_EVALUATOR_SYSTEM, build_context
from ai_interviewer.state import InterviewState

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1)


def evaluate_momentum_async(
    state: InterviewState, 
    topics: list, 
    cfg: LLMConfig
) -> Future[bool]:
    
    """
    Kick off a non-blocking momentum evaluation.
    Returns a Future[bool] — call .result() on the next turn to consume the signal.
    Defaults to True (continue probing) if the future is not ready or fails.
    """
    topic = topics[min(state.current_topic_idx, len(topics) - 1)]
    logger.info("[TopicEvaluator] dispatched async call — topic: '%s'", topic.topic_title)

    snapshot = {
        "context": build_context(state, topics),
        "topic_title": topic.topic_title,
    }

    # Submit the evaluation task to the executor
    return _executor.submit(_evaluate, snapshot, cfg)


def _evaluate(
    snapshot: dict, 
    cfg: LLMConfig
) -> bool:
    
    t0 = time.time()
    result = _invoke_json(
        cfg.topic_eval_model,
        cfg.temperature,
        TOPIC_EVALUATOR_SYSTEM,
        f"Evaluate momentum for the current topic.\n\n{snapshot['context']}",
        base_url=cfg.topic_eval_base_url,
    )
    continue_probing = bool(result.get("continue_probing", True))
    reason = result.get("reason", "")

    # Implement logging for the evaluation result
    elapsed = time.time() - t0
    logger.info(
        "[TopicEvaluator] result for '%s' — continue_probing=%s | reason: %s (%.2fs)",
        snapshot["topic_title"],
        continue_probing,
        reason,
        elapsed,
    )

    return continue_probing
