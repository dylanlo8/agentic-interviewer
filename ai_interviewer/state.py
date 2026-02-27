from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InterviewState:
    # Time tracking
    elapsed_min: float = 0.0
    total_min: float = 45.0
    wrapup_min: float = 3.0

    # Topic management
    current_topic_idx: int = 0
    topic_time_used: float = 0.0

    # Follow-up tracking
    followups_in_thread: int = 0
    max_followups_per_thread: int = 10

    # Memory
    open_loops: list[str] = field(default_factory=list)
    topic_momentum: bool = True

    # Full conversation history
    transcript: list[dict] = field(default_factory=list)
