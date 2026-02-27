from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Topic:
    topic_id: str
    topic_title: str
    guiding_question: str
    budget_minutes: float


@dataclass
class Protocol:
    protocol_name: str
    version: str
    total_minutes: float
    wrapup_minutes: float
    max_followups_per_thread: int
    topics: list[Topic]


def load_protocol(path: Path) -> Protocol:
    data = json.loads(path.read_text())
    topics = [
        Topic(
            topic_id=t["topic_id"],
            topic_title=t["topic_title"],
            guiding_question=t["guiding_question"],
            budget_minutes=t["budget_minutes"],
        )
        for t in data["topics"]
    ]
    return Protocol(
        protocol_name=data["protocol_name"],
        version=data["version"],
        total_minutes=data["total_minutes"],
        wrapup_minutes=data["wrapup_minutes"],
        max_followups_per_thread=data["max_followups_per_thread"],
        topics=topics,
    )
