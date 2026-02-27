from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


@dataclass
class LLMConfig:
    topic_eval_model: str = "gpt-4o-mini"
    socratic_model: str = "gpt-4o-mini"
    active_listening_model: str = "gpt-4o-mini"
    temperature: float = 0.2


def _invoke_json(model: str, temperature: float, system_prompt: str, user_message: str) -> dict:
    """Call an LLM and return a parsed JSON dict. Falls back to regex extraction on parse failure."""
    llm = ChatOpenAI(model=model, temperature=temperature)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    response = llm.invoke(messages)
    content = response.content

    if os.environ.get("LLM_DEBUG"):
        print(f"\n[LLM DEBUG] model={model}\n{content}\n")

    # Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Regex fallback: extract first {...} block
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response:\n{content}")
