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
    topic_eval_base_url: str = None     # set to Ollama endpoint to use a local model
    followup_model: str = "gpt-4o-mini"
    followup_base_url: str = None       # set to Ollama endpoint to use a local model
    active_listening_model: str = "gpt-4o-mini"
    summariser_model: str = "gpt-4o-mini"
    summariser_base_url: str = None       # set to Ollama endpoint to use a local model
    temperature: float = 0.2


def _invoke_raw(model: str, temperature: float, system_prompt: str, user_message: str, base_url: str = None) -> str:
    """Call an LLM and return the raw response string."""
    kwargs = {"model": model, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["api_key"] = "ollama"
    llm = ChatOpenAI(**kwargs)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    return llm.invoke(messages).content


def _invoke_json(model: str, temperature: float, system_prompt: str, user_message: str, base_url: str = None) -> dict:
    """Call an LLM and return a parsed JSON dict. Falls back to regex extraction on parse failure."""
    kwargs = {"model": model, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
        kwargs["api_key"] = "ollama"  # Ollama ignores this but the SDK requires a non-empty value
    else:
        # Force valid JSON output on OpenAI models — prevents malformed JSON and markdown fences
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    llm = ChatOpenAI(**kwargs)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    response = llm.invoke(messages)
    content = response.content

    if os.environ.get("LLM_DEBUG"):
        print(f"\n[LLM DEBUG] model={model}\n{content}\n")

    # Strip markdown code fences (```json ... ```) that LLMs sometimes inject
    def _strip_fences(s: str) -> str:
        s = s.strip()
        # Remove a single leading ```... line
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n", "", s)
        # Remove a single trailing ``` on its own line
        s = re.sub(r"\n```$", "", s)
        return s.strip()

    # Strip JS-style line comments (// ...) that LLMs sometimes inject
    def _strip_comments(s: str) -> str:
        return re.sub(r"^\s*//[^\n]*\n?", "", s, flags=re.MULTILINE)

    # Clean once up front
    cleaned = _strip_comments(_strip_fences(content))

    # 1) Try direct parse of cleaned content
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2) Regex fallback: extract first {...} block from cleaned content
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 3) As a last resort, show both raw and cleaned content in debug and raise
    if os.environ.get("LLM_DEBUG"):
        print("[LLM DEBUG] Failed to parse JSON. Cleaned content was:\n", cleaned)

    raise ValueError(f"Could not parse JSON from LLM response:\n{content}")
