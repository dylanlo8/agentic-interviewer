"""Gradio chat interface for the Agentic Interviewer.

Run with:
    python app.py
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import gradio as gr

from ai_interviewer.llm import LLMConfig
from ai_interviewer.protocol import load_protocol
from ai_interviewer.state import InterviewState
from ai_interviewer.agents.active_listening import generate_turn
from ai_interviewer.agents.followup import generate_probe
from ai_interviewer.agents.summariser import summarise_async
from ai_interviewer.agents.topic_evaluator import evaluate_momentum_async
from ai_interviewer.agents.orchestrator import decide_action
from ai_interviewer.runner import (
    _init_state,
    _get_turn_text,
    _apply_action,
    _consume_momentum,
    _consume_summary,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

_protocol_path = Path(os.environ.get("AI_PROTOCOL_PATH", "sample_protocols/protocol_caregiver.json"))
PROTOCOL = load_protocol(_protocol_path)

CFG = LLMConfig(
    topic_eval_model=os.environ.get("TOPIC_EVAL_MODEL", "gpt-4o-mini"),
    topic_eval_base_url=os.environ.get("TOPIC_EVAL_BASE_URL") or None,
    followup_model=os.environ.get("FOLLOWUP_MODEL", "gpt-4o-mini"),
    followup_base_url=os.environ.get("FOLLOWUP_BASE_URL") or None,
    active_listening_model=os.environ.get("ACTIVE_LISTENING_MODEL", "gpt-4o-mini"),
    summariser_model=os.environ.get("SUMMARISER_MODEL", "gpt-4o-mini"),
    summariser_base_url=os.environ.get("SUMMARISER_BASE_URL") or None,
    temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
)


# Session helpers

def _make_session() -> dict:
    """Initialise a fresh interview session and generate the opening turn."""
    state = _init_state(PROTOCOL)
    topics = PROTOCOL.topics
    opening = generate_turn(topics[0].guiding_question, state, topics, CFG, is_first_turn=True)
    return {
        "state": state,
        "momentum_future": None,
        "summary_future": None,
        "done": False,
        "last_interviewer_turn": opening,
        "turn_start": time.time(),
    }


def _status_text(session: dict) -> str:
    state: InterviewState = session.get("state")
    if state is None:
        return ""
    topics = PROTOCOL.topics
    idx = min(state.current_topic_idx, len(topics) - 1)
    title = topics[idx].topic_title if topics else "—"
    budget = topics[idx].budget_minutes if topics else 0
    return (
        f"**Topic {idx + 1}/{len(topics)}:** {title} &nbsp;|&nbsp; "
        f"**Elapsed:** {state.elapsed_min:.1f} min &nbsp;|&nbsp; "
        f"**Topic time:** {state.topic_time_used:.1f}/{budget:.0f} min"
    )


# Interview logic (one turn at a time)

def start_interview():
    """Called on page load — initialise session and display opening question."""
    session = _make_session()
    history = [{"role": "assistant", "content": session["last_interviewer_turn"]}]
    return history, session, _status_text(session)


def respond(user_message: str, history: list, session: dict):
    """Process one interviewee turn; generator so user message appears immediately."""
    _placeholder = "Type your response and press Enter…"

    if not user_message.strip():
        yield history, session, _status_text(session), gr.update()
        return

    if session.get("done"):
        yield history, session, "Interview complete.", gr.update(interactive=False)
        return

    # Show user message immediately; disable input while thinking
    history = history + [{"role": "user", "content": user_message}]
    yield history, session, _status_text(session), gr.update(value="", interactive=False, placeholder="Thinking…")

    topics = PROTOCOL.topics
    state: InterviewState = session["state"]

    # Timing
    elapsed_turn = (time.time() - session["turn_start"]) / 60.0
    state.elapsed_min += elapsed_turn
    state.topic_time_used += elapsed_turn
    session["turn_start"] = time.time()

    # Update transcript
    state.transcript.append({"role": "interviewer", "content": session["last_interviewer_turn"]})
    state.transcript.append({"role": "interviewee", "content": user_message})

    # Consume previous async results
    _consume_momentum(session["momentum_future"], state)
    _consume_summary(session["summary_future"], state)

    # Orchestrator action
    action = decide_action(state, topics)
    logging.getLogger(__name__).info("[Orchestrator] action=%s", action)

    # WRAP_UP
    if action == "WRAP_UP":
        core = _get_turn_text("WRAP_UP", state, PROTOCOL)
        interviewer_turn = generate_turn(core, state, topics, CFG)
        state.transcript.append({"role": "interviewer", "content": interviewer_turn})
        history = history + [{"role": "assistant", "content": interviewer_turn}]
        session["done"] = True
        session["last_interviewer_turn"] = interviewer_turn
        yield history, session, "Interview complete.", gr.update(interactive=False, placeholder=_placeholder)
        return

    # Kick off async agents
    session["momentum_future"] = evaluate_momentum_async(state, topics, CFG)
    if len(state.transcript) > 8:
        session["summary_future"] = summarise_async(state, CFG)

    # Generate core content (aka Probing)
    probe_result: Optional[dict] = None
    if action == "PROBE":
        probe_result = generate_probe(state, topics, CFG)
        core = probe_result["probe_question"]
    else:
        # If not probe, either transition or wrapup
        core = _get_turn_text(action, state, PROTOCOL)

    _apply_action(action, state, PROTOCOL, probe_result)

    if state.current_topic_idx >= len(topics):
        core = _get_turn_text("WRAP_UP", state, PROTOCOL)

    # Active listening wrap
    interviewer_turn = generate_turn(core, state, topics, CFG)
    session["last_interviewer_turn"] = interviewer_turn

    history = history + [{"role": "assistant", "content": interviewer_turn}]
    yield history, session, _status_text(session), gr.update(interactive=True, placeholder=_placeholder)


# Gradio UI

with gr.Blocks(title="Agentic Interviewer") as demo:
    gr.Markdown(f"# {PROTOCOL.protocol_name}")

    if PROTOCOL.description:
        gr.Markdown(f"*{PROTOCOL.description}*")

    status_bar = gr.Markdown("")

    chatbot = gr.Chatbot(height=520, show_label=False)

    session_state = gr.State({})

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Type your response and press Enter…",
            scale=9,
            show_label=False,
            container=False,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")

    # Wire up interactions
    submit_inputs = [msg_box, chatbot, session_state]
    submit_outputs = [chatbot, session_state, status_bar, msg_box]

    send_btn.click(respond, submit_inputs, submit_outputs)
    msg_box.submit(respond, submit_inputs, submit_outputs)

    # Auto-start on page load
    demo.load(start_interview, outputs=[chatbot, session_state, status_bar])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
