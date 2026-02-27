# AI Interviewer (Local Runner)

Local Python runner using LangChain LLMs and a protocol JSON.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export AI_PROTOCOL_PATH=notebooks/protocol.json
export ROUTER_MODEL=gpt-4o-mini
export PROBE_MODEL=gpt-4o-mini
export SCRIPT_MODEL=gpt-4o-mini
python run_local.py
```

## Notes
- The protocol schema is in `notebooks/protocol.json`.
- The local runner expects `total_minutes`, `wrapup_minutes`, `max_followups_per_thread`, and per-topic `budget_minutes`.
