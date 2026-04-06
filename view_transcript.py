#!/usr/bin/env python3
"""
Transcript Viewer — browse and read interview transcripts from eval/results/.

Usage:
  python view_transcript.py                  # interactive file picker
  python view_transcript.py <path/to/file>   # open a specific transcript
"""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import box

RESULTS_DIR = Path(__file__).parent / "eval" / "results"
console = Console()

ROLE_STYLES = {
    "interviewer": ("bold cyan", "Interviewer"),
    "interviewee": ("bold green", "Interviewee"),
}


def find_transcripts() -> list[Path]:
    """Return all non-scores transcript JSON files, sorted by folder then name."""
    return sorted(
        p for p in RESULTS_DIR.rglob("*.json")
        if "_scores" not in p.name
    )


def pick_transcript(files: list[Path]) -> Path:
    """Interactive numbered menu to select a transcript file."""
    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold white")
    table.add_column("#", style="dim", width=4)
    table.add_column("Protocol", style="yellow")
    table.add_column("Interviewer", style="cyan")
    table.add_column("Agent ID", style="white")
    table.add_column("Date / Time", style="dim")

    for i, f in enumerate(files, 1):
        parts = f.stem.split("_")
        # filename pattern: <interviewer_type>_<uuid>_<date>_<time>
        # uuid is 5 groups joined by -, so split carefully
        # split on _ but uuid has 5 segments: handle by rejoining middle parts
        interviewer = parts[0] if parts else "?"
        date_time = f"{parts[-2]} {parts[-1]}" if len(parts) >= 2 else "?"
        agent_id = "_".join(parts[1:-2]) if len(parts) >= 4 else "?"
        protocol = f.parent.name
        table.add_row(str(i), protocol, interviewer, agent_id, date_time)

    console.print()
    console.print(table)

    choice = Prompt.ask(
        "[bold]Enter number[/bold]",
        console=console,
    )
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(files)):
            raise ValueError
        return files[idx]
    except ValueError:
        console.print("[red]Invalid selection.[/red]")
        sys.exit(1)


def render_transcript(path: Path) -> None:
    """Load and pretty-print a transcript file."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[red]Failed to read file:[/red] {e}")
        sys.exit(1)

    # ── Header ────────────────────────────────────────────────────────────────
    protocol = data.get("protocol_name", "Unknown Protocol")
    interviewer_id = data.get("interviewer_id", "?")
    agent_id = data.get("agent_id", "?")
    timestamp = data.get("timestamp", "?")
    turns = data.get("turns", "?")
    minutes_per_turn = data.get("minutes_per_turn", "?")

    header = (
        f"[bold white]{protocol}[/bold white]\n"
        f"[dim]Interviewer:[/dim] [cyan]{interviewer_id}[/cyan]   "
        f"[dim]Agent:[/dim] [white]{agent_id}[/white]\n"
        f"[dim]Timestamp:[/dim] {timestamp}   "
        f"[dim]Turns:[/dim] {turns}   "
        f"[dim]Min/turn:[/dim] {minutes_per_turn}"
    )
    console.print(Panel(header, title="[bold]Transcript[/bold]", border_style="bright_blue"))
    console.print()

    # ── Turns ─────────────────────────────────────────────────────────────────
    transcript = data.get("transcript", [])
    if not transcript:
        console.print("[yellow]No transcript entries found.[/yellow]")
        return

    for i, entry in enumerate(transcript, 1):
        role = entry.get("role", "unknown").lower()
        content = entry.get("content", "")
        style, label = ROLE_STYLES.get(role, ("white", role.capitalize()))

        turn_label = Text(f"[{i}] {label}", style=style)
        console.print(turn_label)
        console.print(f"  {content}")
        console.print()

    console.print(
        Panel(
            f"[dim]End of transcript — {len(transcript)} entries[/dim]",
            border_style="dim",
        )
    )


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            console.print(f"[red]File not found:[/red] {path}")
            sys.exit(1)
    else:
        files = find_transcripts()
        if not files:
            console.print(f"[red]No transcript files found in[/red] {RESULTS_DIR}")
            sys.exit(1)
        path = pick_transcript(files)

    render_transcript(path)


if __name__ == "__main__":
    main()
