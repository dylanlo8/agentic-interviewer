from __future__ import annotations

"""
Aggregate and compare judge scores across interviewers for a protocol.

Usage:
    python eval/analyze.py --dir eval/results/protocol_costofliving/

    # Save to CSV as well:
    python eval/analyze.py --dir eval/results/protocol_costofliving/ --csv

Output: a per-dimension comparison table (mean ± std, min–max) and an
overall ranking, grouped by interviewer.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DIMENSIONS = [
    "probe_tailoring", "ambiguity_resolution", "insight_density",
    "paraphrase_fidelity", "sustained_engagement", "adaptive_sequencing",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(results_dir: Path) -> list[dict]:
    """Return all score records from *_scores.json files in the directory."""
    files = sorted(
        p for p in results_dir.glob("*_scores.json")
        if not p.stem.startswith("benchmark_")
    )
    if not files:
        raise FileNotFoundError(f"No *_scores.json files found in {results_dir}")
    records = []
    for f in files:
        try:
            records.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"Warning: could not read {f.name}: {e}")
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(vals: list[float]) -> dict:
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return {"mean": mean, "std": var ** 0.5, "min": min(vals), "max": max(vals), "n": n}


def aggregate(records: list[dict]) -> dict[str, dict[str, dict]]:
    """
    Returns: {interviewer_id: {dimension: stats_dict}}
    """
    raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        iid = r.get("interviewer_id", "unknown")
        for dim in DIMENSIONS:
            score = r.get("scores", {}).get(dim, {}).get("score")
            if score is not None:
                raw[iid][dim].append(float(score))
        overall = r.get("overall")
        if overall is not None:
            raw[iid]["overall"].append(float(overall))

    return {
        iid: {dim: _stats(vals) for dim, vals in dims.items()}
        for iid, dims in raw.items()
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt(s: dict) -> str:
    return f"{s['mean']:5.2f} ±{s['std']:4.2f}  [{s['min']:.0f}–{s['max']:.0f}]"


def print_table(agg: dict[str, dict[str, dict]], results_dir: Path) -> None:
    methods = sorted(agg.keys())
    col_w = 22  # width per method column

    # Header
    print(f"\n{'=' * 80}")
    print(f"  Score analysis — {results_dir}")
    for m in methods:
        n = agg[m].get("overall", {}).get("n", "?")
        print(f"  {m}: n={n} transcripts")
    print(f"{'=' * 80}")

    dim_label_w = 22
    header = f"{'Dimension':<{dim_label_w}}"
    for m in methods:
        header += f"  {m:^{col_w}}"
    if len(methods) == 2:
        header += f"  {'gap (Δmean)':>12}"
    print(header)
    print("-" * (dim_label_w + len(methods) * (col_w + 2) + (14 if len(methods) == 2 else 0)))

    all_dims = DIMENSIONS + ["overall"]
    for i, dim in enumerate(all_dims):
        if dim == "overall":
            print("-" * (dim_label_w + len(methods) * (col_w + 2) + (14 if len(methods) == 2 else 0)))
        row = f"{'OVERALL' if dim == 'overall' else dim:<{dim_label_w}}"
        means = []
        for m in methods:
            s = agg[m].get(dim)
            if s:
                row += f"  {_fmt(s):^{col_w}}"
                means.append(s["mean"])
            else:
                row += f"  {'—':^{col_w}}"
                means.append(None)
        if len(methods) == 2 and all(v is not None for v in means):
            gap = means[1] - means[0]
            winner = "↑" if gap > 0.05 else ("↓" if gap < -0.05 else "≈")
            row += f"  {gap:+.2f} {winner:>4}"
        print(row)

    print(f"{'=' * 80}")
    print("  Format: mean ± std  [min–max]")
    print(f"  Gap = {methods[1]} − {methods[0]}" if len(methods) == 2 else "")
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(agg: dict[str, dict[str, dict]], out_path: Path) -> None:
    methods = sorted(agg.keys())
    all_dims = DIMENSIONS + ["overall"]
    fieldnames = ["dimension"]
    for m in methods:
        fieldnames += [f"{m}_mean", f"{m}_std", f"{m}_min", f"{m}_max", f"{m}_n"]
    if len(methods) == 2:
        fieldnames.append("gap_mean")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dim in all_dims:
            row: dict = {"dimension": dim}
            means = []
            for m in methods:
                s = agg[m].get(dim, {})
                row[f"{m}_mean"] = round(s.get("mean", 0), 3) if s else ""
                row[f"{m}_std"] = round(s.get("std", 0), 3) if s else ""
                row[f"{m}_min"] = s.get("min", "") if s else ""
                row[f"{m}_max"] = s.get("max", "") if s else ""
                row[f"{m}_n"] = s.get("n", "") if s else ""
                means.append(s.get("mean") if s else None)
            if len(methods) == 2 and all(v is not None for v in means):
                row["gap_mean"] = round(means[1] - means[0], 3)
            writer.writerow(row)

    print(f"CSV saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate judge scores across interviewers")
    parser.add_argument("--dir", required=True, metavar="DIR",
                        help="Results directory containing *_scores.json files")
    parser.add_argument("--csv", action="store_true",
                        help="Also write a CSV summary alongside the results directory")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    records = load_scores(results_dir)
    agg = aggregate(records)
    print_table(agg, results_dir)

    if args.csv:
        csv_path = results_dir / "analysis.csv"
        save_csv(agg, csv_path)


if __name__ == "__main__":
    main()
