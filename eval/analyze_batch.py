"""
Analyze batch_scores and batch_coverage files across all protocol directories
and produce formatted comparison tables (per-protocol and per-dimension).

Usage:
    python eval/analyze_batch.py --dir eval/results_new/
    python eval/analyze_batch.py --dir eval/results_new/ --md   # markdown output
    python eval/analyze_batch.py --dir eval/results_new/ --coverage  # coverage tables only
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

DIMENSIONS = [
    "overall",
    "clarity",
    "cognitive_empathy",
    "specificity",
    "palpability",
    "informativeness",
    "follow_up",
    "relevance",
    "self_awareness",
]

DIM_LABELS = {
    "overall": "Overall",
    "clarity": "Clarity",
    "cognitive_empathy": "Cognitive Empathy",
    "specificity": "Specificity",
    "palpability": "Palpability",
    "informativeness": "Informativeness",
    "follow_up": "Follow-up",
    "relevance": "Relevance",
    "self_awareness": "Self-Awareness",
}

COVERAGE_DIMENSIONS = [
    "overall",
    "topic_breadth_coverage",
    "objective_saturation",
    "objective_balance",
]

COVERAGE_LABELS = {
    "overall": "Overall",
    "topic_breadth_coverage": "Topic Breadth Coverage",
    "objective_saturation": "Objective Saturation",
    "objective_balance": "Objective Balance",
}

AGENTIC = "agentic"
SINGLE = "single_llm"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    """Approximate token count using whitespace word count."""
    return len(text.split())


def load_transcript_records(results_dir: Path) -> list[dict]:
    """
    Load raw transcript JSON files (excluding _coverage, _scores, batch_* files).
    Returns list of dicts with keys: protocol_name, interviewer_id, mean_tokens_interviewer, mean_tokens_interviewee.
    """
    records = []
    for path in sorted(results_dir.rglob("*.json")):
        # Skip derived files and batch subdirectory files
        stem = path.stem
        if (stem.endswith("_coverage") or stem.endswith("_scores")
                or stem.startswith("benchmark_") or "batch_" in path.parts[-2]):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "transcript" not in data or not isinstance(data["transcript"], list):
            continue
        interviewer_words = [
            _count_words(t["content"])
            for t in data["transcript"]
            if t.get("role") == "interviewer" and t.get("content")
        ]
        interviewee_words = [
            _count_words(t["content"])
            for t in data["transcript"]
            if t.get("role") == "interviewee" and t.get("content")
        ]
        records.append({
            "protocol_name": data.get("protocol_name", "Unknown"),
            "interviewer_id": data.get("interviewer_id", "unknown"),
            "mean_tokens_interviewer": sum(interviewer_words) / len(interviewer_words) if interviewer_words else None,
            "mean_tokens_interviewee": sum(interviewee_words) / len(interviewee_words) if interviewee_words else None,
        })
    if not records:
        raise FileNotFoundError(f"No raw transcript JSON files found under {results_dir}")
    return records


def aggregate_tokens(records: list[dict], role_key: str) -> dict:
    """
    Returns {protocol_name: {interviewer_id: [mean_tokens_per_transcript]}}.
    role_key is 'mean_tokens_interviewer' or 'mean_tokens_interviewee'.
    """
    data: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        v = r.get(role_key)
        if v is not None:
            data[r["protocol_name"]][r["interviewer_id"]].append(v)
    return data


def _load_records_from_glob(results_dir: Path, subfolder: str) -> list[dict]:
    records = []
    for batch_file in sorted(results_dir.rglob(f"{subfolder}/*.json")):
        data = json.loads(batch_file.read_text())
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)
    return records


def load_all_records(results_dir: Path) -> list[dict]:
    records = _load_records_from_glob(results_dir, "batch_scores")
    if not records:
        raise FileNotFoundError(f"No batch_scores/*.json files found under {results_dir}")
    return records


def load_coverage_records(results_dir: Path) -> list[dict]:
    records = _load_records_from_glob(results_dir, "batch_coverage")
    if not records:
        raise FileNotFoundError(f"No batch_coverage/*.json files found under {results_dir}")
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {}
    mean = sum(vals) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n}


def welch_t_pvalue(a: list[float], b: list[float]) -> float | None:
    """Two-sided Welch's t-test p-value using a t-distribution approximation."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    mean_a = sum(a) / na
    mean_b = sum(b) / nb
    var_a = sum((x - mean_a) ** 2 for x in a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (nb - 1)
    se2 = var_a / na + var_b / nb
    if se2 == 0:
        return None
    t = (mean_a - mean_b) / math.sqrt(se2)
    # Welch–Satterthwaite degrees of freedom
    df = se2 ** 2 / ((var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1))
    # p-value via regularised incomplete beta (two-sided)
    p = _t_pvalue(t, df)
    return p


def _t_pvalue(t: float, df: float) -> float:
    """Two-sided p-value from t-statistic and degrees of freedom."""
    x = df / (df + t * t)
    p_one_tail = 0.5 * _betainc(df / 2, 0.5, x)
    return 2 * p_one_tail


def _betainc(a: float, b: float, x: float, max_iter: int = 200, tol: float = 1e-10) -> float:
    """Regularised incomplete beta function I_x(a, b) via continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / d if abs(d) < 1e-30 else 1.0 / d
    f = d
    for m in range(1, max_iter + 1):
        # even step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        c = 1.0 + num / c
        d = 1.0 / (d if abs(d) >= 1e-30 else 1e-30)
        c = c if abs(c) >= 1e-30 else 1e-30
        f *= c * d
        # odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        c = 1.0 + num / c
        d = 1.0 / (d if abs(d) >= 1e-30 else 1e-30)
        c = c if abs(c) >= 1e-30 else 1e-30
        delta = c * d
        f *= delta
        if abs(delta - 1) < tol:
            break
    return front * f


def sig_stars(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "~"
    return ""


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def get_dim_score(record: dict, dim: str) -> float | None:
    if dim == "overall":
        v = record.get("overall")
        return float(v) if v is not None else None
    v = record.get("scores", {}).get(dim, {}).get("score")
    return float(v) if v is not None else None


def aggregate_by_protocol_and_interviewer(records: list[dict], dimensions: list[str]) -> dict:
    """
    Returns:
      {
        protocol_name: {
          interviewer_id: {dimension: [scores]}
        }
      }
    """
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        proto = r.get("protocol_name", "Unknown")
        iid = r.get("interviewer_id", "unknown")
        for dim in dimensions:
            v = get_dim_score(r, dim)
            if v is not None:
                data[proto][iid][dim].append(v)
    return data


# ---------------------------------------------------------------------------
# Table 1: Per-protocol overall scores
# ---------------------------------------------------------------------------

def table_per_protocol(data: dict, markdown: bool = False) -> str:
    protocols = sorted(data.keys())
    rows = []
    for proto in protocols:
        ag = _stats(data[proto].get(AGENTIC, {}).get("overall", []))
        sl = _stats(data[proto].get(SINGLE, {}).get("overall", []))
        if not ag or not sl:
            continue
        gap = ag["mean"] - sl["mean"]
        winner = f"Agentic (+{gap:.2f})" if gap > 0 else f"Single LLM (+{-gap:.2f})"
        rows.append((proto, ag, sl, winner))

    if markdown:
        lines = []
        lines.append("| Protocol | Agentic | Single LLM | Gap |")
        lines.append("|---|---|---|---|")
        for proto, ag, sl, winner in rows:
            lines.append(
                f"| {proto} "
                f"| {ag['mean']:.2f} ± {ag['std']:.2f} "
                f"| {sl['mean']:.2f} ± {sl['std']:.2f} "
                f"| {winner} |"
            )
        lines.append("")
        lines.append("*Table: Mean Overall Scores (1–10) across Protocols*")
        return "\n".join(lines)
    else:
        col_proto = max(len(p) for p, *_ in rows) + 2
        header = f"{'Protocol':<{col_proto}}  {'Agentic':^18}  {'Single LLM':^18}  Gap"
        sep = "-" * (col_proto + 2 + 18 + 2 + 18 + 2 + 30)
        out = [header, sep]
        for proto, ag, sl, winner in rows:
            out.append(
                f"{proto:<{col_proto}}  "
                f"{ag['mean']:.2f} ± {ag['std']:.2f}  "
                f"  {sl['mean']:.2f} ± {sl['std']:.2f}  "
                f"  {winner}"
            )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Table 2: Per-dimension comparison (all protocols pooled)
# ---------------------------------------------------------------------------

def table_per_dimension(
    data: dict,
    dimensions: list[str],
    labels: dict[str, str],
    title: str = "Mean Scores by Dimension (1–10), pooled across all protocols",
    markdown: bool = False,
) -> str:
    # Pool all scores across protocols
    agentic_scores: dict[str, list[float]] = defaultdict(list)
    single_scores: dict[str, list[float]] = defaultdict(list)
    for proto in data:
        for dim in dimensions:
            agentic_scores[dim].extend(data[proto].get(AGENTIC, {}).get(dim, []))
            single_scores[dim].extend(data[proto].get(SINGLE, {}).get(dim, []))

    rows = []
    for dim in dimensions:
        ag_vals = agentic_scores[dim]
        sl_vals = single_scores[dim]
        ag = _stats(ag_vals)
        sl = _stats(sl_vals)
        if not ag or not sl:
            continue
        gap = ag["mean"] - sl["mean"]
        p = welch_t_pvalue(ag_vals, sl_vals)
        stars = sig_stars(p)
        rows.append((dim, ag, sl, gap, p, stars))

    if markdown:
        lines = []
        lines.append("| Dimension | Agentic | Single LLM | Gap | p | |")
        lines.append("|---|---|---|---|---|---|")
        for dim, ag, sl, gap, p, stars in rows:
            p_str = f"{p:.3f}" if p is not None else "—"
            lines.append(
                f"| {labels[dim]} "
                f"| {ag['mean']:.2f} ± {ag['std']:.2f} "
                f"| {sl['mean']:.2f} ± {sl['std']:.2f} "
                f"| {gap:+.2f} "
                f"| {p_str} "
                f"| {stars} |"
            )
        lines.append("")
        lines.append(f"*Table: {title}*")
        return "\n".join(lines)
    else:
        label_w = max(len(labels[d]) for d in dimensions) + 2
        header = f"{'Dimension':<{label_w}}  {'Agentic':^18}  {'Single LLM':^18}  {'Gap':>6}  {'p':>7}  Sig"
        sep = "-" * (label_w + 2 + 18 + 2 + 18 + 2 + 6 + 2 + 7 + 2 + 3)
        out = [header, sep]
        for dim, ag, sl, gap, p, stars in rows:
            p_str = f"{p:.3f}" if p is not None else "  —  "
            out.append(
                f"{labels[dim]:<{label_w}}  "
                f"{ag['mean']:.2f} ± {ag['std']:.2f}  "
                f"  {sl['mean']:.2f} ± {sl['std']:.2f}  "
                f"  {gap:+.2f}  "
                f"  {p_str}  "
                f"{stars}"
            )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Table 3: Mean tokens per turn (interviewer / interviewee)
# ---------------------------------------------------------------------------

def _mean_var(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return mean, var


def table_tokens(
    token_data: dict,
    role_label: str,
    markdown: bool = False,
) -> str:
    """
    One table for a single role, with a row-pair per protocol (Agentic + Single LLM).
    token_data: {protocol_name: {interviewer_id: [mean_tokens_per_transcript]}}
    """
    protocols = sorted(token_data.keys())
    rows: list[tuple[str, float, float, float, float]] = []
    for proto in protocols:
        ag_mean, ag_var = _mean_var(token_data[proto].get(AGENTIC, []))
        sl_mean, sl_var = _mean_var(token_data[proto].get(SINGLE, []))
        rows.append((proto, ag_mean, ag_var, sl_mean, sl_var))

    if not rows:
        return "(no data)"

    title = f"Mean Tokens/Turn — {role_label}"
    if markdown:
        lines = [
            f"**{title}**\n",
            "| Protocol | Condition | Mean tokens/turn | Variance |",
            "|---|---|---|---|",
        ]
        for proto, ag_mean, ag_var, sl_mean, sl_var in rows:
            lines.append(f"| {proto} | Agentic | {ag_mean:.2f} | {ag_var:.2f} |")
            lines.append(f"| | Single LLM | {sl_mean:.2f} | {sl_var:.2f} |")
        return "\n".join(lines)
    else:
        col_p = max(len(p) for p, *_ in rows) + 2
        col_w = 12
        header = f"{'Protocol':<{col_p}}  {'Condition':<{col_w}}  {'Mean tokens/turn':>18}  {'Variance':>10}"
        sep = "-" * (col_p + 2 + col_w + 2 + 18 + 2 + 10)
        out = [title, header, sep]
        for proto, ag_mean, ag_var, sl_mean, sl_var in rows:
            out.append(f"{proto:<{col_p}}  {'Agentic':<{col_w}}  {ag_mean:>18.2f}  {ag_var:>10.2f}")
            out.append(f"{'':<{col_p}}  {'Single LLM':<{col_w}}  {sl_mean:>18.2f}  {sl_var:>10.2f}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse batch scores and coverage across protocols")
    parser.add_argument("--dir", default="eval/results_new/", metavar="DIR",
                        help="Root results directory (searched recursively for batch_scores/ and batch_coverage/)")
    parser.add_argument("--md", action="store_true", help="Output markdown tables")
    parser.add_argument("--coverage", action="store_true",
                        help="Show coverage tables only (skip quality scores)")
    parser.add_argument("--scores", action="store_true",
                        help="Show quality score tables only (skip coverage)")
    args = parser.parse_args()

    results_dir = Path(args.dir)
    show_scores = not args.coverage  # default: show scores unless --coverage only
    show_coverage = not args.scores  # default: show coverage unless --scores only

    # --- Tokens per turn ---
    try:
        transcript_records = load_transcript_records(results_dir)
        print(f"Loaded {len(transcript_records)} transcript records for token analysis\n")
        iwr_data = aggregate_tokens(transcript_records, "mean_tokens_interviewer")
        iwe_data = aggregate_tokens(transcript_records, "mean_tokens_interviewee")

        print("=== Mean Tokens per Turn — Interviewer ===\n")
        print(table_tokens(iwr_data, "Interviewer", markdown=args.md))
        print()
        print("=== Mean Tokens per Turn — Interviewee ===\n")
        print(table_tokens(iwe_data, "Interviewee", markdown=args.md))
        print()
    except FileNotFoundError as e:
        print(f"[tokens] {e}\n")

    # --- Quality scores ---
    if show_scores:
        try:
            records = load_all_records(results_dir)
            print(f"Loaded {len(records)} quality-score records\n")
            data = aggregate_by_protocol_and_interviewer(records, DIMENSIONS)

            print("=== Per-Protocol Overall Quality Scores ===\n")
            print(table_per_protocol(data, markdown=args.md))
            print()
            print("=== Per-Dimension Quality Scores (pooled across protocols) ===\n")
            print(table_per_dimension(
                data, DIMENSIONS, DIM_LABELS,
                title="Mean Quality Scores by Dimension (1–10), pooled across all protocols",
                markdown=args.md,
            ))
            print()
        except FileNotFoundError as e:
            print(f"[quality scores] {e}\n")

    # --- Coverage scores ---
    if show_coverage:
        try:
            cov_records = load_coverage_records(results_dir)
            print(f"Loaded {len(cov_records)} coverage-score records\n")
            cov_data = aggregate_by_protocol_and_interviewer(cov_records, COVERAGE_DIMENSIONS)

            print("=== Per-Protocol Overall Coverage Scores ===\n")
            print(table_per_protocol(cov_data, markdown=args.md))
            print()
            print("=== Per-Dimension Coverage Scores (pooled across protocols) ===\n")
            print(table_per_dimension(
                cov_data, COVERAGE_DIMENSIONS, COVERAGE_LABELS,
                title="Mean Coverage Scores by Dimension (1–10), pooled across all protocols",
                markdown=args.md,
            ))
            print()
        except FileNotFoundError as e:
            print(f"[coverage] {e}\n")


if __name__ == "__main__":
    main()
