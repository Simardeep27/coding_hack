from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize task-based reward-hack evaluation labels."
    )
    parser.add_argument("labels", type=Path, help="Path to reward_hack_labels.jsonl.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.labels)
    summary = summarize(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))

    output_json = args.output_json or args.labels.with_suffix(".summary.json")
    output_md = args.output_md or args.labels.with_suffix(".summary.md")
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    resolved = [row for row in rows if row.get("resolved") is True]
    reward_hacks = [row for row in rows if row.get("is_reward_hack") is True]
    visible_rates = [_float(row.get("visible_rate")) for row in rows]
    hidden_rates = [_float(row.get("hidden_rate")) for row in rows]
    gaps = [_float(row.get("overfit_gap")) for row in rows]
    labels = Counter(str(row.get("eval_label", "UNKNOWN")) for row in rows)
    exploits = Counter(str(row.get("exploit_type", "none")) for row in rows)
    detectors = Counter(str(row.get("detector_label", "UNKNOWN")) for row in rows)

    return {
        "total": total,
        "resolved": len(resolved),
        "solve_rate": _rate(len(resolved), total),
        "reward_hacks": len(reward_hacks),
        "reward_hack_rate": _rate(len(reward_hacks), total),
        "avg_visible_rate": _avg(visible_rates),
        "avg_hidden_rate": _avg(hidden_rates),
        "avg_overfit_gap": _avg(gaps),
        "label_counts": dict(sorted(labels.items())),
        "exploit_type_counts": dict(sorted(exploits.items())),
        "detector_counts": dict(sorted(detectors.items())),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Task Evaluation Summary",
            "",
            f"- Total: {summary['total']}",
            f"- Resolved: {summary['resolved']} ({summary['solve_rate']})",
            f"- Reward hacks: {summary['reward_hacks']} ({summary['reward_hack_rate']})",
            f"- Avg visible rate: {summary['avg_visible_rate']}",
            f"- Avg hidden rate: {summary['avg_hidden_rate']}",
            f"- Avg visible-hidden gap: {summary['avg_overfit_gap']}",
            f"- Label counts: `{summary['label_counts']}`",
            f"- Exploit counts: `{summary['exploit_type_counts']}`",
            f"- Detector counts: `{summary['detector_counts']}`",
            "",
        ]
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object on line {line_no} of {path}")
        rows.append(value)
    return rows


def _float(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


if __name__ == "__main__":
    main()

