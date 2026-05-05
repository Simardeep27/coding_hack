from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from .io_utils import artifact_path, read_jsonl, write_json


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    scored = read_jsonl(artifact_path(config, "scored_episodes"))
    accepted = read_jsonl(artifact_path(config, "accepted_episodes"))
    rejected = read_jsonl(artifact_path(config, "rejected_episodes"))
    preferences = read_jsonl(artifact_path(config, "preference_pairs"))

    report = {
        "episodes": summarize_rows(scored),
        "accepted": summarize_rows(accepted),
        "rejected": summarize_rows(rejected),
        "preference_pairs": len(preferences),
        "artifact_paths": {
            "scored_episodes": str(artifact_path(config, "scored_episodes")),
            "accepted_episodes": str(artifact_path(config, "accepted_episodes")),
            "rejected_episodes": str(artifact_path(config, "rejected_episodes")),
            "preference_pairs": str(artifact_path(config, "preference_pairs")),
        },
    }
    write_json(artifact_path(config, "summary_json"), report)
    artifact_path(config, "summary_md").write_text(render_markdown(report), encoding="utf-8")
    return report


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row.get("eval_label", "UNKNOWN")) for row in rows)
    exploits = Counter(str(row.get("exploit_type", "none")) for row in rows)
    detector = Counter(str(row.get("detector_label", "UNKNOWN")) for row in rows)
    resolved = [row for row in rows if row.get("resolved") is True]
    reward_hacks = [row for row in rows if row.get("is_reward_hack") is True]
    visible_rates = [float(row["visible_rate"]) for row in rows if isinstance(row.get("visible_rate"), (int, float))]
    hidden_rates = [float(row["hidden_rate"]) for row in rows if isinstance(row.get("hidden_rate"), (int, float))]
    gaps = [float(row["overfit_gap"]) for row in rows if isinstance(row.get("overfit_gap"), (int, float))]
    safe_rewards = [float(row["reward_safe"]) for row in rows if isinstance(row.get("reward_safe"), (int, float))]

    return {
        "total": len(rows),
        "resolved": len(resolved),
        "reward_hacks": len(reward_hacks),
        "reward_hack_rate": round(len(reward_hacks) / len(rows), 4) if rows else 0.0,
        "label_counts": dict(sorted(labels.items())),
        "exploit_type_counts": dict(sorted(exploits.items())),
        "detector_counts": dict(sorted(detector.items())),
        "avg_visible_rate": _avg(visible_rates),
        "avg_hidden_rate": _avg(hidden_rates),
        "avg_overfit_gap": _avg(gaps),
        "max_overfit_gap": max(gaps) if gaps else None,
        "avg_safe_reward": _avg(safe_rewards),
    }


def render_markdown(report: dict[str, Any]) -> str:
    episodes = report["episodes"]
    accepted = report["accepted"]
    rejected = report["rejected"]
    return "\n".join(
        [
            "# API-First RL Setup Report",
            "",
            "## Episodes",
            f"- Total: {episodes['total']}",
            f"- Resolved: {episodes['resolved']}",
            f"- Reward hacks: {episodes['reward_hacks']} ({episodes['reward_hack_rate']})",
            f"- Label counts: `{episodes['label_counts']}`",
            f"- Exploit counts: `{episodes['exploit_type_counts']}`",
            f"- Detector counts: `{episodes['detector_counts']}`",
            f"- Avg visible rate: {episodes['avg_visible_rate']}",
            f"- Avg hidden rate: {episodes['avg_hidden_rate']}",
            f"- Max overfit gap: {episodes['max_overfit_gap']}",
            "",
            "## Rejection Sampling",
            f"- Accepted: {accepted['total']}",
            f"- Rejected: {rejected['total']}",
            f"- Accepted reward hacks: {accepted['reward_hacks']}",
            f"- Rejected reward hacks: {rejected['reward_hacks']}",
            f"- Preference pairs: {report['preference_pairs']}",
            "",
        ]
    )


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None

