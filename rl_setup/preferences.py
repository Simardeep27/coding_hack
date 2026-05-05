from __future__ import annotations

from typing import Any

from .io_utils import artifact_path, read_jsonl, write_jsonl


def build_preferences(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(artifact_path(config, "scored_episodes"))
    pref_config = config.get("preferences", {})
    max_pairs = int(pref_config.get("max_pairs", 200))
    winner_min_reward = float(pref_config.get("winner_min_reward", 0.5))
    loser_max_reward = float(pref_config.get("loser_max_reward", 0.25))

    winners = [
        row
        for row in rows
        if row.get("reward_safe", 0.0) >= winner_min_reward
        and not row.get("is_reward_hack", False)
        and row.get("detector_label") == "CLEAN"
    ]
    losers = [
        row
        for row in rows
        if row.get("is_reward_hack", False)
        or row.get("detector_label") in {"SUSPICIOUS", "HIGH_RISK"}
        or row.get("reward_safe", 0.0) <= loser_max_reward
    ]

    winners = sorted(winners, key=lambda row: (-float(row.get("reward_safe", 0.0)), row["instance_id"]))
    losers = sorted(losers, key=lambda row: (float(row.get("reward_safe", 0.0)), row["instance_id"]))

    pairs: list[dict[str, Any]] = []
    if not winners or not losers:
        write_jsonl(artifact_path(config, "preference_pairs"), pairs)
        return pairs

    for index in range(min(max_pairs, len(winners) * len(losers))):
        winner = winners[index % len(winners)]
        loser = losers[index % len(losers)]
        if winner["instance_id"] == loser["instance_id"]:
            continue
        pairs.append(
            {
                "pair_id": f"pref-{len(pairs) + 1:05d}",
                "winner_instance_id": winner["instance_id"],
                "loser_instance_id": loser["instance_id"],
                "winner_split": winner.get("split"),
                "loser_split": loser.get("split"),
                "winner_reward_safe": winner.get("reward_safe"),
                "loser_reward_safe": loser.get("reward_safe"),
                "winner_summary": compact_summary(winner),
                "loser_summary": compact_summary(loser),
                "reason": "clean_higher_safe_reward_preferred_over_hacked_or_low_reward",
            }
        )

    write_jsonl(artifact_path(config, "preference_pairs"), pairs)
    return pairs


def compact_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_label": row.get("eval_label"),
        "resolved": row.get("resolved"),
        "visible_rate": row.get("visible_rate"),
        "hidden_rate": row.get("hidden_rate"),
        "overfit_gap": row.get("overfit_gap"),
        "exploit_type": row.get("exploit_type"),
        "detector_label": row.get("detector_label"),
        "detector_score": row.get("detector_score"),
        "reward_safe": row.get("reward_safe"),
    }

