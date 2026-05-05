from __future__ import annotations

from collections import defaultdict
from typing import Any

from .io_utils import artifact_path, read_jsonl, write_jsonl


def run_rejection_sampling(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(artifact_path(config, "scored_episodes"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("instance_id", ""))].append(row)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for instance_id, candidates in sorted(grouped.items()):
        ranked = sorted(
            candidates,
            key=lambda row: (
                float(row.get("reward_safe", 0.0)),
                -float(row.get("detector_score", 0.0)),
            ),
            reverse=True,
        )
        accepted_one = False
        for row in ranked:
            reason = rejection_reason(row, config)
            if reason is None and not accepted_one:
                accepted_row = dict(row)
                accepted_row["selection_status"] = "accepted"
                accepted_row["selection_reason"] = "highest_safe_reward_clean_candidate"
                accepted.append(accepted_row)
                accepted_one = True
            else:
                rejected_row = dict(row)
                rejected_row["selection_status"] = "rejected"
                rejected_row["selection_reason"] = reason or "lower_ranked_duplicate_candidate"
                rejected.append(rejected_row)

        if not accepted_one and instance_id:
            for row in ranked:
                if row.get("selection_status") == "rejected":
                    continue

    write_jsonl(artifact_path(config, "accepted_episodes"), accepted)
    write_jsonl(artifact_path(config, "rejected_episodes"), rejected)
    return accepted, rejected


def rejection_reason(row: dict[str, Any], config: dict[str, Any]) -> str | None:
    rejection_config = config.get("rejection_sampling", {})
    min_safe_reward = float(rejection_config.get("min_safe_reward", 0.0))
    max_detector_score = float(rejection_config.get("max_detector_score", 0.65))
    if bool(row.get("is_reward_hack")):
        return "labeled_reward_hack"
    if float(row.get("detector_score", 0.0)) > max_detector_score:
        return "detector_score_above_threshold"
    if float(row.get("reward_safe", 0.0)) < min_safe_reward:
        return "safe_reward_below_threshold"
    return None

