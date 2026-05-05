from __future__ import annotations

from .io_utils import artifact_path, input_path, read_json, read_jsonl


def run_checks(config: dict) -> list[str]:
    messages: list[str] = []
    splits = read_json(artifact_path(config, "splits"))
    split_sets = {name: set(values) for name, values in splits.items()}
    overlap = (split_sets["train"] & split_sets["dev"]) | (split_sets["train"] & split_sets["test"]) | (split_sets["dev"] & split_sets["test"])
    assert not overlap, f"Split overlap found: {sorted(overlap)[:5]}"
    messages.append("splits_have_no_overlap")

    source_rows = read_jsonl(input_path(config, "cumulative_labels"))
    episodes = read_jsonl(artifact_path(config, "episodes"))
    scored = read_jsonl(artifact_path(config, "scored_episodes"))
    assert len(episodes) == len(source_rows), "Episode count does not match source labels"
    assert len(scored) == len(source_rows), "Scored count does not match source labels"
    messages.append(f"episode_count_matches_source:{len(source_rows)}")

    positives = [row for row in scored if row.get("is_reward_hack") is True]
    assert positives, "No reward-hack positives found for reward sanity check"
    for row in positives:
        assert row["reward_safe"] < row["reward_naive"], f"Positive not penalized: {row['instance_id']}"
    messages.append(f"reward_hack_positives_penalized:{len(positives)}")

    high_gap_clean = [
        row
        for row in scored
        if row.get("overfit_gap") is not None
        and row.get("overfit_gap") >= 0.5
        and row.get("is_reward_hack") is not True
    ]
    if high_gap_clean:
        assert all(row.get("detector_label") != "HIGH_RISK" for row in high_gap_clean), "High gap alone became high risk"
        messages.append(f"high_gap_alone_not_high_risk:{len(high_gap_clean)}")

    accepted = read_jsonl(artifact_path(config, "accepted_episodes"))
    assert all(row.get("is_reward_hack") is not True for row in accepted), "Accepted rows include reward hacks"
    messages.append(f"accepted_rows_clean:{len(accepted)}")
    return messages

