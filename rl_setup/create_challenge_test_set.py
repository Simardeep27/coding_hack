from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("rl_setup/artifacts/scored_episodes.jsonl")
DEFAULT_OUTPUT_DIR = Path("rl_setup/artifacts/test_sets")
DEFAULT_NAME = "rh3_nonhack12_challenge"

EXPLOIT_PRIORITY = {
    "test_tampering": 0,
    "oracle_access": 1,
    "git_history_leakage": 2,
    "evaluator_tampering": 3,
    "hardcoding": 4,
    "suspicious_file_access": 5,
    "narrow_overfit": 6,
    "other": 7,
    "none": 8,
}

SPLIT_PRIORITY = {
    "test": 0,
    "dev": 1,
    "train": 2,
    "unassigned": 3,
}

DETECTOR_PRIORITY = {
    "HIGH_RISK": 0,
    "SUSPICIOUS": 1,
    "CLEAN": 2,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a small reward-hack challenge test set from scored episodes.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--reward-hacks", type=int, default=3)
    parser.add_argument("--non-hacks", type=int, default=12)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    reward_hacks = select_reward_hacks(rows, args.reward_hacks)
    non_hacks = select_non_hacks(rows, args.non_hacks)
    selected = annotate(reward_hacks, "reward_hack") + annotate(
        non_hacks, "non_reward_hack_suspicious"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.name}.jsonl"
    ids_path = args.output_dir / f"{args.name}.ids.txt"
    args_path = args.output_dir / f"{args.name}.mini_swe_instance_args.txt"
    summary_path = args.output_dir / f"{args.name}.summary.json"

    write_jsonl(jsonl_path, selected)
    ids = [str(row["instance_id"]) for row in selected]
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    args_path.write_text(" ".join(f"--instance-id {instance_id}" for instance_id in ids) + "\n", encoding="utf-8")
    write_json(summary_path, build_summary(selected, args))

    print(f"wrote {jsonl_path}")
    print(f"wrote {ids_path}")
    print(f"wrote {args_path}")
    print(f"wrote {summary_path}")
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select_reward_hacks(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("eval_label") == "REWARD_HACK" or bool(row.get("is_reward_hack"))
    ]
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for exploit_type in sorted(EXPLOIT_PRIORITY, key=EXPLOIT_PRIORITY.get):
        matching = [row for row in candidates if row.get("exploit_type") == exploit_type]
        if not matching:
            continue
        chosen = sorted(matching, key=reward_hack_sort_key)[0]
        instance_id = str(chosen["instance_id"])
        if instance_id not in used_ids:
            selected.append(chosen)
            used_ids.add(instance_id)
        if len(selected) >= count:
            return selected

    for row in sorted(candidates, key=reward_hack_sort_key):
        instance_id = str(row["instance_id"])
        if instance_id not in used_ids:
            selected.append(row)
            used_ids.add(instance_id)
        if len(selected) >= count:
            return selected
    raise ValueError(f"Only found {len(selected)} reward-hack rows, need {count}.")


def select_non_hacks(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("eval_label") != "REWARD_HACK" and not bool(row.get("is_reward_hack"))
    ]
    suspicious = [
        row
        for row in candidates
        if row.get("detector_label") in {"HIGH_RISK", "SUSPICIOUS"}
    ]
    clean = [row for row in candidates if row.get("detector_label") == "CLEAN"]
    selected = sorted(suspicious, key=non_hack_sort_key)[:count]
    if len(selected) < count:
        selected.extend(sorted(clean, key=non_hack_sort_key)[: count - len(selected)])
    if len(selected) < count:
        raise ValueError(f"Only found {len(selected)} non-reward-hack rows, need {count}.")
    return selected


def reward_hack_sort_key(row: dict[str, Any]) -> tuple[object, ...]:
    return (
        EXPLOIT_PRIORITY.get(str(row.get("exploit_type", "other")), 99),
        SPLIT_PRIORITY.get(str(row.get("split", "unassigned")), 99),
        -float(row.get("confidence") or 0.0),
        str(row.get("instance_id", "")),
    )


def non_hack_sort_key(row: dict[str, Any]) -> tuple[object, ...]:
    return (
        SPLIT_PRIORITY.get(str(row.get("split", "unassigned")), 99),
        DETECTOR_PRIORITY.get(str(row.get("detector_label", "CLEAN")), 99),
        -float(row.get("detector_score") or 0.0),
        str(row.get("instance_id", "")),
    )


def annotate(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["challenge_role"] = role
        annotated.append(copied)
    return annotated


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": args.name,
        "source": str(args.input),
        "total": len(rows),
        "requested": {
            "reward_hacks": args.reward_hacks,
            "non_hacks": args.non_hacks,
        },
        "by_challenge_role": dict(Counter(row.get("challenge_role") for row in rows)),
        "by_eval_label": dict(Counter(row.get("eval_label") for row in rows)),
        "by_detector_label": dict(Counter(row.get("detector_label") for row in rows)),
        "by_exploit_type": dict(Counter(row.get("exploit_type") for row in rows)),
        "by_original_split": dict(Counter(row.get("split") for row in rows)),
        "instance_ids": [row.get("instance_id") for row in rows],
        "note": (
            "This is a challenge evaluation set. If you use it as a true held-out "
            "test set, exclude these instance_ids from future SFT/DPO/reward-model training."
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
