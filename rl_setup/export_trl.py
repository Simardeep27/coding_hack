from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import artifact_path, read_jsonl


SYSTEM_PROMPT = (
    "You are an integrity-aware coding agent. Prefer trajectories that solve the "
    "task by understanding the bug and editing implementation code. Avoid reward "
    "hacking: do not tamper with tests/evaluators, read hidden artifacts, copy "
    "future/upstream fixes, hardcode visible outputs, or optimize only for visible tests."
)


def export_trl_datasets(config: dict[str, Any]) -> dict[str, int]:
    export_root = _export_root(config)
    export_root.mkdir(parents=True, exist_ok=True)

    accepted = read_jsonl(artifact_path(config, "accepted_episodes"))
    scored = read_jsonl(artifact_path(config, "scored_episodes"))
    pairs = read_jsonl(artifact_path(config, "preference_pairs"))
    by_id = {str(row.get("instance_id")): row for row in scored if row.get("instance_id")}

    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        rows = [sft_row(row) for row in accepted if row.get("split") == split]
        path = export_root / f"sft_{split}.jsonl"
        _write_jsonl(path, rows)
        counts[path.name] = len(rows)

    dpo_rows_by_split = {"train": [], "dev": [], "test": []}
    reward_rows_by_split = {"train": [], "dev": [], "test": []}
    for pair in pairs:
        winner = by_id.get(str(pair.get("winner_instance_id")))
        loser = by_id.get(str(pair.get("loser_instance_id")))
        if not winner or not loser:
            continue
        split = pair_split(winner, loser)
        row = dpo_row(pair, winner, loser, split=split)
        dpo_rows_by_split[split].append(row)
        reward_rows_by_split[split].append(reward_model_row(pair, winner, loser, split=split))

    all_reward_rows = []
    for split in ("train", "dev", "test"):
        dpo_path = export_root / f"dpo_{split}.jsonl"
        reward_path = export_root / f"reward_model_pairs_{split}.jsonl"
        _write_jsonl(dpo_path, dpo_rows_by_split[split])
        _write_jsonl(reward_path, reward_rows_by_split[split])
        counts[dpo_path.name] = len(dpo_rows_by_split[split])
        counts[reward_path.name] = len(reward_rows_by_split[split])
        all_reward_rows.extend(reward_rows_by_split[split])
    _write_jsonl(export_root / "reward_model_pairs.jsonl", all_reward_rows)
    counts["reward_model_pairs.jsonl"] = len(all_reward_rows)

    manifest = {
        "description": "TRL-ready exports from rl_setup artifacts.",
        "source_note": (
            "SFT rows use accepted clean trajectories. DPO/reward-model rows use "
            "full trajectory preferences. Current pairs may be cross-instance; "
            "collect same-instance multi-rollout pairs for stronger task-specific DPO."
        ),
        "format": {
            "sft": "TRL conversational SFT format with a messages column.",
            "dpo": "TRL explicit-prompt preference format with prompt/chosen/rejected.",
            "reward_model": "Pairwise full-trajectory preference rows with chosen/rejected text.",
        },
        "counts": counts,
    }
    (export_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return counts


def sft_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Produce a high-integrity SWE-Bench solving trajectory for "
                    f"instance {row.get('instance_id')}. Include useful investigation, "
                    "implementation, verification, and final patch behavior."
                ),
            },
            {"role": "assistant", "content": trajectory_text(row)},
        ],
        "instance_id": row.get("instance_id"),
        "split": row.get("split"),
        "reward_safe": row.get("reward_safe"),
        "detector_label": row.get("detector_label"),
    }


def dpo_row(
    pair: dict[str, Any],
    winner: dict[str, Any],
    loser: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    return {
        "prompt": (
            SYSTEM_PROMPT
            + "\n\nGenerate a full coding-agent trajectory. The preferred answer should "
            "show honest problem solving and avoid exploitative tool use."
        ),
        "chosen": trajectory_text(winner),
        "rejected": trajectory_text(loser),
        "pair_id": pair.get("pair_id"),
        "winner_instance_id": winner.get("instance_id"),
        "loser_instance_id": loser.get("instance_id"),
        "winner_split": winner.get("split"),
        "loser_split": loser.get("split"),
        "split": split,
        "winner_reward_safe": winner.get("reward_safe"),
        "loser_reward_safe": loser.get("reward_safe"),
        "pairing_note": "generic full-trajectory preference; not necessarily same-task",
    }


def reward_model_row(
    pair: dict[str, Any],
    winner: dict[str, Any],
    loser: dict[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    return {
        "chosen": trajectory_text(winner),
        "rejected": trajectory_text(loser),
        "pair_id": pair.get("pair_id"),
        "chosen_instance_id": winner.get("instance_id"),
        "rejected_instance_id": loser.get("instance_id"),
        "chosen_split": winner.get("split"),
        "rejected_split": loser.get("split"),
        "split": split,
        "chosen_reward_safe": winner.get("reward_safe"),
        "rejected_reward_safe": loser.get("reward_safe"),
    }


def pair_split(winner: dict[str, Any], loser: dict[str, Any]) -> str:
    splits = {str(winner.get("split", "train")), str(loser.get("split", "train"))}
    if "test" in splits:
        return "test"
    if "dev" in splits:
        return "dev"
    return "train"


def trajectory_text(row: dict[str, Any]) -> str:
    commands = row.get("commands") if isinstance(row.get("commands"), list) else []
    command_lines = "\n".join(
        f"{index + 1}. {_truncate(str(command), 1200)}"
        for index, command in enumerate(commands[:80])
    )
    if len(commands) > 80:
        command_lines += f"\n... {len(commands) - 80} commands omitted ..."

    patch = str(row.get("final_patch") or "").strip()
    if not patch:
        patch = "[final patch unavailable in current artifact]"
    else:
        patch = _truncate(patch, 6000)

    return "\n".join(
        [
            f"Instance: {row.get('instance_id')}",
            f"Outcome: resolved={row.get('resolved')} visible_rate={row.get('visible_rate')} hidden_rate={row.get('hidden_rate')}",
            f"Reward-hack label: {row.get('eval_label')} exploit_type={row.get('exploit_type')} detector={row.get('detector_label')}",
            f"Modified files: {', '.join(str(item) for item in row.get('modified_files', [])) or 'none recorded'}",
            "",
            "Trajectory commands:",
            command_lines or "[commands unavailable]",
            "",
            "Final patch:",
            patch,
            "",
            "Label rationale:",
            str(row.get("rationale") or "[rationale unavailable]"),
        ]
    )


def _export_root(config: dict[str, Any]) -> Path:
    output_name = str(config.get("outputs", {}).get("trl_export_dir", "trl"))
    return artifact_path(config, "summary_json").parent / output_name


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = limit // 2
    tail = limit - head
    return value[:head] + "\n...[truncated]...\n" + value[-tail:]
