#!/usr/bin/env python3
"""
Merge multiple labeled trajectory JSONL files into a single deduplicated file.

The reward-hack labeler can produce overlapping outputs across batches
(trajectories.jsonl and trajectories2.jsonl in this repo). This step is the
single source of truth for "all trajectories we have ever labeled".

Dedup rules:
    - Primary key:  instance_id
    - When two rows share an instance_id, the one with the longer commands
      list wins, breaking ties by higher confidence and then file order.
    - A SHA256 of the command sequence is used to detect *disagreeing*
      duplicates (same instance_id, different command list). These are
      logged as warnings but kept (winner only) so we never silently drop
      a useful row.

Usage:
    python -m data.merge_trajectories \
        --inputs data/trajectories.jsonl data/trajectories2.jsonl \
        --output data/merged_trajectories.jsonl

    # Or, drive everything from the central config:
    python -m data.merge_trajectories --config config/sft_config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _command_hash(commands: list[Any]) -> str:
    payload = json.dumps([str(c) for c in (commands or [])], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_quality_key(row: dict) -> tuple:
    # Bigger commands list wins; ties broken by confidence; resolved beats
    # not-resolved if everything else ties.
    return (
        len(row.get("commands") or []),
        float(row.get("confidence") or 0.0),
        1 if row.get("resolved") else 0,
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[WARN] Skipping missing input: {path}", file=sys.stderr)
        return []
    rows = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{line_num} bad JSON: {e}", file=sys.stderr)
    return rows


def merge(inputs: list[Path]) -> tuple[list[dict], dict]:
    by_id: dict[str, dict] = {}
    sources: dict[str, str] = {}
    cmd_hashes: dict[str, str] = {}
    disagreements: list[str] = []
    per_input_counts: Counter[str] = Counter()

    for path in inputs:
        rows = load_jsonl(path)
        per_input_counts[str(path)] = len(rows)
        for row in rows:
            iid = row.get("instance_id")
            if not iid:
                continue
            current_hash = _command_hash(row.get("commands") or [])
            if iid not in by_id:
                by_id[iid] = row
                sources[iid] = str(path)
                cmd_hashes[iid] = current_hash
                continue
            # Conflict: keep the higher-quality row.
            if cmd_hashes[iid] != current_hash:
                disagreements.append(iid)
            if _row_quality_key(row) > _row_quality_key(by_id[iid]):
                by_id[iid] = row
                sources[iid] = str(path)
                cmd_hashes[iid] = current_hash

    merged = list(by_id.values())
    stats = {
        "per_input_counts": dict(per_input_counts),
        "merged_count": len(merged),
        "disagreement_count": len(disagreements),
        "disagreement_ids": disagreements[:20],
    }
    return merged, stats


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/sft_config.yaml",
        help="Centralized YAML config (provides defaults for --inputs/--output).",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help="Override list of input JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override merged output path.",
    )
    args = parser.parse_args()

    config = load_config(args.config) if Path(args.config).exists() else {}
    data_cfg = config.get("data", {})

    inputs = args.inputs or [Path(p) for p in data_cfg.get("trajectory_inputs", [])]
    if not inputs:
        print(
            "ERROR: No inputs provided. Set data.trajectory_inputs in config "
            "or pass --inputs.",
            file=sys.stderr,
        )
        sys.exit(2)

    output = args.output or Path(
        data_cfg.get("merged_path", "data/merged_trajectories.jsonl")
    )

    print(f"[INFO] Inputs : {[str(p) for p in inputs]}")
    print(f"[INFO] Output : {output}")

    merged, stats = merge(inputs)
    write_jsonl(merged, output)

    print("\n=== MERGE SUMMARY ===")
    print(f"  per-input counts : {stats['per_input_counts']}")
    print(f"  merged unique    : {stats['merged_count']}")
    print(f"  command-hash disagreements: {stats['disagreement_count']}")
    if stats["disagreement_count"]:
        print("  sample disagreement ids (winners written):")
        for iid in stats["disagreement_ids"]:
            print(f"    - {iid}")
    print(f"\n[INFO] Wrote {len(merged)} merged rows to {output}")


if __name__ == "__main__":
    main()