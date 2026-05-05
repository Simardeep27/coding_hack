from __future__ import annotations

import glob
import hashlib
from pathlib import Path
from typing import Any

from .io_utils import artifact_path, input_path, read_json, read_jsonl, write_json, write_jsonl
from .io_utils import repo_root_from_config
from .schemas import Episode


def create_splits(config: dict[str, Any]) -> dict[str, list[str]]:
    rows = read_jsonl(input_path(config, "cumulative_labels"))
    ids = sorted({str(row["instance_id"]) for row in rows if row.get("instance_id")})
    split_config = config.get("splits", {})
    seed = str(split_config.get("seed", 42))
    train_ratio = float(split_config.get("train", 0.7))
    dev_ratio = float(split_config.get("dev", 0.15))

    ordered = sorted(
        ids,
        key=lambda instance_id: hashlib.sha256(f"{seed}:{instance_id}".encode("utf-8")).hexdigest(),
    )
    n_total = len(ordered)
    n_train = int(round(n_total * train_ratio))
    n_dev = int(round(n_total * dev_ratio))
    if n_train + n_dev > n_total:
        n_dev = max(0, n_total - n_train)

    splits = {
        "train": ordered[:n_train],
        "dev": ordered[n_train : n_train + n_dev],
        "test": ordered[n_train + n_dev :],
    }
    write_json(artifact_path(config, "splits"), splits)
    return splits


def build_episodes(config: dict[str, Any]) -> list[dict[str, Any]]:
    splits_path = artifact_path(config, "splits")
    if not splits_path.exists():
        create_splits(config)
    splits = read_json(splits_path)
    id_to_split = {
        instance_id: split
        for split, instance_ids in splits.items()
        for instance_id in instance_ids
    }
    run_index = build_run_index(config)
    rows = read_jsonl(input_path(config, "cumulative_labels"))
    episodes: list[dict[str, Any]] = []
    for row in rows:
        instance_id = str(row.get("instance_id", ""))
        if not instance_id:
            continue
        run_info = run_index.get(instance_id, {})
        episode = Episode.from_row(
            row,
            split=id_to_split.get(instance_id, "unassigned"),
            run_id=run_info.get("run_id"),
            run_root=run_info.get("run_root"),
            final_patch=str(run_info.get("final_patch", "")),
        )
        episodes.append(episode.to_dict())
    write_jsonl(artifact_path(config, "episodes"), episodes)
    return episodes


def build_run_index(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    config_path = Path(str(config["_config_path"]))
    root = repo_root_from_config(config_path)
    run_glob = root / str(config.get("inputs", {}).get("run_glob", "runs/*"))
    index: dict[str, dict[str, str]] = {}
    for manifest_path_raw in sorted(glob.glob(str(run_glob / "manifest.json"))):
        manifest_path = Path(manifest_path_raw)
        run_root = manifest_path.parent
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        run_id = str(manifest.get("run_id") or run_root.name)
        predictions = read_predictions(run_root)
        for instance_id in manifest.get("instance_ids", []) or []:
            instance_id = str(instance_id)
            row = {
                "run_id": run_id,
                "run_root": str(run_root),
                "final_patch": predictions.get(instance_id, ""),
            }
            index.setdefault(instance_id, row)
    return index


def read_predictions(run_root: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for path in (run_root / "predictions.eval.jsonl", run_root / "predictions.jsonl"):
        for row in read_jsonl(path):
            instance_id = row.get("instance_id")
            if not instance_id:
                continue
            patch = row.get("model_patch") or row.get("patch") or row.get("submission") or ""
            if patch:
                predictions[str(instance_id)] = str(patch)
    return predictions
