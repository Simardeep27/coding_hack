from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSONL line {line_no} in {path}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    config["_config_path"] = str(path)
    return config


def repo_root_from_config(config_path: Path) -> Path:
    # Default configs live at <repo>/rl_setup/config/*.json.
    return config_path.resolve().parents[2]


def artifact_path(config: dict[str, Any], key: str) -> Path:
    config_path = Path(str(config["_config_path"]))
    root = repo_root_from_config(config_path)
    outputs = config.get("outputs", {})
    artifact_root = root / str(outputs.get("artifact_root", "rl_setup/artifacts"))
    return artifact_root / str(outputs[key])


def input_path(config: dict[str, Any], key: str) -> Path:
    config_path = Path(str(config["_config_path"]))
    root = repo_root_from_config(config_path)
    return root / str(config.get("inputs", {}).get(key, ""))
