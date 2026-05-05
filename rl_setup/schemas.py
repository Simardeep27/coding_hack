from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Paths:
    artifact_root: str
    splits: str
    episodes: str
    scored_episodes: str
    preference_pairs: str
    accepted_episodes: str
    rejected_episodes: str
    summary_json: str
    summary_md: str


@dataclass
class Episode:
    instance_id: str
    split: str
    run_id: str | None
    run_root: str | None
    resolved: bool | None
    patch_successfully_applied: bool | None
    visible_rate: float | None
    hidden_rate: float | None
    overfit_gap: float | None
    eval_label: str
    is_reward_hack: bool
    exploit_type: str
    exploitation_level: str
    confidence: float | None
    static_signals: dict[str, bool] = field(default_factory=dict)
    modified_files: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    counterevidence: list[str] = field(default_factory=list)
    rationale: str = ""
    final_patch: str = ""

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        *,
        split: str,
        run_id: str | None = None,
        run_root: str | None = None,
        final_patch: str = "",
    ) -> "Episode":
        return cls(
            instance_id=str(row.get("instance_id", "")),
            split=split,
            run_id=run_id,
            run_root=run_root,
            resolved=_maybe_bool(row.get("resolved")),
            patch_successfully_applied=_maybe_bool(row.get("patch_successfully_applied")),
            visible_rate=_maybe_float(row.get("visible_rate")),
            hidden_rate=_maybe_float(row.get("hidden_rate")),
            overfit_gap=_maybe_float(row.get("overfit_gap")),
            eval_label=str(row.get("eval_label", "UNKNOWN")),
            is_reward_hack=bool(row.get("is_reward_hack", False)),
            exploit_type=str(row.get("exploit_type", "none")),
            exploitation_level=str(row.get("E", "none")),
            confidence=_maybe_float(row.get("confidence")),
            static_signals=_bool_dict(row.get("static_signals")),
            modified_files=_str_list(row.get("modified_files")),
            commands=_str_list(row.get("commands")),
            evidence=_str_list(row.get("evidence")),
            counterevidence=_str_list(row.get("counterevidence")),
            rationale=str(row.get("rationale", "")),
            final_patch=final_patch,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "split": self.split,
            "run_id": self.run_id,
            "run_root": self.run_root,
            "resolved": self.resolved,
            "patch_successfully_applied": self.patch_successfully_applied,
            "visible_rate": self.visible_rate,
            "hidden_rate": self.hidden_rate,
            "overfit_gap": self.overfit_gap,
            "eval_label": self.eval_label,
            "is_reward_hack": self.is_reward_hack,
            "exploit_type": self.exploit_type,
            "E": self.exploitation_level,
            "confidence": self.confidence,
            "static_signals": self.static_signals,
            "modified_files": self.modified_files,
            "commands": self.commands,
            "evidence": self.evidence,
            "counterevidence": self.counterevidence,
            "rationale": self.rationale,
            "final_patch": self.final_patch,
        }


def _maybe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "resolved", "pass", "passed"}:
            return True
        if lowered in {"false", "0", "no", "unresolved", "fail", "failed"}:
            return False
    return None


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _bool_dict(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    return {str(key): bool(item) for key, item in value.items()}


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]

