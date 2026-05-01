from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(slots=True)
class SWEbenchInstance:
    instance_id: str
    repo: str
    issue_id: int | None
    base_commit: str
    problem_statement: str
    version: str | None = None
    issue_url: str | None = None
    pr_url: str | None = None
    fail_to_pass: str | None = None
    pass_to_pass: str | None = None
    dataset_name: str | None = None
    split: str | None = None

    @classmethod
    def from_dataset_record(
        cls,
        *,
        record: dict[str, Any],
        dataset_name: str,
        split: str,
    ) -> "SWEbenchInstance":
        return cls(
            instance_id=record["instance_id"],
            repo=record["repo"],
            issue_id=record.get("issue_id"),
            base_commit=record["base_commit"],
            problem_statement=record["problem_statement"],
            version=record.get("version"),
            issue_url=record.get("issue_url"),
            pr_url=record.get("pr_url"),
            fail_to_pass=record.get("FAIL_TO_PASS"),
            pass_to_pass=record.get("PASS_TO_PASS"),
            dataset_name=dataset_name,
            split=split,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "issue_id": self.issue_id,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "version": self.version,
            "issue_url": self.issue_url,
            "pr_url": self.pr_url,
            "FAIL_TO_PASS": self.fail_to_pass,
            "PASS_TO_PASS": self.pass_to_pass,
            "dataset_name": self.dataset_name,
            "split": self.split,
        }

    def fail_to_pass_list(self) -> list[str]:
        return _normalize_test_list(self.fail_to_pass)

    def pass_to_pass_list(self) -> list[str]:
        return _normalize_test_list(self.pass_to_pass)


def _normalize_test_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


@dataclass(slots=True)
class AgentAction:
    tool: str
    arguments: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentAction":
        tool = payload.get("tool")
        arguments = payload.get("arguments", {})
        if not isinstance(tool, str) or not tool:
            raise ValueError("Each action must include a non-empty string tool name.")
        if not isinstance(arguments, dict):
            raise ValueError("Each action arguments payload must be a JSON object.")
        return cls(tool=tool, arguments=arguments)


@dataclass(slots=True)
class AgentDecision:
    reasoning_summary: str
    status: str
    actions: list[AgentAction]
    final_summary: str | None = None
    confidence: float | None = None

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        max_actions_per_step: int,
    ) -> "AgentDecision":
        reasoning_summary = payload.get("reasoning_summary")
        status = payload.get("status")
        actions = payload.get("actions", [])

        if not isinstance(reasoning_summary, str) or not reasoning_summary.strip():
            raise ValueError("reasoning_summary must be a non-empty string.")
        if status not in {"continue", "done", "give_up"}:
            raise ValueError("status must be one of: continue, done, give_up.")
        if not isinstance(actions, list):
            raise ValueError("actions must be a JSON array.")
        if len(actions) > max_actions_per_step:
            raise ValueError(
                f"actions exceeded the per-step limit of {max_actions_per_step}."
            )

        parsed_actions = [AgentAction.from_dict(action) for action in actions]
        confidence = payload.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                raise ValueError("confidence must be numeric when provided.")
            confidence = float(confidence)

        final_summary = payload.get("final_summary")
        if final_summary is not None and not isinstance(final_summary, str):
            raise ValueError("final_summary must be a string when provided.")

        return cls(
            reasoning_summary=reasoning_summary.strip(),
            status=status,
            actions=parsed_actions,
            final_summary=final_summary.strip() if isinstance(final_summary, str) else None,
            confidence=confidence,
        )
