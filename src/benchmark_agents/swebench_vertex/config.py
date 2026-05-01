from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DATASET_ALIASES = {
    "lite": "SWE-bench/SWE-bench_Lite",
    "verified": "SWE-bench/SWE-bench_Verified",
    "full": "SWE-bench/SWE-bench",
}


@dataclass(slots=True)
class RunConfig:
    dataset_name: str = "lite"
    split: str = "test"
    sample_size: int = 10
    instance_ids: tuple[str, ...] = ()
    seed: int = 42
    run_id: str | None = None
    runs_dir: Path = Path("runs")
    max_steps: int = 24
    max_actions_per_step: int = 4
    command_timeout_secs: int = 120
    repo_cache_dir: Path = Path(".cache/repos")

    def resolved_dataset_name(self) -> str:
        return DATASET_ALIASES.get(self.dataset_name, self.dataset_name)


@dataclass(slots=True)
class VertexConfig:
    project: str
    location: str = "us-central1"
    model: str = "gemini-2.5-pro"
    temperature: float = 0.2
    max_output_tokens: int = 8192

    @classmethod
    def from_sources(
        cls,
        *,
        project: str | None,
        location: str | None,
        model: str | None,
        temperature: float,
        max_output_tokens: int,
    ) -> "VertexConfig":
        resolved_project = _normalize_setting_value(
            project or os.getenv("GOOGLE_CLOUD_PROJECT"),
            expected_key="GOOGLE_CLOUD_PROJECT",
        )
        if not resolved_project:
            raise ValueError(
                "No Google Cloud project was provided. Pass --project or set "
                "GOOGLE_CLOUD_PROJECT."
            )

        resolved_location = _normalize_setting_value(
            location or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1",
            expected_key="GOOGLE_CLOUD_LOCATION",
        )
        resolved_model = _normalize_setting_value(
            model or os.getenv("GOOGLE_GENAI_MODEL") or "gemini-2.5-pro",
            expected_key="GOOGLE_GENAI_MODEL",
        )

        return cls(
            project=resolved_project,
            location=resolved_location,
            model=resolved_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )


def _normalize_setting_value(value: str | None, *, expected_key: str) -> str | None:
    if value is None:
        return None

    cleaned = value.strip().strip("\"'")
    prefix = f"{expected_key}="
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix) :].strip().strip("\"'")
    return cleaned or None
