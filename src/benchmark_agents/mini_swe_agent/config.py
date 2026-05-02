from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from benchmark_agents.swebench_vertex.config import _normalize_setting_value


@dataclass(slots=True)
class MiniSweAgentConfig:
    model: str
    project: str | None = None
    location: str = "us-central1"
    workers: int = 1
    environment_class: str | None = None
    executable: str = "mini-extra"
    config_file: Path | None = None
    docker_platform: str | None = None
    env_file: Path = Path(".env")

    @classmethod
    def from_sources(
        cls,
        *,
        model: str | None,
        project: str | None,
        location: str | None,
        workers: int,
        environment_class: str | None,
        executable: str,
        config_file: Path | None,
        docker_platform: str | None,
        env_file: Path,
    ) -> "MiniSweAgentConfig":
        env_values = read_env_file(env_file)
        resolved_project = _normalize_setting_value(
            project
            or env_values.get("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT"),
            expected_key="GOOGLE_CLOUD_PROJECT",
        )
        resolved_location = _normalize_setting_value(
            location
            or env_values.get("GOOGLE_CLOUD_LOCATION")
            or os.getenv("GOOGLE_CLOUD_LOCATION")
            or "us-central1",
            expected_key="GOOGLE_CLOUD_LOCATION",
        )
        resolved_model = _normalize_setting_value(
            model
            or env_values.get("MINI_SWE_MODEL")
            or os.getenv("MINI_SWE_MODEL")
            or env_values.get("GOOGLE_GENAI_MODEL")
            or os.getenv("GOOGLE_GENAI_MODEL")
            or "gemini-2.5-pro",
            expected_key="GOOGLE_GENAI_MODEL",
        )

        return cls(
            model=to_litellm_vertex_model(resolved_model or "gemini-2.5-pro"),
            project=resolved_project,
            location=resolved_location or "us-central1",
            workers=workers,
            environment_class=environment_class,
            executable=executable,
            config_file=config_file,
            docker_platform=docker_platform,
            env_file=env_file,
        )


def to_litellm_vertex_model(model: str) -> str:
    cleaned = model.strip().strip("\"'")
    if "/" in cleaned:
        return cleaned
    return f"vertex_ai/{cleaned}"


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values
