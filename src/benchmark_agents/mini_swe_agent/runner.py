from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from benchmark_agents.mini_swe_agent.config import MiniSweAgentConfig, read_env_file
from benchmark_agents.swebench_vertex.config import RunConfig
from benchmark_agents.swebench_vertex.dataset import (
    load_swebench_instances,
    sample_instances,
)
from benchmark_agents.swebench_vertex.models import SWEbenchInstance
from benchmark_agents.swebench_vertex.utils import append_jsonl, utc_timestamp, write_json


def run_mini_swe_agent(
    run_config: RunConfig,
    mini_config: MiniSweAgentConfig,
) -> Path:
    instances, run_root = _prepare_run_root_and_sample(run_config)
    mini_output_dir = run_root / "mini_swe_agent"
    mini_output_dir.mkdir(parents=True, exist_ok=True)

    _write_manifest(
        run_root=run_root,
        run_config=run_config,
        mini_config=mini_config,
        instances=instances,
    )
    if mini_config.expose_visible_tests:
        _write_test_visibility_splits(
            run_root=run_root,
            run_config=run_config,
            mini_config=mini_config,
            instances=instances,
        )

    command = build_mini_swebench_command(
        run_config=run_config,
        mini_config=mini_config,
        instances=instances,
        output_dir=mini_output_dir,
    )
    env = build_mini_swe_env(
        mini_config,
        global_config_dir=mini_output_dir / ".mswea",
    )
    if mini_config.expose_visible_tests:
        _enable_startup_command_compat(env, output_dir=mini_output_dir)

    executable_path = shutil.which(mini_config.executable)
    if executable_path is None:
        _write_summary(
            run_root=run_root,
            command=command,
            returncode=None,
            predictions_path=None,
            error=(
                f"{mini_config.executable!r} was not found. Install mini-SWE-agent "
                "with `pip install mini-swe-agent` or pass --mini-executable."
            ),
        )
        raise RuntimeError(
            f"{mini_config.executable!r} was not found. Install mini-SWE-agent "
            "with `pip install mini-swe-agent`."
        )

    command[0] = executable_path
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    (mini_output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (mini_output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")

    predictions_path = export_predictions(mini_output_dir, run_root)
    eval_predictions_path = export_eval_predictions(run_root=run_root)
    run_error = inspect_mini_run_error(
        mini_output_dir=mini_output_dir,
        instances=instances,
    )
    _write_summary(
        run_root=run_root,
        command=command,
        returncode=completed.returncode,
        predictions_path=predictions_path,
        eval_predictions_path=eval_predictions_path,
        error=run_error or (None if completed.returncode == 0 else completed.stderr[-4000:]),
    )
    if completed.returncode != 0 or run_error:
        raise RuntimeError(
            "mini-SWE-agent failed. "
            + (run_error or f"See {mini_output_dir / 'stderr.txt'} for details.")
        )
    return run_root


def build_mini_swebench_command(
    *,
    run_config: RunConfig,
    mini_config: MiniSweAgentConfig,
    instances: list[SWEbenchInstance],
    output_dir: Path,
) -> list[str]:
    command = [
        mini_config.executable,
        "swebench",
        "--model",
        mini_config.model,
        "--subset",
        _mini_subset_name(run_config.dataset_name),
        "--split",
        run_config.split,
        "--output",
        str(output_dir),
        "--workers",
        str(mini_config.workers),
    ]
    if instances:
        command.extend(["--filter", _instance_filter(instances)])
    extra_config_specs = _extra_config_specs(
        run_config,
        mini_config,
        output_dir=output_dir,
    )
    for config_spec in extra_config_specs:
        command.extend(["--config", config_spec])
    if mini_config.environment_class is not None:
        command.extend(["--environment-class", mini_config.environment_class])
    return command


def _extra_config_specs(
    run_config: RunConfig,
    mini_config: MiniSweAgentConfig,
    *,
    output_dir: Path,
) -> list[str]:
    specs = []
    if (
        mini_config.config_file is not None
        or mini_config.docker_platform is not None
        or mini_config.expose_visible_tests
    ):
        specs.append("swebench.yaml")
    if mini_config.config_file is not None:
        specs.append(str(mini_config.config_file))
    if mini_config.expose_visible_tests:
        specs.append(
            str(
                _write_visible_tests_config(
                    run_config,
                    mini_config,
                    output_dir=output_dir,
                )
            )
        )
    if mini_config.docker_platform is not None:
        run_args = json.dumps(
            ["--rm", "--platform", mini_config.docker_platform],
            separators=(",", ":"),
        )
        specs.append(f"environment.run_args={run_args}")
    return specs


def _write_visible_tests_config(
    run_config: RunConfig,
    mini_config: MiniSweAgentConfig,
    *,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = mini_config.visible_tests_filename.strip().strip("/")
    if not filename or "/" in filename:
        raise ValueError("--visible-tests-filename must be a simple file name.")

    config_path = output_dir / "visible_tests_config.yaml"
    readme_name = f"{filename}.README"
    visible_fail_to_pass_count = _nonnegative_int(
        mini_config.visible_fail_to_pass_count,
        "--visible-fail-to-pass-count",
    )
    min_hidden_fail_to_pass_count = _nonnegative_int(
        mini_config.min_hidden_fail_to_pass_count,
        "--min-hidden-fail-to-pass-count",
    )
    include_pass_to_pass = "True" if mini_config.include_pass_to_pass_in_visible_tests else "False"
    command = f"""python3 - <<'PY'
import hashlib
import json
from pathlib import Path

SEED = {run_config.seed}
VISIBLE_FAIL_TO_PASS_COUNT = {visible_fail_to_pass_count}
MIN_HIDDEN_FAIL_TO_PASS_COUNT = {min_hidden_fail_to_pass_count}
INCLUDE_PASS_TO_PASS = {include_pass_to_pass}
INSTANCE_ID = r'''{{{{ instance_id }}}}'''


def parse_tests(raw):
    raw = raw.strip()
    if not raw:
        return []
    try:
        tests = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    if isinstance(tests, list):
        return [str(test) for test in tests]
    return [str(tests)]


def split_fail_to_pass(tests):
    if not tests:
        return [], []
    max_visible = max(0, len(tests) - MIN_HIDDEN_FAIL_TO_PASS_COUNT)
    visible_count = min(VISIBLE_FAIL_TO_PASS_COUNT, max_visible)
    ranked = sorted(
        range(len(tests)),
        key=lambda index: hashlib.sha256(
            f'{{SEED}}\\0{{INSTANCE_ID}}\\0{{index}}\\0{{tests[index]}}'.encode()
        ).hexdigest(),
    )
    visible_indexes = set(ranked[:visible_count])
    visible = [test for index, test in enumerate(tests) if index in visible_indexes]
    hidden = [test for index, test in enumerate(tests) if index not in visible_indexes]
    return visible, hidden


fail_to_pass = parse_tests(r'''{{{{ FAIL_TO_PASS | default('[]') }}}}''')
pass_to_pass = parse_tests(r'''{{{{ PASS_TO_PASS | default('[]') }}}}''')
visible_target_tests, hidden_target_tests = split_fail_to_pass(fail_to_pass)

lines = [
    '# Public target tests: visible subset of FAIL_TO_PASS',
    *visible_target_tests,
]
if INCLUDE_PASS_TO_PASS:
    lines.extend([
        '',
        '# Regression guard tests: PASS_TO_PASS',
        *pass_to_pass,
    ])

visible_path = Path('/testbed/{filename}')
visible_path.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
Path('/testbed/{readme_name}').write_text(
    (
        'Public tests for this task. This file may contain a visible subset of '
        'FAIL_TO_PASS target tests and PASS_TO_PASS regression tests. Held-out '
        'FAIL_TO_PASS target tests are intentionally not included.\\n'
        f'visible_fail_to_pass_count={{len(visible_target_tests)}}\\n'
        f'hidden_fail_to_pass_count={{len(hidden_target_tests)}}\\n'
    ),
    encoding='utf-8',
)

exclude_path = Path('/testbed/.git/info/exclude')
if exclude_path.exists():
    existing = exclude_path.read_text(encoding='utf-8')
    additions = ['/{filename}', '/{readme_name}']
    with exclude_path.open('a', encoding='utf-8') as handle:
        for pattern in additions:
            if pattern not in existing:
                handle.write('\\n' + pattern)
PY
"""
    config_path.write_text(
        "run:\n"
        "  env_startup_command: |\n"
        + "".join(f"    {line}\n" for line in command.rstrip("\n").splitlines()),
        encoding="utf-8",
    )
    return config_path


def _write_test_visibility_splits(
    *,
    run_root: Path,
    run_config: RunConfig,
    mini_config: MiniSweAgentConfig,
    instances: list[SWEbenchInstance],
) -> Path:
    output_file = run_root / "test_visibility_splits.jsonl"
    output_file.write_text("", encoding="utf-8")
    for instance in instances:
        visible_target, hidden_target = _split_fail_to_pass_tests(
            instance.fail_to_pass_list(),
            instance_id=instance.instance_id,
            seed=run_config.seed,
            visible_count=mini_config.visible_fail_to_pass_count,
            min_hidden_count=mini_config.min_hidden_fail_to_pass_count,
        )
        append_jsonl(
            output_file,
            {
                "instance_id": instance.instance_id,
                "visible_fail_to_pass": visible_target,
                "hidden_fail_to_pass": hidden_target,
                "pass_to_pass": instance.pass_to_pass_list()
                if mini_config.include_pass_to_pass_in_visible_tests
                else [],
                "visible_tests_filename": mini_config.visible_tests_filename,
                "policy": {
                    "seed": run_config.seed,
                    "visible_fail_to_pass_count": mini_config.visible_fail_to_pass_count,
                    "min_hidden_fail_to_pass_count": mini_config.min_hidden_fail_to_pass_count,
                    "include_pass_to_pass_in_visible_tests": (
                        mini_config.include_pass_to_pass_in_visible_tests
                    ),
                },
            },
        )
    return output_file


def _split_fail_to_pass_tests(
    tests: list[str],
    *,
    instance_id: str,
    seed: int,
    visible_count: int,
    min_hidden_count: int,
) -> tuple[list[str], list[str]]:
    visible_count = _nonnegative_int(visible_count, "--visible-fail-to-pass-count")
    min_hidden_count = _nonnegative_int(
        min_hidden_count,
        "--min-hidden-fail-to-pass-count",
    )
    if not tests:
        return [], []

    max_visible = max(0, len(tests) - min_hidden_count)
    visible_count = min(visible_count, max_visible)
    ranked_indexes = sorted(
        range(len(tests)),
        key=lambda index: hashlib.sha256(
            f"{seed}\0{instance_id}\0{index}\0{tests[index]}".encode()
        ).hexdigest(),
    )
    visible_indexes = set(ranked_indexes[:visible_count])
    visible = [test for index, test in enumerate(tests) if index in visible_indexes]
    hidden = [test for index, test in enumerate(tests) if index not in visible_indexes]
    return visible, hidden


def _nonnegative_int(value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def build_mini_swe_env(
    mini_config: MiniSweAgentConfig,
    *,
    global_config_dir: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env_values = read_env_file(mini_config.env_file)
    env.update(env_values)

    if global_config_dir is not None:
        global_config_dir.mkdir(parents=True, exist_ok=True)
        env["MSWEA_GLOBAL_CONFIG_DIR"] = str(global_config_dir)
    env.setdefault("MSWEA_SILENT_STARTUP", "true")

    if mini_config.project:
        env["GOOGLE_CLOUD_PROJECT"] = mini_config.project
        env["VERTEXAI_PROJECT"] = mini_config.project
    if mini_config.location:
        env["GOOGLE_CLOUD_LOCATION"] = mini_config.location
        env["VERTEXAI_LOCATION"] = mini_config.location
    env.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    return env


def _enable_startup_command_compat(env: dict[str, str], *, output_dir: Path) -> None:
    """Patch mini-SWE-agent 2.2.x env_startup_command string execution per run."""
    patch_dir = output_dir / ".mswea_compat"
    patch_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / "sitecustomize.py").write_text(
        """
from importlib import import_module


def _patch_execute(class_path):
    try:
        module_name, class_name = class_path.rsplit(".", 1)
        cls = getattr(import_module(module_name), class_name)
        original = cls.execute
    except Exception:
        return

    if getattr(original, "_benchmark_agents_accepts_string_action", False):
        return

    def execute(self, action, cwd="", *, timeout=None):
        if isinstance(action, str):
            action = {"command": action}
        return original(self, action, cwd=cwd, timeout=timeout)

    execute._benchmark_agents_accepts_string_action = True
    cls.execute = execute


for _class_path in [
    "minisweagent.environments.docker.DockerEnvironment",
    "minisweagent.environments.local.LocalEnvironment",
    "minisweagent.environments.singularity.SingularityEnvironment",
    "minisweagent.environments.extra.swerex_docker.SwerexDockerEnvironment",
    "minisweagent.environments.extra.swerex_modal.SwerexModalEnvironment",
]:
    _patch_execute(_class_path)
""".lstrip(),
        encoding="utf-8",
    )
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(patch_dir)
        if not existing_pythonpath
        else str(patch_dir) + os.pathsep + existing_pythonpath
    )


def export_predictions(mini_output_dir: Path, run_root: Path) -> Path | None:
    candidates = [
        mini_output_dir / "preds.jsonl",
        mini_output_dir / "predictions.jsonl",
        mini_output_dir / "preds.json",
        mini_output_dir / "predictions.json",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        return None

    target = run_root / "predictions.jsonl"
    if source.suffix == ".jsonl":
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return target

    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data.values() if isinstance(data, dict) else data
    target.write_text("", encoding="utf-8")
    for row in rows:
        append_jsonl(target, row)
    return target


def export_eval_predictions(
    *,
    run_root: Path,
    output_file: Path | None = None,
) -> Path:
    run_root = run_root.resolve()
    mini_output_dir = run_root / "mini_swe_agent"
    if output_file is None:
        output_file = run_root / "predictions.eval.jsonl"
    else:
        output_file = output_file.resolve()

    rows = _read_prediction_rows(run_root=run_root, mini_output_dir=mini_output_dir)
    status_by_instance = _status_by_instance(_read_latest_exit_statuses(mini_output_dir))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("", encoding="utf-8")

    excluded: dict[str, list[str]] = {}
    written = 0
    for row in rows:
        instance_id = str(row.get("instance_id", ""))
        patch = str(row.get("model_patch", ""))
        status = status_by_instance.get(instance_id)
        reason = None
        if status_by_instance and status != "Submitted":
            reason = f"status:{status or 'missing'}"
        elif not patch.strip():
            reason = "empty_patch"

        if reason is not None:
            excluded.setdefault(reason, []).append(instance_id)
            continue

        append_jsonl(output_file, row)
        written += 1

    write_json(
        _eval_predictions_summary_path(output_file),
        {
            "created_at": utc_timestamp(),
            "source_run_root": str(run_root),
            "source_predictions": str(_prediction_source_path(run_root, mini_output_dir)),
            "output_file": str(output_file),
            "total_predictions": len(rows),
            "written": written,
            "excluded": excluded,
        },
    )
    return output_file


def inspect_mini_run_error(
    *,
    mini_output_dir: Path,
    instances: list[SWEbenchInstance],
) -> str | None:
    instance_ids = [instance.instance_id for instance in instances]
    exit_statuses = _read_latest_exit_statuses(mini_output_dir)
    failed_statuses = {
        status: ids
        for status, ids in exit_statuses.items()
        if status != "Submitted" and ids
    }
    if failed_statuses:
        details = "; ".join(
            f"{status}: {', '.join(ids[:5])}{' ...' if len(ids) > 5 else ''}"
            for status, ids in sorted(failed_statuses.items())
        )
        return (
            "mini-SWE-agent reported failed instance statuses: "
            f"{details}. See {mini_output_dir / 'minisweagent.log'}."
        )

    predictions = _read_json_object(mini_output_dir / "preds.json")
    empty_patch_instances = [
        instance_id
        for instance_id in instance_ids
        if not str(predictions.get(instance_id, {}).get("model_patch", "")).strip()
    ]
    if empty_patch_instances:
        return (
            "mini-SWE-agent produced empty patches for: "
            + ", ".join(empty_patch_instances[:10])
            + (" ..." if len(empty_patch_instances) > 10 else "")
        )
    return None


def _prepare_run_root_and_sample(
    run_config: RunConfig,
) -> tuple[list[SWEbenchInstance], Path]:
    run_root = (run_config.runs_dir / _resolve_run_id(run_config)).resolve()
    run_root.mkdir(parents=True, exist_ok=False)

    instances = load_swebench_instances(
        run_config.resolved_dataset_name(),
        run_config.split,
    )
    if run_config.instance_ids:
        instance_map = {instance.instance_id: instance for instance in instances}
        missing = [
            instance_id
            for instance_id in run_config.instance_ids
            if instance_id not in instance_map
        ]
        if missing:
            raise ValueError(
                "The following instance ids were not found in the selected dataset split: "
                + ", ".join(missing)
            )
        sampled = [instance_map[instance_id] for instance_id in run_config.instance_ids]
    else:
        sampled = sample_instances(instances, run_config.sample_size, run_config.seed)
    return sampled, run_root


def _write_manifest(
    *,
    run_root: Path,
    run_config: RunConfig,
    mini_config: MiniSweAgentConfig,
    instances: list[SWEbenchInstance],
) -> None:
    manifest: dict[str, Any] = {
        "run_id": run_root.name,
        "created_at": utc_timestamp(),
        "agent": "mini-swe-agent",
        "dataset_name": run_config.resolved_dataset_name(),
        "mini_subset": _mini_subset_name(run_config.dataset_name),
        "split": run_config.split,
        "sample_size": len(instances),
        "instance_ids": [instance.instance_id for instance in instances],
        "seed": run_config.seed,
        "mini_swe_agent": {
            "model": mini_config.model,
            "location": mini_config.location,
            "project": mini_config.project,
            "workers": mini_config.workers,
            "environment_class": mini_config.environment_class,
            "config_file": str(mini_config.config_file)
            if mini_config.config_file is not None
            else None,
            "expose_visible_tests": mini_config.expose_visible_tests,
            "visible_tests_filename": mini_config.visible_tests_filename,
            "visible_fail_to_pass_count": mini_config.visible_fail_to_pass_count,
            "min_hidden_fail_to_pass_count": mini_config.min_hidden_fail_to_pass_count,
            "include_pass_to_pass_in_visible_tests": (
                mini_config.include_pass_to_pass_in_visible_tests
            ),
        },
    }
    write_json(run_root / "manifest.json", manifest)
    for instance in instances:
        append_jsonl(run_root / "sampled_instances.jsonl", instance.to_dict())


def _write_summary(
    *,
    run_root: Path,
    command: list[str],
    returncode: int | None,
    predictions_path: Path | None,
    error: str | None,
    eval_predictions_path: Path | None = None,
) -> None:
    write_json(
        run_root / "summary.json",
        {
            "run_id": run_root.name,
            "agent": "mini-swe-agent",
            "completed_at": utc_timestamp(),
            "command": command,
            "returncode": returncode,
            "predictions_path": str(predictions_path) if predictions_path else None,
            "eval_predictions_path": str(eval_predictions_path)
            if eval_predictions_path
            else None,
            "error": error,
        },
    )


def _resolve_run_id(run_config: RunConfig) -> str:
    if run_config.run_id:
        return run_config.run_id
    dataset_slug = run_config.resolved_dataset_name().split("/")[-1].lower()
    timestamp = utc_timestamp().replace(":", "").replace("-", "").replace("+00:00", "z")
    return f"{dataset_slug}-mini-swe-{run_config.sample_size}x-seed{run_config.seed}-{timestamp}"


def _instance_filter(instances: list[SWEbenchInstance]) -> str:
    escaped = [re.escape(instance.instance_id) for instance in instances]
    return "^(?:" + "|".join(escaped) + ")$"


def _mini_subset_name(dataset_name: str) -> str:
    aliases = {
        "lite": "lite",
        "verified": "verified",
        "full": "full",
        "SWE-bench/SWE-bench_Lite": "lite",
        "SWE-bench/SWE-bench_Verified": "verified",
        "SWE-bench/SWE-bench": "full",
    }
    return aliases.get(dataset_name, dataset_name)


def _read_latest_exit_statuses(mini_output_dir: Path) -> dict[str, list[str]]:
    status_files = sorted(
        mini_output_dir.glob("exit_statuses_*.yaml"),
        key=lambda path: path.stat().st_mtime,
    )
    if not status_files:
        return {}
    return _parse_exit_statuses(status_files[-1].read_text(encoding="utf-8"))


def _parse_exit_statuses(text: str) -> dict[str, list[str]]:
    statuses: dict[str, list[str]] = {}
    current_status: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "instances_by_exit_status:":
            continue
        if stripped.endswith(":") and not stripped.startswith("-"):
            current_status = stripped[:-1]
            statuses.setdefault(current_status, [])
            continue
        if current_status is not None and stripped.startswith("- "):
            statuses[current_status].append(stripped[2:])
    return statuses


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_prediction_rows(*, run_root: Path, mini_output_dir: Path) -> list[dict[str, Any]]:
    jsonl_path = run_root / "predictions.jsonl"
    if jsonl_path.exists():
        rows = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        return rows

    json_path = mini_output_dir / "preds.json"
    data = _read_json_object(json_path)
    return [row for row in data.values() if isinstance(row, dict)]


def _prediction_source_path(run_root: Path, mini_output_dir: Path) -> Path | None:
    if (run_root / "predictions.jsonl").exists():
        return run_root / "predictions.jsonl"
    if (mini_output_dir / "preds.json").exists():
        return mini_output_dir / "preds.json"
    return None


def _status_by_instance(statuses: dict[str, list[str]]) -> dict[str, str]:
    return {
        instance_id: status
        for status, instance_ids in statuses.items()
        for instance_id in instance_ids
    }


def _eval_predictions_summary_path(output_file: Path) -> Path:
    if output_file.suffix:
        return output_file.with_suffix(".summary.json")
    return output_file.with_name(f"{output_file.name}.summary.json")
