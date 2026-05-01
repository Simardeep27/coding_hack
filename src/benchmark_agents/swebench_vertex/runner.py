from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark_agents.swebench_vertex.config import RunConfig, VertexConfig
from benchmark_agents.swebench_vertex.dataset import (
    load_swebench_instances,
    sample_instances,
)
from benchmark_agents.swebench_vertex.models import AgentDecision, SWEbenchInstance
from benchmark_agents.swebench_vertex.prompts import build_agent_prompt
from benchmark_agents.swebench_vertex.utils import (
    append_jsonl,
    extract_json_object,
    truncate_text,
    utc_timestamp,
    write_json,
)
from benchmark_agents.swebench_vertex.vertex import VertexResponder
from benchmark_agents.swebench_vertex.workspace import WorkspaceSession


def create_sample_manifest(run_config: RunConfig) -> Path:
    instances, run_root = _prepare_run_root_and_sample(run_config)
    _write_manifest(run_root=run_root, run_config=run_config, instances=instances)
    return run_root


def run_swebench_vertex_agent(run_config: RunConfig, vertex_config: VertexConfig) -> Path:
    instances, run_root = _prepare_run_root_and_sample(run_config)
    _write_manifest(
        run_root=run_root,
        run_config=run_config,
        instances=instances,
        vertex_config=vertex_config,
    )

    predictions_path = run_root / "predictions.jsonl"
    summary: dict[str, Any] = {
        "run_id": run_root.name,
        "started_at": utc_timestamp(),
        "dataset_name": run_config.resolved_dataset_name(),
        "split": run_config.split,
        "sample_size": len(instances),
        "seed": run_config.seed,
        "model_name_or_path": vertex_config.model,
        "tasks": [],
    }

    vertex = VertexResponder(vertex_config)
    try:
        for index, instance in enumerate(instances, start=1):
            task_result = _run_single_instance(
                index=index,
                instance=instance,
                run_root=run_root,
                run_config=run_config,
                vertex=vertex,
            )
            if task_result["prediction_written"]:
                append_jsonl(predictions_path, task_result["prediction"])
            summary["tasks"].append({**task_result, "prediction": None})
    finally:
        vertex.close()

    summary["completed_at"] = utc_timestamp()
    summary["predictions_path"] = str(predictions_path)
    summary["submitted_predictions"] = sum(
        1 for task in summary["tasks"] if task["prediction_written"]
    )
    write_json(run_root / "summary.json", summary)
    return run_root


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
    instances: list[SWEbenchInstance],
    vertex_config: VertexConfig | None = None,
) -> None:
    manifest = {
        "run_id": run_root.name,
        "created_at": utc_timestamp(),
        "dataset_name": run_config.resolved_dataset_name(),
        "split": run_config.split,
        "sample_size": len(instances),
        "instance_ids": list(run_config.instance_ids),
        "seed": run_config.seed,
        "max_steps": run_config.max_steps,
        "max_actions_per_step": run_config.max_actions_per_step,
        "command_timeout_secs": run_config.command_timeout_secs,
        "repo_cache_dir": str(run_config.repo_cache_dir),
    }
    if vertex_config is not None:
        manifest["vertex"] = {
            "project": vertex_config.project,
            "location": vertex_config.location,
            "model": vertex_config.model,
            "temperature": vertex_config.temperature,
            "max_output_tokens": vertex_config.max_output_tokens,
        }

    write_json(run_root / "manifest.json", manifest)
    for instance in instances:
        append_jsonl(run_root / "sampled_instances.jsonl", instance.to_dict())


def _run_single_instance(
    *,
    index: int,
    instance: SWEbenchInstance,
    run_root: Path,
    run_config: RunConfig,
    vertex: VertexResponder,
) -> dict[str, Any]:
    task_root = run_root / "tasks" / instance.instance_id
    prompts_root = task_root / "prompts"
    workspace_root = task_root / "workspace"
    task_root.mkdir(parents=True, exist_ok=True)
    prompts_root.mkdir(parents=True, exist_ok=True)

    write_json(task_root / "instance.json", instance.to_dict())

    workspace = WorkspaceSession(
        instance=instance,
        workspace_root=workspace_root,
        repo_cache_dir=run_config.repo_cache_dir,
        default_command_timeout_secs=run_config.command_timeout_secs,
    )

    task_summary: dict[str, Any] = {
        "index": index,
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "prediction_written": False,
        "status": "running",
        "started_at": utc_timestamp(),
        "steps_executed": 0,
        "error": None,
    }

    recent_trace_entries: list[str] = []

    try:
        workspace.prepare()
        runtime_metadata = workspace.runtime_metadata()
        _write_agent_context_files(
            instance=instance,
            repo_dir=workspace.repo_dir,
            runtime_metadata=runtime_metadata,
        )

        final_decision: AgentDecision | None = None
        for step_number in range(1, run_config.max_steps + 1):
            git_status = workspace.get_git_status()
            git_diff_excerpt = workspace.get_git_diff(truncate_chars=12_000)
            prompt = build_agent_prompt(
                instance=instance,
                step_number=step_number,
                max_actions_per_step=run_config.max_actions_per_step,
                command_timeout_secs=run_config.command_timeout_secs,
                recent_trace_excerpt="\n\n".join(recent_trace_entries[-8:]),
                git_status=git_status,
                git_diff_excerpt=git_diff_excerpt,
                environment_summary=_format_environment_summary(runtime_metadata),
            )
            prompt_path = prompts_root / f"step_{step_number:03d}_prompt.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")

            raw_response, decision = _request_decision(
                vertex=vertex,
                prompt=prompt,
                max_actions_per_step=run_config.max_actions_per_step,
            )
            response_path = prompts_root / f"step_{step_number:03d}_response.txt"
            response_path.write_text(raw_response + "\n", encoding="utf-8")

            observations = []
            for action in decision.actions:
                observations.append(_execute_action(workspace, action.tool, action.arguments))

            post_status = workspace.get_git_status()
            post_diff = workspace.get_git_diff(truncate_chars=12_000)
            trace_entry = {
                "timestamp": utc_timestamp(),
                "step": step_number,
                "reasoning_summary": decision.reasoning_summary,
                "status": decision.status,
                "final_summary": decision.final_summary,
                "confidence": decision.confidence,
                "actions": [
                    {"tool": action.tool, "arguments": action.arguments}
                    for action in decision.actions
                ],
                "observations": observations,
                "git_status": post_status,
                "git_diff_excerpt": post_diff,
            }
            append_jsonl(task_root / "trace.jsonl", trace_entry)
            append_jsonl(workspace.repo_dir / ".benchmark_agent" / "history.jsonl", trace_entry)
            recent_trace_entries.append(_format_trace_excerpt(trace_entry))
            task_summary["steps_executed"] = step_number
            final_decision = decision

            if decision.status in {"done", "give_up"}:
                break

        final_patch = workspace.get_git_diff()
        prediction_payload: dict[str, Any] | None = None
        if final_patch.strip():
            (task_root / "final_patch.diff").write_text(final_patch + "\n", encoding="utf-8")
            prediction_payload = {
                "instance_id": instance.instance_id,
                "model_name_or_path": vertex.model_name,
                "model_patch": final_patch,
            }
            task_summary["prediction_written"] = True
            task_summary["final_patch_path"] = str(task_root / "final_patch.diff")
        else:
            task_summary["final_patch_path"] = None

        task_summary["status"] = (
            final_decision.status
            if final_decision is not None and final_decision.status in {"done", "give_up"}
            else "max_steps_reached"
        )
        task_summary["final_summary"] = (
            final_decision.final_summary if final_decision is not None else None
        )
        task_summary["completed_at"] = utc_timestamp()
        task_summary["prediction"] = prediction_payload
        write_json(
            task_root / "task_summary.json",
            {**task_summary, "prediction": None},
        )
        return task_summary

    except Exception as exc:  # noqa: BLE001
        task_summary["status"] = "error"
        task_summary["error"] = str(exc)
        task_summary["completed_at"] = utc_timestamp()
        append_jsonl(
            task_root / "trace.jsonl",
            {
                "timestamp": utc_timestamp(),
                "step": task_summary["steps_executed"],
                "error": str(exc),
            },
        )
        write_json(task_root / "task_summary.json", task_summary)
        return task_summary


def _request_decision(
    *,
    vertex: VertexResponder,
    prompt: str,
    max_actions_per_step: int,
) -> tuple[str, AgentDecision]:
    last_error = None
    augmented_prompt = prompt
    for attempt in range(1, 3):
        raw_response = vertex.generate_json_response(augmented_prompt)
        try:
            payload = extract_json_object(raw_response)
            decision = AgentDecision.from_dict(
                payload,
                max_actions_per_step=max_actions_per_step,
            )
            return raw_response, decision
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            augmented_prompt = (
                f"{prompt}\n\nThe previous response was invalid JSON for this schema: "
                f"{last_error}. Return one valid JSON object only."
            )
    raise ValueError(f"Unable to parse model response as AgentDecision: {last_error}")


def _execute_action(
    workspace: WorkspaceSession,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    tool_map = {
        "list_files": workspace.list_files,
        "search_code": workspace.search_code,
        "read_file": workspace.read_file,
        "replace_text": workspace.replace_text,
        "write_file": workspace.write_file,
        "run_command": workspace.run_command,
        "get_git_status": workspace.get_git_status,
        "get_git_diff": workspace.get_git_diff,
    }

    tool = tool_map.get(tool_name)
    if tool is None:
        return {
            "tool": tool_name,
            "ok": False,
            "output": f"Unknown tool: {tool_name}",
        }

    try:
        output = tool(**arguments)
        return {
            "tool": tool_name,
            "ok": True,
            "output": truncate_text(str(output), 12_000),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "tool": tool_name,
            "ok": False,
            "output": str(exc),
        }


def _format_trace_excerpt(trace_entry: dict[str, Any]) -> str:
    actions = ", ".join(action["tool"] for action in trace_entry["actions"]) or "no actions"
    observation_lines = []
    for observation in trace_entry["observations"][:3]:
        prefix = "ok" if observation["ok"] else "error"
        observation_lines.append(
            f"- {observation['tool']} ({prefix}): {truncate_text(observation['output'], 500)}"
        )
    observations = "\n".join(observation_lines) if observation_lines else "- no observations"
    return (
        f"Step {trace_entry['step']} [{trace_entry['status']}]\n"
        f"Reasoning: {trace_entry['reasoning_summary']}\n"
        f"Actions: {actions}\n"
        f"Observations:\n{observations}"
    )


def _write_agent_context_files(
    *,
    instance: SWEbenchInstance,
    repo_dir: Path,
    runtime_metadata: dict[str, object],
) -> None:
    context_dir = repo_dir / ".benchmark_agent"
    context_dir.mkdir(parents=True, exist_ok=True)

    fail_to_pass = instance.fail_to_pass_list()
    pass_to_pass = instance.pass_to_pass_list()

    task_context = {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "base_commit": instance.base_commit,
        "issue_url": instance.issue_url,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
    }

    (context_dir / "task_context.json").write_text(
        json.dumps(task_context, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (context_dir / "environment.json").write_text(
        json.dumps(runtime_metadata, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (context_dir / "fail_to_pass_tests.txt").write_text(
        _render_context_test_list(fail_to_pass),
        encoding="utf-8",
    )
    (context_dir / "pass_to_pass_tests.txt").write_text(
        _render_context_test_list(pass_to_pass),
        encoding="utf-8",
    )
    (context_dir / "history.jsonl").write_text("", encoding="utf-8")
    (context_dir / "README.txt").write_text(
        (
            "This directory is created for the benchmark agent.\n"
            "It contains the SWE-bench task metadata and the priority test lists.\n"
            "Use FAIL_TO_PASS as the main target set and PASS_TO_PASS as regression checks.\n"
            "The full step-by-step trace is mirrored into history.jsonl as the run progresses.\n"
            "Environment metadata is available in environment.json.\n"
        ),
        encoding="utf-8",
    )

    _ensure_git_exclude_entry(repo_dir=repo_dir, pattern=".benchmark_agent/")


def _render_context_test_list(test_names: list[str]) -> str:
    if not test_names:
        return "(none provided)\n"
    return "\n".join(test_names) + "\n"


def _ensure_git_exclude_entry(*, repo_dir: Path, pattern: str) -> None:
    exclude_path = repo_dir / ".git" / "info" / "exclude"
    existing_lines = []
    if exclude_path.exists():
        existing_lines = exclude_path.read_text(encoding="utf-8").splitlines()
    if pattern in existing_lines:
        return

    prefix = "\n" if exclude_path.exists() and exclude_path.stat().st_size > 0 else ""
    with exclude_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{prefix}{pattern}\n")


def _format_environment_summary(runtime_metadata: dict[str, object]) -> str:
    available_commands = runtime_metadata.get("available_commands", {})
    if not isinstance(available_commands, dict):
        available_commands = {}

    command_lines = []
    for name in ("python", "python3", "pytest", "rg", "grep", "pip", "uv", "tox"):
        path = available_commands.get(name)
        command_lines.append(f"- {name}: {path or 'not found'}")

    notes = runtime_metadata.get("validation_notes", [])
    note_lines = []
    if isinstance(notes, list):
        note_lines = [f"- {str(note)}" for note in notes]

    python_version = runtime_metadata.get("python_version", "unknown")
    return "\n".join(
        [
            f"- host python version: {python_version}",
            "- full trace history: .benchmark_agent/history.jsonl",
            "- environment metadata: .benchmark_agent/environment.json",
            *command_lines,
            *note_lines,
            "- If validation fails with interpreter or dependency errors, treat that as an environment clue, not automatic proof the code change is wrong.",
            "- Prefer repo-native test runners and targeted tests over broad package installs.",
        ]
    )


def _resolve_run_id(run_config: RunConfig) -> str:
    if run_config.run_id:
        return run_config.run_id
    dataset_slug = run_config.resolved_dataset_name().split("/")[-1].lower()
    timestamp = utc_timestamp().replace(":", "").replace("-", "").replace("+00:00", "z")
    return f"{dataset_slug}-{run_config.sample_size}x-seed{run_config.seed}-{timestamp}"
