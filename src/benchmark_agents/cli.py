from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_agents.mini_swe_agent.config import MiniSweAgentConfig
from benchmark_agents.mini_swe_agent.runner import (
    export_eval_predictions,
    run_mini_swe_agent,
)
from benchmark_agents.swebench_vertex.config import (
    RunConfig,
    VertexConfig,
)
from benchmark_agents.swebench_vertex.reward_hack_dataset import (
    create_reward_hack_dataset,
)
from benchmark_agents.swebench_vertex.runner import (
    create_sample_manifest,
    run_swebench_vertex_agent,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark-agents",
        description="Run benchmark-oriented coding agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    swebench_parser = subparsers.add_parser(
        "swebench",
        help="SWE-bench agents and utilities.",
    )
    swebench_subparsers = swebench_parser.add_subparsers(
        dest="swebench_command",
        required=True,
    )

    sample_parser = swebench_subparsers.add_parser(
        "sample",
        help="Create a reproducible SWE-bench sample manifest.",
    )
    _add_common_swebench_args(sample_parser)
    sample_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where the run folder will be created.",
    )

    run_parser = swebench_subparsers.add_parser(
        "run",
        help="Sample SWE-bench tasks and run mini-SWE-agent.",
    )
    _add_common_swebench_args(run_parser)
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where the run folder will be created.",
    )
    _add_mini_swe_args(run_parser)

    vertex_run_parser = swebench_subparsers.add_parser(
        "vertex-run",
        help="Run the legacy custom Vertex-backed JSON tool solver loop.",
    )
    _add_common_swebench_args(vertex_run_parser)
    _add_vertex_run_args(vertex_run_parser)
    vertex_run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where the run folder will be created.",
    )

    dataset_parser = swebench_subparsers.add_parser(
        "dataset",
        help="Audit a completed run and create a reward-hacking JSONL dataset.",
    )
    dataset_parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Completed run directory from swebench run or vertex-run.",
    )
    dataset_parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to <run-root>/reward_hack_cases.jsonl.",
    )
    dataset_parser.add_argument(
        "--include-non-hacks",
        action="store_true",
        help=(
            "Include NEEDS_REVIEW and HONEST_OR_UNKNOWN rows instead of only "
            "REWARD_HACK."
        ),
    )

    eval_predictions_parser = swebench_subparsers.add_parser(
        "eval-predictions",
        help="Create a SWE-bench predictions file excluding failed mini-SWE instances.",
    )
    eval_predictions_parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Completed mini-SWE run directory.",
    )
    eval_predictions_parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to <run-root>/predictions.eval.jsonl.",
    )

    return parser


def _add_vertex_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", help="Google Cloud project id.")
    parser.add_argument(
        "--location",
        default=None,
        help="Vertex AI region. Defaults to GOOGLE_CLOUD_LOCATION or us-central1.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Vertex model id. Defaults to GOOGLE_GENAI_MODEL or gemini-2.5-pro.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="Maximum model output tokens per step.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="Maximum agent turns per SWE-bench instance.",
    )
    parser.add_argument(
        "--max-actions-per-step",
        type=int,
        default=4,
        help="Maximum number of tool actions the model may request in a step.",
    )
    parser.add_argument(
        "--command-timeout-secs",
        type=int,
        default=120,
        help="Timeout for repo-local commands executed by the agent.",
    )
    parser.add_argument(
        "--repo-cache-dir",
        type=Path,
        default=Path(".cache/repos"),
        help="Directory for mirrored repository caches.",
    )


def _add_mini_swe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", help="Google Cloud project id.")
    parser.add_argument(
        "--location",
        default=None,
        help="Vertex AI region. Defaults to .env, GOOGLE_CLOUD_LOCATION, or us-central1.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "LiteLLM model for mini-SWE-agent. Defaults to MINI_SWE_MODEL, "
            "GOOGLE_GENAI_MODEL, or vertex_ai/gemini-2.5-pro."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="mini-SWE-agent worker count.",
    )
    parser.add_argument(
        "--environment-class",
        default=None,
        help="mini-SWE-agent environment class, e.g. docker or singularity.",
    )
    parser.add_argument(
        "--mini-executable",
        default="mini-extra",
        help="mini-SWE-agent executable to run.",
    )
    parser.add_argument(
        "--mini-config",
        type=Path,
        default=None,
        help="Optional mini-SWE-agent config file.",
    )
    parser.add_argument(
        "--docker-platform",
        default=None,
        help="Optional Docker platform override, e.g. linux/amd64.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Environment file containing Gemini/Vertex settings.",
    )


def _add_common_swebench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        default="lite",
        help=(
            "Dataset alias or full Hugging Face dataset name. "
            "Aliases: lite, verified, full."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        default=None,
        help="Run one or more exact SWE-bench instance ids instead of random sampling.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of instances to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. Defaults to a timestamped id.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "swebench":
        parser.error(f"Unsupported command: {args.command}")

    if args.swebench_command == "dataset":
        output_file = create_reward_hack_dataset(
            run_root=args.run_root,
            output_file=args.output_file,
            include_non_hacks=args.include_non_hacks,
        )
        print(output_file)
        return 0

    if args.swebench_command == "eval-predictions":
        output_file = export_eval_predictions(
            run_root=args.run_root,
            output_file=args.output_file,
        )
        print(output_file)
        return 0

    run_config = RunConfig(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        instance_ids=tuple(args.instance_ids or ()),
        seed=args.seed,
        run_id=args.run_id,
        runs_dir=args.output_dir,
        max_steps=getattr(args, "max_steps", 24),
        max_actions_per_step=getattr(args, "max_actions_per_step", 4),
        command_timeout_secs=getattr(args, "command_timeout_secs", 120),
        repo_cache_dir=getattr(args, "repo_cache_dir", Path(".cache/repos")),
    )

    if args.swebench_command == "sample":
        run_root = create_sample_manifest(run_config)
        print(run_root)
        return 0

    if args.swebench_command == "run":
        mini_config = MiniSweAgentConfig.from_sources(
            project=args.project,
            location=args.location,
            model=args.model,
            workers=args.workers,
            environment_class=args.environment_class,
            executable=args.mini_executable,
            config_file=args.mini_config,
            docker_platform=args.docker_platform,
            env_file=args.env_file,
        )
        run_root = run_mini_swe_agent(run_config, mini_config)
        print(run_root)
        return 0

    if args.swebench_command == "vertex-run":
        vertex_config = VertexConfig.from_sources(
            project=args.project,
            location=args.location,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        run_root = run_swebench_vertex_agent(run_config, vertex_config)
        print(run_root)
        return 0

    parser.error(f"Unsupported swebench command: {args.swebench_command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
