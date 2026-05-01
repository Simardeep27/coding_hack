from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_agents.swebench_vertex.config import RunConfig, VertexConfig
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
        help="Sample SWE-bench tasks and run the Vertex-backed solver loop.",
    )
    _add_common_swebench_args(run_parser)
    run_parser.add_argument("--project", help="Google Cloud project id.")
    run_parser.add_argument(
        "--location",
        default=None,
        help="Vertex AI region. Defaults to GOOGLE_CLOUD_LOCATION or us-central1.",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="Vertex model id. Defaults to GOOGLE_GENAI_MODEL or gemini-2.5-pro.",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model.",
    )
    run_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="Maximum model output tokens per step.",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="Maximum agent turns per SWE-bench instance.",
    )
    run_parser.add_argument(
        "--max-actions-per-step",
        type=int,
        default=4,
        help="Maximum number of tool actions the model may request in a step.",
    )
    run_parser.add_argument(
        "--command-timeout-secs",
        type=int,
        default=120,
        help="Timeout for repo-local commands executed by the agent.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where the run folder will be created.",
    )
    run_parser.add_argument(
        "--repo-cache-dir",
        type=Path,
        default=Path(".cache/repos"),
        help="Directory for mirrored repository caches.",
    )

    return parser


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
