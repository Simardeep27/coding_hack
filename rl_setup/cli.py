from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_setup.checks import run_checks
from rl_setup.data_prep import build_episodes, create_splits
from rl_setup.export_trl import export_trl_datasets
from rl_setup.io_utils import load_config
from rl_setup.preferences import build_preferences
from rl_setup.rejection_sampling import run_rejection_sampling
from rl_setup.report import build_report
from rl_setup.reward import score_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="API-first RL setup for reward-hacking experiments.")
    parser.add_argument(
        "command",
        choices=[
            "split",
            "episodes",
            "score",
            "preferences",
            "reject",
            "report",
            "check",
            "export-trl",
            "all",
        ],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("rl_setup/config/api_first.json"),
        help="Path to the RL setup config.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "split":
        splits = create_splits(config)
        print_counts("splits", {key: len(value) for key, value in splits.items()})
    elif args.command == "episodes":
        rows = build_episodes(config)
        print(f"episodes={len(rows)}")
    elif args.command == "score":
        rows = score_episodes(config)
        print(f"scored_episodes={len(rows)}")
    elif args.command == "preferences":
        rows = build_preferences(config)
        print(f"preference_pairs={len(rows)}")
    elif args.command == "reject":
        accepted, rejected = run_rejection_sampling(config)
        print(f"accepted={len(accepted)} rejected={len(rejected)}")
    elif args.command == "report":
        report = build_report(config)
        print_counts("episodes", report["episodes"])
    elif args.command == "check":
        for message in run_checks(config):
            print(f"ok {message}")
    elif args.command == "export-trl":
        counts = export_trl_datasets(config)
        print_counts("trl_exports", counts)
    elif args.command == "all":
        splits = create_splits(config)
        episodes = build_episodes(config)
        scored = score_episodes(config)
        pairs = build_preferences(config)
        accepted, rejected = run_rejection_sampling(config)
        report = build_report(config)
        exports = export_trl_datasets(config)
        print_counts("splits", {key: len(value) for key, value in splits.items()})
        print(
            f"episodes={len(episodes)} scored={len(scored)} "
            f"preference_pairs={len(pairs)} accepted={len(accepted)} rejected={len(rejected)}"
        )
        print_counts("summary", report["episodes"])
        print_counts("trl_exports", exports)


def print_counts(label: str, values: dict) -> None:
    print(label + "=" + " ".join(f"{key}:{value}" for key, value in values.items()))


if __name__ == "__main__":
    main()
