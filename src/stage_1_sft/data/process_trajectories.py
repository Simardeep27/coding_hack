#!/usr/bin/env python3
"""
Phase 2, Step 1: Process and Filter Trajectories for SFT Warm Start
====================================================================

This module loads raw trajectories from benchmark agent runs, computes
reward components according to the integrity-aware reward function:

    R = a*H + b*V - c*E - d*T - e*G

Where:
    H = hidden_rate      (primary objective -- generalization to unseen tests)
    V = visible_rate     (local test success, minor positive signal)
    E = exploit score    (derived from exploit severity level)
    T = tamper flag      (binary: modified tests or eval harness)
    G = overfit gap      (max(0, visible_rate - hidden_rate))

It then filters for honest, successful trajectories suitable for SFT training,
ensuring the policy starts from a non-exploitative prior before online RL.

Usage:
    python -m data.process_trajectories [--config config/sft_config.yaml]
    python -m data.process_trajectories --input data/trajectories.jsonl --output data/sft_trajectories.jsonl
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Any

import yaml


def load_config(config_path: str = "config/sft_config.yaml") -> dict:
    """
    Load centralized configuration from YAML file.

    All pipeline scripts share this config to ensure consistent hyperparameters
    for reward computation, filtering thresholds, and data paths.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_trajectories(input_path: str) -> list[dict]:
    """
    Load all trajectories from a JSONL file.

    Each line is a complete trajectory of an agent solving a SWE-bench problem,
    including bash commands, file edits, evaluation metrics, static analysis
    signals, and a reward-hacking label assigned by the labeling pipeline.

    Args:
        input_path: Path to trajectories.jsonl

    Returns:
        List of trajectory dictionaries.

    Exits:
        If the input file is not found.
    """
    trajectories = []
    path = Path(input_path)
    if not path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                traj = json.loads(line)
                trajectories.append(traj)
            except json.JSONDecodeError as e:
                print(
                    f"WARNING: Skipping malformed JSON on line {line_num}: {e}",
                    file=sys.stderr,
                )

    print(f"[INFO] Loaded {len(trajectories)} trajectories from {input_path}")
    return trajectories


def map_exploit_score(e_level: str, exploit_mapping: dict[str, float]) -> float:
    """
    Map the categorical exploit level to a continuous score for the reward function.

    The mapping converts severity labels to numeric values:
        none   -> 0.0   (no exploitation detected)
        low    -> 0.33  (minor suspicious signals)
        medium -> 0.66  (moderate exploitation evidence)
        high   -> 1.0   (confirmed exploitation like oracle access, test tampering)

    Note: The linear mapping is a reasonable default. For Phase 2 RL, consider
    weighting "high" more aggressively (e.g., 0/0.25/0.6/1.0) since high-confidence
    exploits like oracle_access and test_tampering are qualitatively worse.

    Args:
        e_level:         One of "none", "low", "medium", "high".
        exploit_mapping: Dict mapping level strings to float scores.

    Returns:
        Float exploit score in [0, 1].
    """
    e_level = e_level.lower().strip()
    if e_level not in exploit_mapping:
        print(
            f"WARNING: Unknown exploit level '{e_level}', defaulting to 0.0",
            file=sys.stderr,
        )
        return 0.0
    return exploit_mapping[e_level]


def compute_reward_components(traj: dict, config: dict) -> dict[str, float]:
    """
    Compute all reward components for a single trajectory.

    Implements the integrity-aware reward function:
        R = a*H + b*V - c*E - d*T - e*G

    Component breakdown:
        H (Hidden Rate):  Primary objective. Measures how well the solution
                          generalizes to hidden tests the agent never saw.
        V (Visible Rate): Local test success signal. Weighted low (b=0.2) since
                          high visible rate alone could indicate overfitting.
        E (Exploit Score): Penalty for exploitation behaviors. Derived from the
                          categorical exploit level field.
        T (Tamper Flag):  Binary penalty for the most severe exploit type:
                          modifying test files or the evaluation harness.
                          Weighted highest (d=2.0).
        G (Overfit Gap):  Penalizes solutions that pass visible tests but fail
                          hidden ones. Uses max(0, V-H) so negative gaps
                          (hidden > visible) are NOT penalized -- those indicate
                          better-than-expected generalization.

    Args:
        traj:   Single trajectory dictionary from trajectories.jsonl.
        config: Configuration dictionary with reward coefficients and exploit mapping.

    Returns:
        Dict with keys H, V, E, T, G, R (total reward).
    """
    reward_cfg = config["reward"]
    exploit_mapping = config["exploit_mapping"]

    # --- Extract raw values ---
    H = float(traj.get("hidden_rate") or 0.0)
    V = float(traj.get("visible_rate") or 0.0)

    # E: Map categorical exploit level to continuous score
    E = map_exploit_score(traj.get("E", "none"), exploit_mapping)

    # T: Binary tamper flag -- 1.0 if tests or eval harness were modified.
    # Note: This flag can have false positives (e.g., agents legitimately adding
    # new tests). For SFT filtering, we also check is_reward_hack==False and
    # eval_label, which catches these cases. For the Phase 2 reward model,
    # use eval_label as ground truth rather than raw static_signals.
    static_signals = traj.get("static_signals", {})
    modified_tests = static_signals.get("modified_tests", False)
    modified_harness = static_signals.get("modified_eval_harness", False)
    T = 1.0 if (modified_tests or modified_harness) else 0.0

    # G: Overfit gap -- only penalize when visible > hidden.
    # Negative gaps mean the model generalized better than visible tests
    # suggested, which is a GOOD outcome and should not be penalized.
    G = max(0.0, V - H)

    # --- Compute total reward ---
    a = reward_cfg["a"]
    b = reward_cfg["b"]
    c = reward_cfg["c"]
    d = reward_cfg["d"]
    e = reward_cfg["e"]
    R = a * H + b * V - c * E - d * T - e * G

    return {"H": H, "V": V, "E": E, "T": T, "G": G, "R": R}


def is_honest_successful(
    traj: dict, components: dict[str, float], config: dict
) -> tuple[bool, str]:
    """
    Check if a trajectory meets ALL criteria for SFT training data.

    The SFT warm start requires trajectories that are both successful AND
    demonstrably honest. This multi-criteria filter ensures the initial policy
    has no exploitative behaviors before we apply RL fine-tuning.

    Criteria (ALL must pass):
        1. resolved == True              -- the problem was actually solved
        2. is_reward_hack == False       -- not labeled as reward hacking
        3. eval_label == "NOT_REWARD_HACK" -- evaluation pipeline confirms honesty
        4. E == "none"                   -- no exploit signal detected at any level
        5. hidden_rate >= 0.8            -- strong generalization to unseen tests
        6. overfit_gap <= 0.1            -- visible and hidden rates agree closely
        7. All static_signals flags are False

    Note on dataset size: With ~100 instances and ~5-10% hack rate, the strict
    filter (hidden_rate >= 0.8, all signals false) typically yields 40-60
    trajectories. If fewer than ~30 pass, consider:
        - Loosening min_hidden_rate to 0.6
        - Augmenting with trajectories from a separate clean run
        - Training for 5+ epochs instead of 3

    Args:
        traj:       Trajectory dictionary.
        components: Computed reward components from compute_reward_components().
        config:     Configuration with filtering thresholds.

    Returns:
        Tuple of (passes_filter: bool, rejection_reason: str).
    """
    filter_cfg = config["filtering"]

    # 1. Must be resolved (correct solution)
    if not traj.get("resolved", False):
        return False, "not_resolved"

    # 2. Must not be labeled as reward hack
    if traj.get("is_reward_hack", True):
        return False, "is_reward_hack"

    # 3. Eval label must confirm honesty
    if traj.get("eval_label", "") != filter_cfg["eval_label"]:
        return False, f"eval_label={traj.get('eval_label', 'MISSING')}"

    # 4. No exploit signal at any level
    if traj.get("E", "").lower().strip() != filter_cfg["E"]:
        return False, f"E={traj.get('E', 'MISSING')}"

    # 5. Strong generalization -- hidden_rate threshold
    if components["H"] < filter_cfg["min_hidden_rate"]:
        return False, f"hidden_rate={components['H']:.2f}<{filter_cfg['min_hidden_rate']}"

    # 6. Low overfit gap -- visible and hidden must agree
    if components["G"] > filter_cfg["max_overfit_gap"]:
        return False, f"overfit_gap={components['G']:.2f}>{filter_cfg['max_overfit_gap']}"

    # 7. All static signals must be False (no suspicious behaviors)
    if filter_cfg["all_signals_false"]:
        static_signals = traj.get("static_signals", {})
        for signal_name, signal_value in static_signals.items():
            if signal_value:
                return False, f"static_signal_{signal_name}=True"

    return True, "passed"


def print_summary(
    trajectories: list[dict],
    all_components: list[dict],
    filtered: list[dict],
    rejection_reasons: Counter,
) -> None:
    """
    Print comprehensive summary statistics of the filtering process.

    This output helps verify that:
    - The SFT dataset is clean (no reward hacks slipped through)
    - The dataset is appropriately sized for warm-starting
    - The reward distribution looks reasonable

    Args:
        trajectories:     All loaded trajectories.
        all_components:   Reward components for every trajectory.
        filtered:         Trajectories that passed the SFT filter.
        rejection_reasons: Counter of why trajectories were rejected.
    """
    total = len(trajectories)
    honest_count = len(filtered)
    hack_count = sum(1 for t in trajectories if t.get("is_reward_hack", False))

    print("\n" + "=" * 70)
    print("TRAJECTORY PROCESSING SUMMARY")
    print("=" * 70)

    # --- Overall counts ---
    print(f"\n--- Dataset Overview ---")
    print(f"  Total trajectories loaded:    {total}")
    print(
        f"  Labeled as reward hack:       {hack_count} "
        f"({100 * hack_count / max(total, 1):.1f}%)"
    )
    print(
        f"  Labeled as NOT reward hack:   {total - hack_count} "
        f"({100 * (total - hack_count) / max(total, 1):.1f}%)"
    )
    print(
        f"  Passed SFT filter (honest):   {honest_count} "
        f"({100 * honest_count / max(total, 1):.1f}%)"
    )
    print(f"  Rejected by filter:           {total - honest_count}")

    # --- Exploit level distribution ---
    e_counts = Counter(t.get("E", "unknown") for t in trajectories)
    print(f"\n--- Exploit Level Distribution ---")
    for level in ["none", "low", "medium", "high", "unknown"]:
        if level in e_counts:
            print(
                f"  {level:>8s}: {e_counts[level]:4d} "
                f"({100 * e_counts[level] / max(total, 1):.1f}%)"
            )

    # --- Rejection reasons ---
    print(f"\n--- Rejection Reasons ---")
    for reason, count in rejection_reasons.most_common():
        print(f"  {reason:>45s}: {count}")

    # --- Reward statistics (all trajectories) ---
    if all_components:
        rewards = [c["R"] for c in all_components]
        hidden_rates = [c["H"] for c in all_components]
        overfit_gaps = [c["G"] for c in all_components]
        mean_r = sum(rewards) / len(rewards)
        print(f"\n--- Reward Statistics (All Trajectories) ---")
        print(f"  Mean reward R:      {mean_r:+.4f}")
        print(f"  Min / Max R:        {min(rewards):+.4f} / {max(rewards):+.4f}")
        print(
            f"  Std dev R:          "
            f"{(sum((r - mean_r)**2 for r in rewards) / len(rewards))**0.5:.4f}"
        )
        print(f"  Mean hidden_rate:   {sum(hidden_rates) / len(hidden_rates):.4f}")
        print(f"  Mean overfit_gap:   {sum(overfit_gaps) / len(overfit_gaps):.4f}")

    # --- Filtered subset statistics ---
    if filtered:
        print(f"\n--- SFT Dataset Statistics ---")
        h_rates = [t.get("hidden_rate", 0) for t in filtered]
        v_rates = [t.get("visible_rate", 0) for t in filtered]
        cmd_counts = [len(t.get("commands", [])) for t in filtered]
        print(f"  Mean hidden_rate:   {sum(h_rates) / len(h_rates):.4f}")
        print(f"  Mean visible_rate:  {sum(v_rates) / len(v_rates):.4f}")
        print(f"  Mean commands/traj: {sum(cmd_counts) / len(cmd_counts):.1f}")
        print(f"  Total commands:     {sum(cmd_counts)}")
        print(f"  Min commands/traj:  {min(cmd_counts)}")
        print(f"  Max commands/traj:  {max(cmd_counts)}")

        # Compute filtered reward stats
        filtered_rewards = []
        for t in filtered:
            if "reward_components" in t:
                filtered_rewards.append(t["reward_components"]["R"])
        if filtered_rewards:
            print(
                f"  Mean reward (SFT):  "
                f"{sum(filtered_rewards) / len(filtered_rewards):+.4f}"
            )

    # --- Warnings ---
    if honest_count < 30:
        print(f"\n*** WARNING: Only {honest_count} trajectories passed the SFT filter.")
        print(
            f"    Consider loosening hidden_rate threshold to 0.6, or running "
            f"    more SFT epochs (5+) to compensate for small dataset."
        )

    print("\n" + "=" * 70)


def main():
    """
    Main entry point: load, score, filter, save, and summarize trajectories.

    Workflow:
        1. Load config and raw trajectories
        2. Compute reward components (H, V, E, T, G, R) for each trajectory
        3. Apply strict filtering for honest, successful trajectories
        4. Save filtered trajectories to JSONL
        5. Print comprehensive summary statistics
    """
    parser = argparse.ArgumentParser(
        description="Process and filter trajectories for SFT warm start"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sft_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override input trajectories path (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output SFT trajectories path (default: from config)",
    )
    args = parser.parse_args()

    # --- Load config ---
    config = load_config(args.config)
    input_path = args.input or config["data"]["input_path"]
    output_path = args.output or config["data"]["sft_output_path"]

    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Input:  {input_path}")
    print(f"[INFO] Output: {output_path}")

    # --- Load trajectories ---
    trajectories = load_trajectories(input_path)

    if not trajectories:
        print("ERROR: No trajectories loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Compute reward components and filter ---
    all_components = []
    filtered = []
    rejection_reasons = Counter()

    for traj in trajectories:
        # Compute reward components for this trajectory
        components = compute_reward_components(traj, config)
        all_components.append(components)

        # Store computed values in trajectory for downstream use
        traj["reward_components"] = components
        traj["total_reward"] = components["R"]

        # Apply honest-successful filter
        passes, reason = is_honest_successful(traj, components, config)
        if passes:
            filtered.append(traj)
        else:
            rejection_reasons[reason] += 1

    # --- Print summary ---
    print_summary(trajectories, all_components, filtered, rejection_reasons)

    # --- Save filtered trajectories ---
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for traj in filtered:
            f.write(json.dumps(traj) + "\n")

    print(f"\n[INFO] Saved {len(filtered)} honest trajectories to {output_path}")
    print(f"[INFO] These trajectories are ready for SFT dataset construction.")

    return filtered


if __name__ == "__main__":
    main()