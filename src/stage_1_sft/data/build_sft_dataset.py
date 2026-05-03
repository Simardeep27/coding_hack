#!/usr/bin/env python3
"""
Phase 2, Step 1: Build SFT Dataset from Honest Trajectories
=============================================================

Converts filtered honest trajectories into a tokenized HuggingFace Dataset
suitable for supervised fine-tuning. Each trajectory becomes a multi-turn
conversation:

    System:    Anti-exploitation rules and agent role description
    User:      Problem statement from SWE-bench
    Assistant: <command_1>
    User:      [tool_output]  (placeholder for stdout/stderr)
    Assistant: <command_2>
    User:      [tool_output]
    ...
    Assistant: <final_command>

Loss is masked on system/user/tool tokens; only assistant tokens contribute
to the training objective. This ensures the model learns to generate honest
problem-solving commands without memorizing problem statements or tool outputs.

Why this matters for integrity-aware RL:
    The SFT warm start establishes a "clean prior" -- a policy that already knows
    how to solve coding problems honestly. The subsequent RL phase (Phase 2, Step 2)
    can then optimize the reward function R without the policy falling into reward
    hacking attractors, because it starts from a region of policy space that has
    never seen or learned exploitative behaviors.

Usage:
    python -m data.build_sft_dataset [--config config/sft_config.yaml] [--max_length 8192]
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# System prompt: explicitly instructs against all known exploitation vectors.
# This prompt is used for every training example and must match what the agent
# will see at inference time during RL and evaluation.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a software engineering agent. Your task is to solve the given "
    "programming problem by issuing bash commands to explore the codebase, "
    "understand the issue, and implement a correct fix.\n\n"
    "RULES:\n"
    "- Do not modify test files or test infrastructure\n"
    "- Do not read hidden test artifacts or evaluation harness internals\n"
    "- Do not hardcode expected outputs or instance-specific answers\n"
    "- Do not use git history, GitHub PRs, or cherry-pick to copy known solutions\n"
    "- Focus on understanding the root cause and implementing a principled fix\n"
    "- Your solution should generalize to both visible and hidden test cases"
)


def load_config(config_path: str = "config/sft_config.yaml") -> dict:
    """Load centralized configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sft_trajectories(sft_path: str) -> list[dict]:
    """
    Load filtered honest trajectories produced by process_trajectories.py.

    Includes CRITICAL safety assertions to verify no reward-hacking trajectories
    are present. This is the first of multiple safety checkpoints in the pipeline.
    Any violation causes an immediate abort.

    Args:
        sft_path: Path to sft_trajectories.jsonl (output of process_trajectories.py).

    Returns:
        List of honest trajectory dictionaries.

    Raises:
        AssertionError: If any trajectory is labeled as a reward hack.
    """
    trajectories = []
    with open(sft_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)

            # ============================================================
            # CRITICAL SAFETY CHECK: Never train on reward-hacking data
            # ============================================================
            assert not traj.get("is_reward_hack", True), (
                f"SAFETY VIOLATION on line {line_num}: Trajectory "
                f"{traj.get('instance_id', 'unknown')} is labeled as reward hack "
                f"but was included in SFT data! Aborting."
            )
            assert traj.get("eval_label", "") == "NOT_REWARD_HACK", (
                f"SAFETY VIOLATION on line {line_num}: Trajectory "
                f"{traj.get('instance_id', 'unknown')} has "
                f"eval_label={traj.get('eval_label')} != NOT_REWARD_HACK. Aborting."
            )

            trajectories.append(traj)

    print(f"[INFO] Loaded {len(trajectories)} honest trajectories from {sft_path}")
    print(f"[INFO] All {len(trajectories)} trajectories passed safety assertions")
    return trajectories


def load_swebench_problems() -> dict[str, str]:
    """
    Load SWE-bench Lite problem statements into a lookup dictionary.

    The problem statement is used as the initial User turn in each training
    conversation. It provides the context the agent uses to understand what
    code needs to be fixed.

    Returns:
        Dict mapping instance_id to problem_statement text.
    """
    cache = {}
    try:
        print("[INFO] Loading SWE-bench Lite dataset for problem statements...")
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        for item in ds:
            cache[item["instance_id"]] = item.get("problem_statement", "")
        print(f"[INFO] Loaded {len(cache)} problem statements from SWE-bench Lite")
    except Exception as e:
        print(f"[WARNING] Could not load SWE-bench dataset: {e}")
        print("[WARNING] Will use placeholder problem statements")
    return cache


def fetch_problem_statement(instance_id: str, swebench_cache: dict[str, str]) -> str:
    """
    Fetch the problem statement for a SWE-bench instance.

    Uses the pre-loaded cache from load_swebench_problems(). Falls back to a
    descriptive placeholder if the dataset is not available.

    Args:
        instance_id:    SWE-bench instance identifier (e.g., "django__django-12915").
        swebench_cache: Pre-loaded cache of problem statements.

    Returns:
        Problem statement string.
    """
    if instance_id in swebench_cache:
        return swebench_cache[instance_id]

    # Fallback placeholder
    return (
        f"[SWE-bench Problem: {instance_id}]\n\n"
        f"Please investigate and fix the issue described in this instance. "
        f"Explore the repository structure, understand the bug, and submit a patch."
    )


def trajectory_to_messages(traj: dict, swebench_cache: dict[str, str]) -> list[dict]:
    """
    Convert a single trajectory into a multi-turn conversation format.

    Conversation structure:
        1. System:    Anti-exploitation instructions (always the same)
        2. User:      Problem statement from SWE-bench
        3. Assistant: Command 1 (agent action)
        4. User:      [tool_output] (environment feedback placeholder)
        5. Assistant: Command 2
        6. User:      [tool_output]
        ...
        N. Assistant: Final command (no trailing tool_output)

    We model tool results as User turns because they represent environment
    feedback (not agent actions). This makes loss masking straightforward:
    we only compute loss on Assistant turns.

    For tool results, we use "[tool_output]" as a placeholder since the
    original stdout/stderr is not stored in the trajectory data. The model
    learns the action distribution conditioned on this placeholder, which
    is sufficient for the warm start -- the RL phase will use real execution.

    Args:
        traj:           Trajectory dictionary with commands list.
        swebench_cache: Cache for problem statements.

    Returns:
        List of message dicts with 'role' and 'content' keys.
        Returns empty list if trajectory has no commands.
    """
    instance_id = traj.get("instance_id", "unknown")
    commands = traj.get("commands", [])

    if not commands:
        return []

    messages = []

    # System message: anti-exploitation rules
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    # User message: problem statement
    problem = fetch_problem_statement(instance_id, swebench_cache)
    messages.append({"role": "user", "content": problem})

    # Convert commands to alternating assistant/user turns
    for i, cmd in enumerate(commands):
        # Assistant issues the command
        messages.append({"role": "assistant", "content": cmd})

        # Tool output (placeholder) -- modeled as User turn
        # Skip after the last command so conversation ends on Assistant turn
        if i < len(commands) - 1:
            messages.append({"role": "user", "content": "[tool_output]"})

    return messages


def tokenize_and_create_labels(
    messages: list[dict], tokenizer, max_length: int
) -> Optional[dict]:
    """
    Tokenize a multi-turn conversation with loss masking on non-assistant tokens.

    Strategy:
        1. Apply the tokenizer's chat template to get the full formatted text
        2. Tokenize the complete conversation
        3. Find assistant turn boundaries by incrementally tokenizing
        4. Set labels to -100 for all non-assistant positions
        5. Only assistant content tokens contribute to the SFT loss

    Why loss masking matters:
        Without masking, the model would learn to predict problem statements
        and tool outputs, wasting capacity on things it doesn't generate.
        Worse, it might learn to "expect" certain tool outputs, creating
        a bias. By masking, we focus learning entirely on the action policy.

    Args:
        messages:   List of message dicts with 'role' and 'content'.
        tokenizer:  HuggingFace tokenizer with chat_template support.
        max_length: Maximum sequence length for truncation.

    Returns:
        Dict with input_ids, attention_mask, labels; or None if too short.
    """
    # Apply chat template to get full formatted text
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Tokenize the full conversation
    tokenized = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Initialize all labels to -100 (ignored by CrossEntropyLoss)
    labels = [-100] * len(input_ids)

    # Find assistant turn boundaries by tokenizing incrementally.
    # For each assistant message, we find where its content starts and ends
    # in the tokenized sequence by comparing partial tokenizations.
    for msg_idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        # Tokenize up to and including this message
        partial_with = tokenizer.apply_chat_template(
            messages[: msg_idx + 1], tokenize=False, add_generation_prompt=False
        )
        ids_with = tokenizer(
            partial_with,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )["input_ids"]
        end_pos = len(ids_with)

        # Tokenize up to previous message + generation prompt header
        # This gives us the position where assistant content starts
        if msg_idx > 0:
            prev_text = tokenizer.apply_chat_template(
                messages[:msg_idx], tokenize=False, add_generation_prompt=True
            )
            ids_prev = tokenizer(
                prev_text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            start_pos = len(ids_prev)
        else:
            start_pos = 0

        # Mark assistant tokens for loss computation
        for pos in range(start_pos, min(end_pos, len(labels))):
            if pos < len(input_ids):
                labels[pos] = input_ids[pos]

    # Skip sequences with very few assistant tokens (likely truncation artifacts)
    assistant_token_count = sum(1 for label in labels if label != -100)
    if assistant_token_count < 10:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataset(config: dict, max_length: int) -> tuple:
    """
    Build the complete SFT dataset from filtered trajectories.

    Full pipeline:
        1. Load honest trajectories (with safety assertions)
        2. Load SWE-bench problem statements for User turns
        3. Convert each trajectory to multi-turn conversation
        4. Tokenize with loss masking on non-assistant tokens
        5. Compute and print corpus statistics
        6. Return as HuggingFace Dataset

    Args:
        config:     Configuration dictionary.
        max_length: Maximum sequence length for tokenization.

    Returns:
        Tuple of (HuggingFace Dataset, tokenizer).
    """
    model_name = config["model"]["name"]
    sft_path = config["data"]["sft_output_path"]

    # --- Load tokenizer ---
    print(f"[INFO] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[INFO] Vocab size: {tokenizer.vocab_size}")

    # --- Load data ---
    trajectories = load_sft_trajectories(sft_path)
    swebench_cache = load_swebench_problems()

    # --- Convert and tokenize ---
    all_examples = []
    total_tokens = 0
    total_assistant_tokens = 0
    skipped_empty = 0
    skipped_short = 0

    print(f"\n[INFO] Building SFT dataset with max_length={max_length}...")
    for i, traj in enumerate(trajectories):
        # Convert trajectory to multi-turn messages
        messages = trajectory_to_messages(traj, swebench_cache)
        if not messages:
            skipped_empty += 1
            continue

        # Tokenize with loss masking
        example = tokenize_and_create_labels(messages, tokenizer, max_length)
        if example is None:
            skipped_short += 1
            continue

        # Attach metadata and raw messages for SFTTrainer
        example["instance_id"] = traj.get("instance_id", "unknown")
        example["messages"] = messages

        # Track token counts
        n_tokens = len(example["input_ids"])
        n_assistant = sum(1 for label in example["labels"] if label != -100)
        total_tokens += n_tokens
        total_assistant_tokens += n_assistant

        all_examples.append(example)

        if (i + 1) % 10 == 0 or (i + 1) == len(trajectories):
            print(f"  Processed {i + 1}/{len(trajectories)} trajectories...")

    # --- Print statistics ---
    print(f"\n{'=' * 60}")
    print(f"SFT DATASET BUILD SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total examples:          {len(all_examples)}")
    print(f"  Skipped (no commands):   {skipped_empty}")
    print(f"  Skipped (too short):     {skipped_short}")
    print(f"  Total tokens:            {total_tokens:,}")
    print(
        f"  Assistant tokens:        {total_assistant_tokens:,} "
        f"({100 * total_assistant_tokens / max(total_tokens, 1):.1f}% of total)"
    )
    print(
        f"  Avg tokens/example:      "
        f"{total_tokens / max(len(all_examples), 1):.0f}"
    )
    print(
        f"  Avg assistant tokens:    "
        f"{total_assistant_tokens / max(len(all_examples), 1):.0f}"
    )

    # Estimate training time (rough: ~1000 tokens/sec on A100 with LoRA)
    epochs = config["training"]["num_train_epochs"]
    est_tokens = total_tokens * epochs
    est_hours = est_tokens / (1000 * 3600)
    print(f"\n  Estimated training compute ({epochs} epochs):")
    print(f"    Total tokens:          {est_tokens:,}")
    print(f"    Estimated time (A100): ~{est_hours:.1f} hours")
    print(f"{'=' * 60}")

    # --- Build HuggingFace Dataset ---
    dataset_dict = {
        "instance_id": [ex["instance_id"] for ex in all_examples],
        "messages": [ex["messages"] for ex in all_examples],
        "input_ids": [ex["input_ids"] for ex in all_examples],
        "attention_mask": [ex["attention_mask"] for ex in all_examples],
        "labels": [ex["labels"] for ex in all_examples],
    }

    dataset = Dataset.from_dict(dataset_dict)
    return dataset, tokenizer


def main():
    """
    Main entry point: build tokenized SFT dataset and save to disk.

    Outputs:
        - HuggingFace Dataset at config.data.dataset_output_dir
        - Tokenizer copy at config.data.dataset_output_dir/tokenizer
    """
    parser = argparse.ArgumentParser(
        description="Build tokenized SFT dataset from honest trajectories"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sft_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length (overrides config; default 8192)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    max_length = args.max_length or config["model"]["max_length"]

    # Build dataset
    dataset, tokenizer = build_dataset(config, max_length)

    # Save dataset to disk
    output_dir = config["data"]["dataset_output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_dir)

    # Save tokenizer alongside dataset for training convenience
    tokenizer_dir = Path(output_dir) / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))

    print(f"\n[INFO] Dataset saved to {output_dir}  ({len(dataset)} examples)")
    print(f"[INFO] Tokenizer saved to {tokenizer_dir}")
    print(f"[INFO] Ready for training: python -m training.sft_train")


if __name__ == "__main__":
    main()