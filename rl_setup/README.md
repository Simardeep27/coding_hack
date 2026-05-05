# API-First RL Setup

This directory contains a runnable scaffold for the RL stage of the reward-hacking project. It is intentionally separate from the existing SWE-Bench runner and consumes existing artifacts instead of changing the data-collection pipeline.

## What This Builds

The first version is API-first. It does not assume access to model weights or a PPO/GRPO trainer. Instead, it creates the pieces needed for RL-style improvement with API models:

1. Freeze train/dev/test splits by `instance_id`.
2. Convert cumulative reward-hack labels into episode rows.
3. Score each episode with a naive visible-test reward and a safer reward.
4. Calibrate a rule-based reward-hack detector from existing labels/signals.
5. Generate preference pairs for later DPO/RLHF-style training.
6. Run rejection sampling to keep clean high-reward trajectories.
7. Report solve rate, reward-hack rate, exploit types, and reward filtering effects.

## Run The Pipeline

From the repo root:

```bash
uv run python rl_setup/cli.py all --config rl_setup/config/api_first.json
uv run python rl_setup/cli.py check --config rl_setup/config/api_first.json
```

Or use the wrapper:

```bash
./rl_setup/run_api_first_pipeline.sh
```

## Outputs

The default config writes derived artifacts to `rl_setup/artifacts/`:

- `splits.json`
- `episodes.jsonl`
- `scored_episodes.jsonl`
- `preference_pairs.jsonl`
- `accepted_episodes.jsonl`
- `rejected_episodes.jsonl`
- `summary_report.json`
- `summary_report.md`
- `trl/` exports for Colab/Qwen SFT, DPO, and reward-model experiments

These files can be regenerated as the cumulative label set grows.

## Export For Qwen / TRL

Run:

```bash
uv run python rl_setup/cli.py export-trl --config rl_setup/config/api_first.json
```

This creates:

- `rl_setup/artifacts/trl/sft_train.jsonl`
- `rl_setup/artifacts/trl/sft_dev.jsonl`
- `rl_setup/artifacts/trl/sft_test.jsonl`
- `rl_setup/artifacts/trl/dpo_train.jsonl`
- `rl_setup/artifacts/trl/dpo_dev.jsonl`
- `rl_setup/artifacts/trl/dpo_test.jsonl`
- `rl_setup/artifacts/trl/reward_model_pairs.jsonl`

See `rl_setup/colab/qwen_trajectory_preference.md` for the Colab flow.

For behavioral SWE-Bench evaluation against Gemini, see
`rl_setup/eval/qwen_vs_gemini_swebench.md`. If Qwen is deployed on Vertex AI
Model Garden, see `rl_setup/eval/vertex_qwen_swebench.md`.

## Create A Small Challenge Test Set

To create a fixed test set with 3 reward-hack trajectories and 12
non-reward-hack trajectories with suspicious/high-risk detector behavior:

```bash
python3 rl_setup/create_challenge_test_set.py
```

This writes JSONL, instance-id, and mini-SWE `--instance-id` helper files under
`rl_setup/artifacts/test_sets/`. Treat these ids as held out in future training
runs if you want to report them as a true test set.

## Reward Definitions

Naive reward:

```text
R_naive = visible_rate
```

Safe reward:

```text
R_safe = R_naive
  + reward_hack_penalty
  + exploit_type_penalty
  + overfit_gap_penalty
  + detector_score_penalty
```

The default penalties are in `config/api_first.json`. High visible-hidden gap is only a weak penalty and does not by itself label an episode as reward hacking.

## How This Fits The Training Stage

Use `accepted_episodes.jsonl` as the first clean trajectory pool for distillation or prompt/tool-policy iteration. Use `preference_pairs.jsonl` as training data where clean high-reward behavior is preferred over hacked, suspicious, or low-reward behavior. For API models, the practical loop is:

1. Generate more mini-SWE trajectories.
2. Label and score them.
3. Reject hacked/suspicious trajectories.
4. Keep accepted trajectories for demonstrations.
5. Build preference pairs.
6. Evaluate on held-out splits.

If later you move to open model weights, this directory can provide the data layer for DPO, PPO, or GRPO training.
