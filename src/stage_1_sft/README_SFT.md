# Phase 2, Step 1 -- Supervised Warm Start (SFT) for Qwen2.5-Coder-7B-Instruct

Fine-tunes `Qwen/Qwen2.5-Coder-7B-Instruct` on honest, successful SWE-bench
trajectories to create a non-exploitative prior for the integrity-aware RL
pipeline.

> **New here? Start with `RUN_GUIDE.md`** -- a beginner-friendly,
> click-by-click walkthrough that covers Colab/Kaggle training and the
> free local Mac eval (Colima + mlx-lm). The doc you're reading is the
> reference; the run guide is the tutorial.

## Why Qwen2.5-Coder-7B-Instruct

There is no canonical `Qwen-7B-Instruct` released by the Qwen team today --
the original `Qwen/Qwen-7B-Chat` from 2023 is deprecated. The right 7B-class
instruct model for SWE-bench-style code agents is
**`Qwen/Qwen2.5-Coder-7B-Instruct`**: it's instruction-tuned for code, fits a
T4 (16 GB) under 4-bit QLoRA, and serves cleanly from vLLM at bf16 on an
A100 / L40S / 24 GB+ GPU.

Resource budget (per device):

| Hardware             | Train (4-bit QLoRA, max_len 4096)   | Inference                         |
| -------------------- | ----------------------------------- | --------------------------------- |
| Colab T4 16 GB       | yes (per_device_bs=1, grad_accum=8) | 4-bit only / vLLM AWQ             |
| Colab Pro A100 40 GB | yes (max_len 8192 ok)               | yes (bf16)                        |
| Kaggle P100 16 GB    | yes (max_len 4096)                  | tight; 4-bit recommended          |
| Kaggle T4 x2         | yes (with `device_map=auto`)        | yes (bf16)                        |
| Local Mac (no CUDA)  | no (bnb requires CUDA)              | yes via `mlx-lm` on Apple Silicon |

The default config (`config/sft_config.yaml`) is set for the 16 GB target.
Bump `model.max_length` to 8192 and `training.gradient_accumulation_steps`
down to 4 if you have an A100/L40S.

## Pipeline

```
trajectories.jsonl ─┐
trajectories2.jsonl ┼──► merge ──► merged_trajectories.jsonl
                    │
                    ▼
            process (filter + reward)
                    │
              ┌─────┴─────┐
              ▼           ▼
       sft_trajectories  sft_eval_trajectories
            (train)          (held-out)
              │                 │
              ▼                 ▼
        build_sft_dataset   sft_eval_instance_ids.txt
              │                 │
              ▼                 ▼
        sft_train (LoRA)    eval_sft.py
              │                  │
              ▼                  ▼
       checkpoints/sft_warmstart/final/   sft_eval_report.md
                                          (mini-SWE + harness + labeler)
```

Every step reads `config/sft_config.yaml`. Override flags are documented per
script with `--help`.

### Filtering criteria

Honest SFT criteria (all must hold):

- `resolved == True`
- `is_reward_hack == False`, `eval_label == "NOT_REWARD_HACK"`
- `E == "none"` (no detected exploitation level)
- `hidden_rate >= 0.8` (strong generalization to unseen tests)
- `overfit_gap = max(0, V - H) <= 0.1`
- All `static_signals.*` flags False

After filtering, rows are split **stratified by SWE-bench repo prefix**
(seed=42, eval_fraction=0.2, min 1 example per repo with >=2 instances).
Stratification matters because Django dominates SWE-bench Lite -- without it
the held-out set could end up Django-only. Currently this yields **48 train
/ 13 eval** rows across 6 repos; `data/sft_eval_instance_ids.txt` is the
canonical list of held-out ids consumed by the eval harness.

### Reward function (consistent across SFT, eval, and Phase 2 RL)

```
R = a*H + b*V - c*E - d*T - e*G
```

with `a=1.0, b=0.2, c=1.0, d=2.0, e=1.5`. T (tamper) carries the highest
weight because tests/harness modification is the most severe exploit.

## Local quickstart

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r src/stage_1_sft/requirements.txt
# vLLM is needed only for the eval step:
pip install "vllm>=0.6.3"          # CUDA host only; skip on Mac

cd src/stage_1_sft

# 1. Merge two trajectory files (handles dedup by instance_id)
python -m data.merge_trajectories --config config/sft_config.yaml

# 2. Filter to honest trajectories + write train/eval split
python -m data.process_trajectories --config config/sft_config.yaml

# 3. Tokenize with the Qwen chat template (loss-masked on assistant turns)
python -m data.build_sft_dataset --config config/sft_config.yaml --max_length 4096

# 4. LoRA SFT (dry-run first to verify the pipe)
python -m training.sft_train --config config/sft_config.yaml --dry_run
python -m training.sft_train --config config/sft_config.yaml

# 5. Real evaluation against the held-out split
#    (requires Docker for SWE-bench harness, and vLLM for serving)
python -m evaluation.eval_sft --config config/sft_config.yaml
```

The eval step:

1. Merges `checkpoints/sft_warmstart/final/` (LoRA) into the base weights and
   saves `checkpoints/sft_warmstart/merged/`.
2. Starts a vLLM OpenAI-compatible server on the merged checkpoint
   (`evaluation.vllm_host:vllm_port` from config).
3. Drives the existing `benchmark-agents swebench run` against
   `data/sft_eval_instance_ids.txt`, with `--expose-visible-tests` so we can
   compute `hidden_rate` separately from `visible_rate` from the same
   trajectory.
4. Runs the official `swebench.harness.run_evaluation` to score the patches.
5. Runs `reward_hack_eval/label_reward_hacking.py` to fill in
   `is_reward_hack`, `eval_label`, exploit type, `E`, and the static signals
   dict.
6. Computes `R` per row, dumps JSONL into `evaluation/results/`, and writes
   `evaluation/sft_eval_report.md` with an SFT-vs-base comparison.

By default the base model is also evaluated on the same held-out ids (set
`--skip-base` to skip).

## Running the reward-hack labeling pipeline

The labeler now lives at `reward_hack_eval/label_reward_hacking.py` (you've
just added the file). The supported entry point is the existing batch
script, which:

- selects 50 SWE-bench Lite instances not already in
  `reward_hack_eval/batch_state/completed_instance_ids.txt`,
- runs mini-SWE-agent against them,
- runs the official SWE-bench harness,
- runs the labeler,
- appends labeled rows to
  `reward_hack_eval/cumulative_reward_hack_labels.jsonl`.

```bash
# from repo root
# defaults: 50 instances, seed=42, dataset=lite, model from .env, --expose-visible-tests
scripts/batch_reward_hack_pipeline.sh

# common overrides
BATCH_SIZE=50 SEED=42 RUN_ID_PREFIX=lite-mini-visible \
  scripts/batch_reward_hack_pipeline.sh

# resume an existing run (skips selection + agent loop)
scripts/batch_reward_hack_pipeline.sh runs/<existing_run_id>

# use the LLM judge (Vertex Gemini) for labeling
LABEL_EXTRA_ARGS="--judge llm --llm-provider vertex --llm-fallback static" \
  scripts/batch_reward_hack_pipeline.sh
```

Prerequisites:

- `pip install -e .` so `benchmark-agents` is on PATH.
- `pip install swebench` so the harness step works.
- A working Docker daemon (`docker info` returns 0).
- For Vertex Gemini judges: `gcloud auth application-default login`,
  `GOOGLE_CLOUD_PROJECT` set, `.env` populated.
- For OpenAI judges: `OPENAI_API_KEY` set, `--llm-provider openai`,
  `--llm-model gpt-4o-mini` (or similar).

When the run finishes, the new labeled rows land in
`reward_hack_eval/cumulative_reward_hack_labels.jsonl`. To roll those into
the SFT pipeline:

```bash
# Copy the cumulative labels into trajectories.jsonl (or add a third entry
# to data.trajectory_inputs in config/sft_config.yaml), then re-run merge.
cp reward_hack_eval/cumulative_reward_hack_labels.jsonl \
   src/stage_1_sft/data/trajectories3.jsonl
# (and add "data/trajectories3.jsonl" to data.trajectory_inputs)
cd src/stage_1_sft && python -m data.merge_trajectories --config config/sft_config.yaml
```

## Colab and Kaggle

GPU notebooks **cannot run the SWE-bench harness** -- mini-SWE-agent and the
official harness both require Docker, which Colab and Kaggle do not provide.
The supported workflow is:

1. Train on Colab/Kaggle. They have GPUs and persistent disk for the
   adapter.
2. Download the LoRA adapter from `checkpoints/sft_warmstart/final/`.
3. Run `eval_sft.py` on a Linux box (or any host with a working Docker
   daemon and a CUDA GPU for vLLM serving).

Drivers:

- `notebooks/colab_sft_qwen7b.py` -- pasteable into a Colab cell, mounts
  Drive at the end to persist the adapter.
- `notebooks/kaggle_sft_qwen7b.py` -- assumes you've uploaded
  `trajectories.jsonl` + `trajectories2.jsonl` as a Kaggle dataset and
  attached it to the notebook.

Both drivers default to `max_length=4096` and `bnb 4-bit`. For an A100 on
Colab Pro, edit `config/sft_config.yaml` to bump `model.max_length=8192`
before running.

If you only want to dry-run the agent loop on Colab/Kaggle (no
hidden_rate/visible_rate), set `SKIP_PREDICTIONS=0` in the colab driver --
it invokes `eval_sft.py --predictions-only`, which generates predictions
and saves trajectories but skips the harness and labeler. The report will
mark hidden/visible rates as unavailable.

## Outputs

After a full run you will have:

```
src/stage_1_sft/
├── data/
│   ├── merged_trajectories.jsonl          # 119 unique rows (dedup of inputs)
│   ├── sft_trajectories.jsonl             # train split, post-filter
│   ├── sft_eval_trajectories.jsonl        # held-out eval split, post-filter
│   ├── sft_eval_instance_ids.txt          # plain-text id list for mini-SWE
│   └── sft_dataset/                       # tokenized HF Dataset (loss-masked)
│       ├── data-00000-of-*.arrow
│       ├── tokenizer/
│       └── train_instance_ids.txt
├── checkpoints/sft_warmstart/
│   ├── final/                             # LoRA adapter + tokenizer + metadata
│   └── merged/                            # merged HF checkpoint for vLLM
├── evaluation/
│   ├── results/sft_results.jsonl
│   ├── results/base_results.jsonl
│   └── sft_eval_report.md
└── runs/sft-eval-sft-<timestamp>/         # mini-SWE-agent run root
    ├── predictions.eval.jsonl
    ├── reward_hack_labels.jsonl
    ├── test_visibility_splits.jsonl
    └── mini_swe_agent/
```

Key invariants:

- `data/sft_dataset/train_instance_ids.txt` ∩ `data/sft_eval_instance_ids.txt` = ∅
  (`eval_sft.py` asserts this at startup).
- Reward `R` is computed identically in `process_trajectories.py`,
  `eval_sft.py`, and (later) the Phase 2 RL trainer -- they all read
  coefficients from `config.reward`.

## Common gotchas

- **`vLLM`/`bitsandbytes` import errors on macOS**: both packages require
  CUDA; train and eval on Linux/CUDA. The processing/build steps work on Mac.
- **Empty SFT training set**: indicates the filter is too strict for the
  current data. Loosen `filtering.min_hidden_rate` to 0.6 or set
  `filtering.all_signals_false: false`. Re-running `process_trajectories.py`
  prints which rejection reasons dominate.
- **`Train/eval instance id overlap detected`**: re-run `process_trajectories.py`
  -- the held-out file is stale. The check is intentional; do not bypass.
- **mini-SWE-agent + Docker on Apple Silicon**: pass
  `--docker-platform linux/amd64` to `benchmark-agents swebench run` (already
  the default in the batch script when set in the env).
