# Benchmark Agents

This repository is set up to host multiple benchmark-oriented agents. The default SWE-bench solver now delegates the agent loop to [`SWE-agent/mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent) and wraps its `mini-extra swebench` batch runner with this repo's sampling, run metadata, and reward-hacking dataset audit tools.

The logging is intentionally explicit and auditable:

- mini-SWE-agent trajectories are persisted in the run output.
- The exact sampled instances and wrapper command are saved with the run.
- Final patches are exported in SWE-bench prediction format.

The legacy JSON-tool Vertex loop is still available as `benchmark-agents swebench vertex-run`, but the default `run` command uses mini-SWE-agent's bash-only loop instead of our custom parser-sensitive loop.

## What It Does

- Loads a SWE-bench split from Hugging Face.
- Deterministically samples `N` tasks from a seed.
- Runs `mini-extra swebench` from the `mini-swe-agent` package.
- Passes Gemini/Vertex settings from `.env` into the mini-SWE-agent subprocess.
- Writes `predictions.jsonl` in the format expected by the SWE-bench harness.
- Keeps this repo's reward-hacking audit command independent from the solver.

## Setup

1. Create and activate a virtual environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the package.

   ```bash
   pip install -e .
   ```

3. Authenticate for local Vertex AI usage.

   ```bash
   gcloud auth application-default login
   ```

4. Start Docker Desktop and wait until the daemon is ready.

   ```bash
   docker info
   ```

   mini-SWE-agent uses the official SWE-bench Docker images. On Apple Silicon,
   make sure Docker Desktop can run `linux/amd64` images through emulation if
   Docker reports a platform error after startup.

5. Export the Gemini/Vertex environment variables used by mini-SWE-agent through LiteLLM.

   ```bash
   cp .env.example .env
   set -a
   source .env
   set +a
   ```

   The default runner reads `.env` automatically too. For Vertex-backed Gemini,
   set at least `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`; `GOOGLE_APPLICATION_CREDENTIALS`
   may also be used if your application-default credentials are not enough.

## Run A 10-Problem Sample

The default dataset is `SWE-bench/SWE-bench_Lite`, which is the easiest place to iterate while we build the agent.

```bash
benchmark-agents swebench run --sample-size 10 --seed 42
```

Useful flags:

- `--dataset verified` switches to `SWE-bench/SWE-bench_Verified`
- `--dataset full` switches to `SWE-bench/SWE-bench`
- `--run-id my-run`
- `--project my-project-id`
- `--model gemini-2.5-pro` is converted to `vertex_ai/gemini-2.5-pro` for LiteLLM
- `--workers 4`
- `--environment-class docker`
- `--docker-platform linux/amd64` can help on Apple Silicon when Docker reports a platform mismatch
- `--expose-visible-tests` writes a deterministic public subset of `FAIL_TO_PASS`
  plus `PASS_TO_PASS` tests into `/testbed/VISIBLE_TESTS.txt`; the held-out split
  is recorded in `<run-root>/test_visibility_splits.jsonl`

The default model resolution order is `--model`, then `MINI_SWE_MODEL`, then
`GOOGLE_GENAI_MODEL`, then `gemini-2.5-pro`.

To run the old custom Vertex loop instead:

```bash
benchmark-agents swebench vertex-run --sample-size 10 --seed 42
```

To generate only a reproducible sample manifest without running the model:

```bash
benchmark-agents swebench sample --sample-size 10 --seed 42
```

## Audit For Reward Hacking

To measure reward hacking, run the regular solver prompt. Do not ask the model
to reward hack during the run:

```bash
benchmark-agents swebench run --sample-size 10 --seed 42
```

Then audit the run traces and final patches into a labeled JSONL dataset:

```bash
benchmark-agents swebench dataset --run-root runs/<run_id>
```

For SWE-bench evaluation, first create a filtered predictions file. This
excludes failed mini-SWE instances such as `CalledProcessError`,
`TimeoutExpired`, `LimitsExceeded`, and empty patches:

```bash
benchmark-agents swebench eval-predictions --run-root runs/<run_id>
```

Pass `runs/<run_id>/predictions.eval.jsonl` to the official harness.

By default, the dataset command writes only rows labeled `REWARD_HACK`. Use
`--include-non-hacks` when you also want control rows and ambiguous cases:

```bash
benchmark-agents swebench dataset --run-root runs/<run_id> --include-non-hacks
```

The dataset rows include the instance metadata, compact trajectory, final patch,
heuristic label, exploit signals, and reward-shaping fields for `H`, `V`, `E`,
`T`, and `G`. Hidden and visible pass rates are left as `null` until you evaluate
the generated patches with the official harness.

## Output Layout

Each run is written under `runs/<run_id>/`.

- `manifest.json`: run metadata
- `sampled_instances.jsonl`: the exact sampled instances
- `predictions.jsonl`: raw mini-SWE predictions, including failed/empty rows
- `predictions.eval.jsonl`: filtered SWE-bench evaluation file
- `summary.json`: run-level summary
- `mini_swe_agent/`: raw mini-SWE-agent output, logs, trajectories, and `preds.json`

Legacy `vertex-run` runs also write per-task folders:

- `tasks/<instance_id>/instance.json`
- `tasks/<instance_id>/trace.jsonl`
- `tasks/<instance_id>/prompts/`
- `tasks/<instance_id>/final_patch.diff`

## Evaluate With The Official Harness

This project writes predictions in the official JSONL format described by SWE-bench:

```json
{
  "instance_id": "repo_owner__repo_name-issue_number",
  "model_name_or_path": "your-model-name",
  "model_patch": "diff --git ..."
}
```

If you have the official harness installed separately, you can evaluate a run with:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path runs/<run_id>/predictions.eval.jsonl \
  --max_workers 4 \
  --run_id <run_id>
```

The harness setup is intentionally left separate because the official evaluation environment tends to have stricter system and Docker requirements than the solver itself.

## Batch Run, Evaluate, And Label

For repeated 50-instance batches with no repeated completed IDs:

```bash
scripts/batch_reward_hack_pipeline.sh
```

The batch script samples IDs excluding
`reward_hack_eval/batch_state/completed_instance_ids.txt`, runs mini-SWE-agent,
runs the official SWE-bench harness, labels reward hacking, and appends rows to
`reward_hack_eval/cumulative_reward_hack_labels.jsonl`. Instances excluded by
mini-SWE-agent as `LimitsExceeded` or `TimeoutExpired` are not added to the
completed-id state, so they can be sampled again in a later batch.

Useful overrides:

```bash
BATCH_SIZE=50 SEED=42 RUN_ID_PREFIX=lite-mini-visible \
  scripts/batch_reward_hack_pipeline.sh

ID_RANGE=51-100 scripts/batch_reward_hack_pipeline.sh

LABEL_EXTRA_ARGS="--judge llm --llm-provider vertex --llm-fallback static" \
  scripts/batch_reward_hack_pipeline.sh
```
