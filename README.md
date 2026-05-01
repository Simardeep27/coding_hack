# Benchmark Agents

This repository is set up to host multiple benchmark-oriented agents. The first agent is a SWE-bench solver that samples tasks, clones the target repositories, asks a Vertex-backed model to inspect and edit the codebase, and writes out both a prediction file and a visible reasoning/action trace for each task.

The logging is intentionally explicit and auditable:

- Each step records a visible reasoning summary.
- Every tool action and observation is persisted to disk.
- Raw prompts and raw model responses are saved per step.
- Final patches are exported in SWE-bench prediction format.

It does not depend on provider-hidden reasoning tokens or hidden chain-of-thought access.

## What It Does

- Loads a SWE-bench split from Hugging Face.
- Deterministically samples `N` tasks from a seed.
- Clones each task repository at its `base_commit`.
- Runs a transparent tool-using loop with Vertex AI.
- Writes `predictions.jsonl` in the format expected by the SWE-bench harness.
- Exposes the task's target test names both in the prompt and inside `.benchmark_agent/` in each repo checkout.

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

4. Export the Vertex environment variables expected by `google-genai`.

   ```bash
   cp .env.example .env
   set -a
   source .env
   set +a
   ```

## Run A 10-Problem Sample

The default dataset is `SWE-bench/SWE-bench_Lite`, which is the easiest place to iterate while we build the agent.

```bash
benchmark-agents swebench run --sample-size 10 --seed 42
```

Useful flags:

- `--dataset verified` switches to `SWE-bench/SWE-bench_Verified`
- `--dataset full` switches to `SWE-bench/SWE-bench`
- `--run-id my-run`
- `--max-steps 14`
- `--project my-project-id`
- `--model gemini-2.5-pro`

To generate only a reproducible sample manifest without running the model:

```bash
benchmark-agents swebench sample --sample-size 10 --seed 42
```

## Output Layout

Each run is written under `runs/<run_id>/`.

- `manifest.json`: run metadata
- `sampled_instances.jsonl`: the exact sampled instances
- `predictions.jsonl`: SWE-bench submission/evaluation file
- `summary.json`: run-level summary
- `tasks/<instance_id>/instance.json`: task metadata
- `tasks/<instance_id>/trace.jsonl`: visible reasoning trace and tool log
- `tasks/<instance_id>/prompts/`: raw prompt and response files per step
- `tasks/<instance_id>/final_patch.diff`: final git diff for that task

Inside each checked-out repo, the agent also gets a hidden context folder:

- `.benchmark_agent/task_context.json`
- `.benchmark_agent/fail_to_pass_tests.txt`
- `.benchmark_agent/pass_to_pass_tests.txt`

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
  --predictions_path runs/<run_id>/predictions.jsonl \
  --max_workers 4 \
  --run_id <run_id>
```

The harness setup is intentionally left separate because the official evaluation environment tends to have stricter system and Docker requirements than the solver itself.
