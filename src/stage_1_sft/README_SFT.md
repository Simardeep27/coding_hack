# Phase 2, Step 1 — SFT Warm Start (Vertex AI Supervised Tuning of Gemini)

Tunes Gemini Flash on the honest, successful trajectories produced in
Phase 1 so the policy starts from a non-exploitative prior. The base model
used to collect trajectories was Gemini via Vertex AI, so we tune Gemini
directly with Vertex AI's hosted SFT API. **No local GPU is required.**

> **About Pro vs Flash.** Vertex AI Supervised Tuning supports the
> Gemini Flash tier (`gemini-2.5-flash`, `gemini-2.5-flash-lite`).
> **`gemini-2.5-pro` cannot be tuned.** You can still call un-tuned Pro
> for inference comparison.

> **About the "Vertex API key".** A Google AI Studio key (`AIza…`) does
> NOT work for tuning. You need either Application Default Credentials
> (`gcloud auth application-default login`) or a service-account JSON
> key file. Your existing mini-swe-agent setup already uses ADC.

---

## 0. Prerequisites

- Python 3.11+
- `gcloud` CLI installed (for ADC) — install with
  `brew install --cask google-cloud-sdk` on macOS.
- A GCP project with **billing enabled** and the **Vertex AI API enabled**.
- Docker Desktop running (only needed for evaluation, which uses
  mini-swe-agent + the SWE-bench Docker images).
- The Phase 1 dataset at
  `src/stage_1_sft/data/trajectories.jsonl`.

### One-time GCP setup

```bash
# 1. Authenticate
gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT

# 2. Enable APIs
gcloud services enable aiplatform.googleapis.com storage.googleapis.com

# 3. (Optional) grant your user the right roles
gcloud projects add-iam-policy-binding YOUR_GCP_PROJECT \
    --member="user:YOUR_EMAIL" \
    --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding YOUR_GCP_PROJECT \
    --member="user:YOUR_EMAIL" \
    --role="roles/storage.objectAdmin"
```

### Configure the project locally

Edit `src/stage_1_sft/config/sft_config.yaml`:

```yaml
vertex:
  project: YOUR_GCP_PROJECT
  location: us-central1            # tuning is in us-central1
  bucket: your-bucket-name-sft     # will be created if missing
  bucket_prefix: sft-warmstart
```

Or leave them `null` and put values in the repo-root `.env` file:

```
GOOGLE_CLOUD_PROJECT=YOUR_GCP_PROJECT
GOOGLE_CLOUD_LOCATION=us-central1
GCS_TUNING_BUCKET=your-bucket-name-sft
```

---

## 1. Install dependencies

From `src/stage_1_sft/`:

```bash
pip install -r requirements.txt
```

This stack is small: `google-cloud-aiplatform`, `google-cloud-storage`,
`google-genai`, `datasets`, `pyyaml`, `tqdm`. **No torch / peft /
bitsandbytes** — all training happens server-side on Vertex.

---

## 2. End-to-end pipeline

Run from `src/stage_1_sft/`. Each stage prints a summary so you can stop
between steps if anything looks off.

```bash
# A. Score and filter trajectories (unchanged from before)
python -m data.process_trajectories --config config/sft_config.yaml

# B. Convert to the Vertex SFT JSONL format
python -m data.build_sft_dataset --config config/sft_config.yaml

# C. (Recommended) dry-run the training script: validates auth,
#    creates the GCS bucket if missing, uploads the dataset, but
#    does NOT submit a tuning job. Costs nothing.
python -m training.sft_train --config config/sft_config.yaml --dry_run

# D. Submit the real tuning job (blocks until done; 30–90 min typical)
python -m training.sft_train --config config/sft_config.yaml

# E. Evaluate the tuned endpoint on previously-hacked instances
python -m evaluation.eval_sft --config config/sft_config.yaml
```

### Expected outputs

| Step | Writes |
|---|---|
| A | `data/sft_trajectories.jsonl` |
| B | `data/vertex_sft_train.jsonl`, `data/vertex_sft_val.jsonl` |
| D | `checkpoints/sft_warmstart/tuned_model.json` |
| E | `evaluation/sft_eval_report.md`, `evaluation/results/*.json`, `runs/sft-eval-tuned-*/`, `runs/sft-eval-base-*/` |

---

## 3. What the tuning step actually costs

Vertex Supervised Tuning for Gemini 2.5 Flash bills per training token.
With ~50 trajectories × ~3K tokens × 3 epochs ≈ **450K training tokens**
the tuning job typically costs **under USD 1**. Hosting the tuned
endpoint costs more — when you're done evaluating, delete it via
`gcloud ai endpoints delete ENDPOINT_ID --region=us-central1` to stop
charges.

---

## 4. Why no GPU / why Colab T4 doesn't apply

You asked about Colab T4. **Skip it for this path.** Vertex AI
Supervised Tuning runs entirely on Google's hardware. Your laptop only
submits the JSON config and polls for completion. The whole local
environment is just a thin client.

If you ever decide to go back to local LoRA on an open-weight model
(e.g. Qwen2.5-Coder), the original LoRA pipeline lives in git history
and you would then need a GPU; T4 (16 GB) only fits ≤3B models in
4-bit, A100 fits 7B comfortably.

---

## 5. Evaluation methodology

`eval_sft.py` does the right thing for an integrity-aware warm start:

1. Reads the tuned endpoint name from `checkpoints/sft_warmstart/tuned_model.json`.
2. Selects N instances from `trajectories.jsonl` where
   `is_reward_hack == True` — the **adversarial set** (default N=10).
3. Selects M previously-honest resolved instances — the **regression
   set** (default M=5), to confirm SFT didn't break good behavior.
4. Runs `benchmark-agents swebench run --instance-id … --model
   vertex_ai/projects/.../endpoints/ENDPOINT_ID` for each model
   (tuned and un-tuned base).
5. Runs `benchmark-agents swebench dataset --include-non-hacks` to
   relabel each new run with the static-signal reward-hack labeler.
6. Compares hack rates and writes `evaluation/sft_eval_report.md`.

The labeler uses static signals (test tampering, oracle access, etc.)
which is enough to answer "did this trajectory still hack?" without
running the full SWE-bench harness. **To also populate `hidden_rate`
and `visible_rate` for the reward calculation**, run the official
harness on each evaluation run root:

```bash
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --predictions_path runs/sft-eval-tuned-<TIMESTAMP>/predictions.eval.jsonl \
    --max_workers 4 \
    --run_id sft-eval-tuned-<TIMESTAMP>-eval
```

…then re-run `swebench dataset` so the labeler picks up rates from the
harness reports.

---

## 6. Troubleshooting

- **`PermissionDenied: 403 ... aiplatform.tuningJobs.create`** — your
  user/service account is missing `roles/aiplatform.user` on the project.
- **`NotFound: 404 ... bucket`** — the script tries to create the bucket
  in your configured `vertex.location`. If your project policy forbids
  bucket creation, create it manually:
  `gsutil mb -p PROJECT -l us-central1 gs://BUCKET_NAME`.
- **`ResourceExhausted: 429 quota`** — supervised tuning has per-project
  quotas. File a quota increase from the GCP console (Quotas page,
  filter for "Vertex AI Tuning Service").
- **`benchmark-agents` not found in eval** — the eval script tries
  `uv run benchmark-agents` first, then `benchmark-agents`. Run
  `pip install -e .` from the repo root to install the CLI.
- **Tuned-endpoint inference returns 404** — the endpoint can take a few
  minutes to become serveable after tuning ends; retry, or check
  `gcloud ai endpoints list --region=us-central1`.

---

## 7. What to hand off to Phase 2 (Online RL)

`checkpoints/sft_warmstart/tuned_model.json` is the entire handoff. It
contains `tuned_model_endpoint_name`, which is what the RL phase will
query as its policy and (if you do PPO/GRPO with rollout-then-reward)
keep updating via successive tuning jobs. The integrity-aware reward
function R = aH + bV − cE − dT − eG is already specified in
`config/sft_config.yaml` and is reused unchanged across stages.