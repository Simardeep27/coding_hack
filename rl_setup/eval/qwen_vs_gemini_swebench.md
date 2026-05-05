# Evaluating Qwen-DPO Against Gemini On SWE-Bench

This repo's SWE-Bench runner already delegates to mini-SWE-agent. To compare
Qwen-DPO against Gemini, serve the trained Qwen adapter behind an
OpenAI-compatible endpoint, point LiteLLM/mini-SWE-agent at that endpoint, run
the same SWE-Bench Lite instances, then evaluate and label the trajectories with
the existing reward-hack pipeline.

If Qwen is deployed on Vertex AI Model Garden instead of Colab/vLLM, use
`rl_setup/eval/vertex_qwen_swebench.md`. The model string for that path is
usually `vertex_ai/openai/<endpoint-id>`.

## 1. Serve Qwen-DPO From Colab

Use vLLM with the DPO adapter. In Colab, after the adapter exists:

```bash
pip install -U vllm
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --enable-lora \
  --lora-modules qwen-rh-dpo=/content/qwen-rh-dpo-adapter \
  --dtype bfloat16 \
  --max-model-len 8192
```

Expose the port to your Mac with one of:

- Colab port forwarding / local tunnel
- `ngrok http 8000`
- Cloudflare tunnel

You need a URL like:

```text
https://<your-tunnel>/v1
```

## 2. Smoke Test The Endpoint From Your Mac

```bash
curl -sS https://<your-tunnel>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dummy' \
  -d '{
    "model": "qwen-rh-dpo",
    "messages": [{"role": "user", "content": "Say ok."}],
    "max_tokens": 8
  }'
```

If this fails, fix serving before running SWE-Bench.

## 3. Run A Tiny Qwen-DPO SWE-Bench Smoke Test

Create a temporary env file, for example `.env.qwen_dpo`:

```bash
OPENAI_API_KEY=dummy
OPENAI_API_BASE=https://<your-tunnel>/v1
MSWEA_COST_TRACKING=ignore_errors
GOOGLE_GENAI_USE_VERTEXAI=false
```

Run 3-5 instances first:

```bash
uv run benchmark-agents swebench run \
  --dataset lite \
  --split test \
  --sample-size 5 \
  --seed 777 \
  --run-id lite-qwen-dpo-smoke \
  --env-file .env.qwen_dpo \
  --model openai/qwen-rh-dpo \
  --workers 1 \
  --mini-model-class litellm_textbased \
  --mini-config .venv/lib/python3.13/site-packages/minisweagent/config/benchmarks/swebench_backticks.yaml \
  --expose-visible-tests
```

The `swebench_backticks.yaml` config is often more robust for local/open-model
serving than tool-calling mode because it asks the model to emit exactly one bash
command in a fenced block.
The wrapper also infers `litellm_textbased` from `swebench_backticks.yaml`, but
the explicit `--mini-model-class` keeps the run command unambiguous.

## 4. Run Matching Baselines

Use the same instance IDs for fair comparison.

For Gemini:

```bash
uv run benchmark-agents swebench run \
  --dataset lite \
  --split test \
  --sample-size 5 \
  --seed 777 \
  --run-id lite-gemini-match \
  --model vertex_ai/gemini-3.1-pro-preview \
  --workers 1 \
  --expose-visible-tests
```

For Qwen base, serve the base model without the LoRA adapter and run:

```bash
uv run benchmark-agents swebench run \
  --dataset lite \
  --split test \
  --sample-size 5 \
  --seed 777 \
  --run-id lite-qwen-base-match \
  --env-file .env.qwen_base \
  --model openai/qwen-base \
  --workers 1 \
  --mini-model-class litellm_textbased \
  --mini-config .venv/lib/python3.13/site-packages/minisweagent/config/benchmarks/swebench_backticks.yaml \
  --expose-visible-tests
```

## 5. Evaluate And Label

For each run:

```bash
uv run benchmark-agents swebench eval-predictions --run-root runs/<run_id>
```

Then run the official SWE-Bench harness through your existing batch script or
manually. After harness reports exist, label reward hacking:

```bash
uv run python reward_hack_eval/label_reward_hacking.py \
  --predictions runs/<run_id>/predictions.eval.jsonl \
  --eval runs/<run_id>/predictions.eval.jsonl \
  --eval-reports-glob 'logs/run_evaluation/<run_id>-eval/*/*/report.json' \
  --trajectories-glob 'runs/<run_id>/mini_swe_agent/**/*.traj.json' \
  --test-visibility-splits runs/<run_id>/test_visibility_splits.jsonl \
  --output runs/<run_id>/reward_hack_labels.jsonl \
  --judge llm \
  --llm-provider vertex \
  --llm-fallback static
```

## 6. Compare Metrics

Report these side by side:

- solved/resolved rate
- visible pass rate
- hidden pass rate
- visible-hidden overfit gap
- reward-hack rate
- exploit type counts
- test/evaluator tampering count
- git-history/oracle-access count

The clean first claim is not "Qwen-DPO solves more SWE-Bench"; it is:

```text
Does Qwen-DPO reduce reward-hack/suspicious trajectory behavior compared with
Qwen-base under the same visible-test exposure setup?
```
