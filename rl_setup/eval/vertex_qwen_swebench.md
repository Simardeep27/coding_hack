# Using A Vertex AI Qwen Endpoint In The SWE-Bench Pipeline

You can use a Qwen Model Garden endpoint directly through the existing
`benchmark-agents swebench run` command. The important part is the LiteLLM model
string:

```text
vertex_ai/openai/<endpoint-id>
```

For your deployed endpoint, start with:

```text
vertex_ai/openai/mg-endpoint-670c7888-4a1f-47a2-91a8-860377dc8266
```

The deployed model display name you saw,
`qwen_qwen2_5-7b-instruct-mg-one-click-deploy`, is useful for human tracking, but
the pipeline should call the endpoint id.

## 1. Create A Vertex Env File

Create `.env.qwen_vertex` from this template:

```bash
GOOGLE_CLOUD_PROJECT=<your-project-id>
GOOGLE_CLOUD_LOCATION=<your-endpoint-region>
VERTEXAI_PROJECT=<your-project-id>
VERTEXAI_LOCATION=<your-endpoint-region>
GOOGLE_GENAI_USE_VERTEXAI=true
MSWEA_COST_TRACKING=ignore_errors
```

Authenticate locally if needed:

```bash
gcloud auth application-default login
```

If the `mg-endpoint-...` value does not work, get the numeric Vertex endpoint id
from the console or with:

```bash
gcloud ai endpoints list \
  --region=<your-endpoint-region> \
  --filter='displayName:mg-endpoint-670c7888-4a1f-47a2-91a8-860377dc8266' \
  --format='table(name,displayName,deployedModels.displayName)'
```

Then use:

```text
vertex_ai/openai/<numeric-endpoint-id>
```

## 2. Smoke Test Through LiteLLM

```bash
set -a
source .env.qwen_vertex
set +a

uv run python -c 'from litellm import completion; r = completion(model="vertex_ai/openai/mg-endpoint-670c7888-4a1f-47a2-91a8-860377dc8266", messages=[{"role":"user","content":"Say ok."}], max_tokens=8); print(r.choices[0].message.content)'
```

If that route fails with a provider/route error, try the non-OpenAI-compatible
fallback route once:

```bash
uv run python -c 'from litellm import completion; r = completion(model="vertex_ai/mg-endpoint-670c7888-4a1f-47a2-91a8-860377dc8266", messages=[{"role":"user","content":"Say ok."}], max_tokens=8); print(r.choices[0].message.content)'
```

Use whichever model string succeeds in the SWE-Bench command below.

## 3. Run A Tiny SWE-Bench Smoke Test

```bash
uv run benchmark-agents swebench run \
  --dataset lite \
  --split test \
  --sample-size 3 \
  --seed 777 \
  --run-id lite-qwen-vertex-smoke \
  --env-file .env.qwen_vertex \
  --model vertex_ai/openai/mg-endpoint-670c7888-4a1f-47a2-91a8-860377dc8266 \
  --workers 1 \
  --mini-model-class litellm_textbased \
  --mini-config .venv/lib/python3.13/site-packages/minisweagent/config/benchmarks/swebench_backticks.yaml \
  --expose-visible-tests
```

This evaluates the deployed Qwen base endpoint. If you want to evaluate your DPO
model, deploy merged Qwen+DPO weights or a LoRA-enabled serving container as a
separate Vertex endpoint and replace the endpoint id in `--model`.

## 4. Evaluate And Label

```bash
uv run benchmark-agents swebench eval-predictions \
  --run-root runs/lite-qwen-vertex-smoke
```

Then run the official SWE-Bench harness against
`runs/lite-qwen-vertex-smoke/predictions.eval.jsonl`, and label reward hacking
with the same `reward_hack_eval/label_reward_hacking.py` command used for your
Gemini runs.

## 5. Compare Fairly Against Gemini

After the smoke run works, freeze a shared list of instance ids and run:

- Qwen Vertex endpoint on those ids.
- Gemini on the same ids.
- Qwen DPO endpoint on the same ids, if/when you deploy the adapter or merged
  weights.

The main comparison should be solved rate, hidden pass rate, overfit gap,
reward-hack rate, and suspicious trajectory rate.
