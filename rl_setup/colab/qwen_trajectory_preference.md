# Qwen Trajectory Preference Training On Colab

This guide turns `rl_setup` artifacts into the proposal stage:

> Trajectory Preference or Reward Modeling: construct pairs such as honest success > exploit success and honest partial solution > exploit-driven perfect visible score. Train a reward model or preference model over full trajectories, not only final code.

## 1. Export TRL Data Locally

From the repo root:

```bash
./rl_setup/run_api_first_pipeline.sh
```

This writes:

```text
rl_setup/artifacts/trl/sft_train.jsonl
rl_setup/artifacts/trl/sft_dev.jsonl
rl_setup/artifacts/trl/sft_test.jsonl
rl_setup/artifacts/trl/dpo_train.jsonl
rl_setup/artifacts/trl/dpo_dev.jsonl
rl_setup/artifacts/trl/dpo_test.jsonl
rl_setup/artifacts/trl/reward_model_pairs.jsonl
rl_setup/artifacts/trl/manifest.json
```

Upload `rl_setup/artifacts/trl/` to Google Drive or directly to the Colab runtime.

## 2. Colab Setup

Use an A100 or H100 runtime.

First verify the runtime:

```python
!nvidia-smi
```

If you see an error like `AttributeError: module 'torch' has no attribute '_utils'`,
the Colab environment has a broken or partially shadowed PyTorch install. Run this
clean install cell, then restart the runtime before importing `transformers`,
`trl`, or `peft`.

```python
%pip uninstall -y torch torchvision torchaudio transformers trl accelerate peft bitsandbytes xformers torchao
%pip cache purge
%pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
%pip install --no-cache-dir transformers==4.56.2 trl==0.23.1 accelerate peft datasets bitsandbytes
```

Then use **Runtime -> Restart runtime**. After the restart, run:

```python
import os
import torch
import torch._utils
import transformers
import trl

print("cwd:", os.getcwd())
print("torch:", torch.__version__, torch.__file__)
print("torch._utils:", torch._utils.__file__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("transformers:", transformers.__version__)
print("trl:", trl.__version__)
```

If `torch.__file__` points to something in `/content/torch.py` or `/content/torch/`,
rename/delete that local file or directory. It is shadowing the real PyTorch package.

If PEFT raises an incompatible `torchao` error, remove the preinstalled package and
restart again:

```python
%pip uninstall -y torchao
```

This workflow uses LoRA/QLoRA through PEFT and bitsandbytes, not torchao.

If the sanity check passes, the normal install cell is enough on future fresh runtimes:

```python
%pip install --no-cache-dir transformers==4.56.2 trl==0.23.1 accelerate peft datasets bitsandbytes
```

Optional login if you want to push adapters:

```python
from huggingface_hub import notebook_login
notebook_login()
```

## 3. SFT Warm Start

Use SFT first to teach Qwen the style of clean trajectories.

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
data_dir = "/content/trl"  # change if mounted from Drive

train_ds = load_dataset("json", data_files=f"{data_dir}/sft_train.jsonl", split="train")
eval_ds = load_dataset("json", data_files=f"{data_dir}/sft_dev.jsonl", split="train")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

args = SFTConfig(
    output_dir="/content/qwen-rh-sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_length=4096,
    logging_steps=5,
    save_steps=50,
    bf16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model_id,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model("/content/qwen-rh-sft-adapter")
```

## 4. DPO / Preference Training

After SFT, train preference behavior using full-trajectory pairs.

For A100 40GB, use 4-bit QLoRA. A plain bf16 DPO run can OOM because DPO
effectively needs policy/reference behavior plus long chosen/rejected
trajectories.

Do not pass `model=model_id` directly to `DPOTrainer` on A100 40GB. That path
lets TRL load the model in its default precision, which can OOM before LoRA helps.
Load the quantized model yourself and pass the model object to the trainer.

```python
import gc
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

gc.collect()
torch.cuda.empty_cache()

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
data_dir = "/content/trl"

dpo_ds = load_dataset("json", data_files=f"{data_dir}/dpo_train.jsonl", split="train")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

args = DPOConfig(
    output_dir="/content/qwen-rh-dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    num_train_epochs=1,
    max_length=2048,
    max_prompt_length=512,
    beta=0.1,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_steps=50,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    args=args,
    train_dataset=dpo_ds,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model("/content/qwen-rh-dpo-adapter")
```

## 5. Preference Evaluation

After DPO, evaluate whether the trained adapter assigns higher likelihood to
held-out chosen trajectories than rejected trajectories. This is the fastest
check that the preference model learned the intended ordering.

```python
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
adapter_dir = "/content/qwen-rh-dpo-adapter"
data_dir = "/content/trl"
eval_file = f"{data_dir}/dpo_dev.jsonl"  # use dpo_test.jsonl once at the end

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
)
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

def sequence_logprob(prompt, response, max_length=1024):
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(model.device)
    full = tokenizer(prompt + response, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = full.input_ids.to(model.device)
    attention_mask = full.attention_mask.to(model.device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    token_count = (labels != -100).sum().item()
    return -out.loss.item() * max(token_count, 1)

correct = 0
total = 0
margin_sum = 0.0

with open(eval_file, "r", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        chosen_lp = sequence_logprob(row["prompt"], row["chosen"])
        rejected_lp = sequence_logprob(row["prompt"], row["rejected"])
        correct += int(chosen_lp > rejected_lp)
        total += 1
        margin_sum += chosen_lp - rejected_lp

print("pairs:", total)
print("preference_accuracy:", correct / total if total else None)
print("avg_logprob_margin:", margin_sum / total if total else None)
```

Run the same code with `eval_file = f"{data_dir}/dpo_test.jsonl"` only after
you finish tuning.

If DPO still OOMs on A100 40GB:

```python
# lower these first
max_length = 1536
max_prompt_length = 384

# or use a smaller LoRA rank
r = 4
lora_alpha = 8
```

Also restart the runtime before retrying a failed OOM run. Otherwise PyTorch can
keep fragmented/reserved memory around.

## 6. Important Experimental Note

The current `dpo_train.jsonl` is a first-pass generic trajectory preference dataset. Many pairs compare different SWE-Bench instances. That is acceptable for a basic preference-modeling demo, but the stronger experiment is:

1. Run Qwen multiple times per same SWE-Bench instance.
2. Label each trajectory with `rl_setup`.
3. Build same-instance pairs:
   - honest success > exploit success
   - honest partial solution > exploit-driven perfect visible score
   - clean failure > test/evaluator tampering
4. Train DPO/reward model on those same-task pairs.

That same-instance version is the cleanest match to the proposal.
