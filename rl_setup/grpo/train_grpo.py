from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_setup.grpo.rewards import (
    make_external_episode_reward,
    make_integrity_reward,
    make_reward_model_reward,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATA_DIR = Path("rl_setup/artifacts/grpo")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GRPO on full trajectory prompts with reward-model and integrity rewards."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default="/content/qwen-rh-grpo")
    parser.add_argument("--policy-adapter", default=None)
    parser.add_argument("--reward-model", default=None)
    parser.add_argument("--reward-model-adapter", default=None)
    parser.add_argument("--reward-model-batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--reward-model-weight", type=float, default=1.0)
    parser.add_argument("--integrity-weight", type=float, default=0.7)
    parser.add_argument(
        "--episode-evaluator-command",
        default=None,
        help=(
            "Optional command returning JSON metrics for a generated episode. "
            "Use {instance_id} and {completion_path} placeholders."
        ),
    )
    parser.add_argument("--episode-reward-weight", type=float, default=1.0)
    parser.add_argument("--episode-evaluator-timeout", type=int, default=600)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force-fp32-lm-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cast inputs to lm_head weight dtype during generation to avoid QLoRA dtype mismatches.",
    )
    parser.add_argument("--use-vllm", action="store_true")
    args = parser.parse_args()

    train_dataset, eval_dataset = load_grpo_datasets(args.data_dir)
    tokenizer, model = load_policy_model(
        args.model,
        adapter_path=args.policy_adapter,
        load_in_4bit=args.load_in_4bit,
    )

    reward_funcs = []
    reward_weights = []
    if args.reward_model or args.reward_model_adapter:
        reward_funcs.append(
            make_reward_model_reward(
                model_name_or_path=args.reward_model,
                adapter_path=args.reward_model_adapter,
                batch_size=args.reward_model_batch_size,
                max_length=args.max_prompt_length + args.max_completion_length,
                load_in_4bit=args.load_in_4bit,
            )
        )
        reward_weights.append(args.reward_model_weight)

    if args.episode_evaluator_command:
        reward_funcs.append(
            make_external_episode_reward(
                command_template=args.episode_evaluator_command,
                timeout_seconds=args.episode_evaluator_timeout,
            )
        )
        reward_weights.append(args.episode_reward_weight)

    reward_funcs.append(make_integrity_reward())
    reward_weights.append(args.integrity_weight)

    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from trl import GRPOConfig, GRPOTrainer

    peft_config = None
    if args.policy_adapter is None:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
            if args.force_fp32_lm_head:
                align_output_head_dtype(model, verbose=True)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        scale_rewards="batch",
        loss_type="dapo",
        reward_weights=reward_weights,
        logging_steps=1,
        save_steps=25,
        eval_strategy="steps",
        eval_steps=25,
        report_to="none",
        log_completions=True,
        num_completions_to_print=2,
        remove_unused_columns=False,
        use_vllm=args.use_vllm,
        vllm_mode="colocate" if args.use_vllm else "server",
        vllm_gpu_memory_utilization=0.25,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if args.force_fp32_lm_head:
        align_output_head_dtype(trainer.model, verbose=True)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def load_grpo_datasets(data_dir: Path) -> tuple[Any, Any]:
    from datasets import load_dataset

    train_path = data_dir / "grpo_train.jsonl"
    dev_path = data_dir / "grpo_dev.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run prepare_grpo_dataset.py first.")
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing {dev_path}. Run prepare_grpo_dataset.py first.")
    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    eval_dataset = load_dataset("json", data_files=str(dev_path), split="train")
    return train_dataset, eval_dataset


def load_policy_model(
    model_id: str,
    *,
    adapter_path: str | None,
    load_in_4bit: bool,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = adapter_path or model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    return tokenizer, model


def align_output_head_dtype(model: Any, *, verbose: bool = False) -> None:
    """Keep QLoRA generation from mixing fp32 hidden states with bf16 lm_head.

    `prepare_model_for_kbit_training()` intentionally leaves some activations in
    fp32 for stability. With Qwen2 causal LM generation, that can reach the final
    `lm_head` while the head is still bf16, causing:
    "RuntimeError: expected scalar type BFloat16 but found Float".
    The patch below wraps the output head so its input is cast to the head's
    actual weight dtype right before the final linear layer. This survives PEFT
    wrapping and targets the exact failing operation.
    """
    patched = 0
    for name, head in output_heads(model):
        if head is None or getattr(head, "_grpo_dtype_forward_patch", False):
            continue
        weight = getattr(head, "weight", None)
        if weight is not None:
            head.to(weight.dtype)

        def pre_hook(module, inputs):
            if not inputs:
                return inputs
            first = inputs[0]
            module_weight = getattr(module, "weight", None)
            if module_weight is not None and hasattr(first, "dtype") and first.dtype != module_weight.dtype:
                return (first.to(module_weight.dtype), *inputs[1:])
            return inputs

        head.register_forward_pre_hook(pre_hook)
        head._grpo_dtype_forward_patch = True
        patched += 1

    if verbose:
        print(f"patched_lm_heads={patched} total_lm_heads_seen={len(output_heads(model))}", flush=True)


def output_heads(model: Any) -> list[tuple[str, Any]]:
    heads: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def add(name: str, head: Any) -> None:
        if head is not None and id(head) not in seen:
            heads.append((name, head))
            seen.add(id(head))

    candidates = [model]
    for attr in ("base_model", "model"):
        value = getattr(model, attr, None)
        if value is not None:
            candidates.append(value)
            nested = getattr(value, "model", None)
            if nested is not None:
                candidates.append(nested)

    for candidate in candidates:
        get_output_embeddings = getattr(candidate, "get_output_embeddings", None)
        if callable(get_output_embeddings):
            add(f"{type(candidate).__name__}.get_output_embeddings", get_output_embeddings())
        add(f"{type(candidate).__name__}.lm_head", getattr(candidate, "lm_head", None))

    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        for name, module in named_modules():
            if name.endswith("lm_head"):
                add(name, module)

    return heads


if __name__ == "__main__":
    main()
