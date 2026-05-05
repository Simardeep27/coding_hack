from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_setup.grpo.rewards import make_integrity_reward, make_reward_model_reward


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATASET = Path("rl_setup/artifacts/grpo/grpo_dev.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate held-out GRPO trajectories and score them with reward functions."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("/content/grpo_eval_results.jsonl"))
    parser.add_argument("--reward-model", default=None)
    parser.add_argument("--reward-model-adapter", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)[: args.limit]
    tokenizer, model = load_policy_model(
        args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
    )

    reward_funcs = [make_integrity_reward()]
    reward_names = ["integrity_reward"]
    if args.reward_model or args.reward_model_adapter:
        reward_funcs.insert(
            0,
            make_reward_model_reward(
                model_name_or_path=args.reward_model,
                adapter_path=args.reward_model_adapter,
                max_length=args.max_prompt_length + args.max_new_tokens,
                load_in_4bit=args.load_in_4bit,
            ),
        )
        reward_names.insert(0, "reward_model_reward")

    results = []
    for row in rows:
        completion = generate_completion(
            tokenizer,
            model,
            prompt=str(row["prompt"]),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        scores = {
            name: func([completion])[0]
            for name, func in zip(reward_names, reward_funcs)
        }
        scores["total_reward"] = sum(scores.values())
        results.append(
            {
                "instance_id": row.get("instance_id"),
                "split": row.get("split"),
                "scores": scores,
                "completion": completion,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, results)
    summary = summarize(results)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {summary_path}")


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
    model.config.pad_token_id = tokenizer.pad_token_id
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_completion(
    tokenizer: Any,
    model: Any,
    *,
    prompt: str,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output_ids[0, encoded.input_ids.shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    score_names = sorted(
        {
            score_name
            for row in results
            for score_name in row.get("scores", {}).keys()
        }
    )
    summary = {"count": len(results)}
    for score_name in score_names:
        values = [
            float(row["scores"][score_name])
            for row in results
            if isinstance(row.get("scores", {}).get(score_name), (int, float))
        ]
        if values:
            summary[score_name] = {
                "avg": round(mean(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object on line {line_no} of {path}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

