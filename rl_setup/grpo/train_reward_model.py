from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_setup.grpo.proposal_reward import ProposalRewardConfig, proposal_episode_reward


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATA_DIR = Path("rl_setup/artifacts/trl")
DEFAULT_SCORED = Path("rl_setup/artifacts/scored_episodes.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a trajectory reward model from rl_setup reward_model_pairs JSONL."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--scored-path", type=Path, default=DEFAULT_SCORED)
    parser.add_argument(
        "--margin-source",
        choices=["proposal", "safe"],
        default="proposal",
        help=(
            "Use proposal formula R=aH+bV-cE-dT-eG from scored episodes, "
            "or existing safe-reward margins embedded in TRL exports."
        ),
    )
    parser.add_argument("--output-dir", default="/content/qwen-rh-reward-model")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    train_dataset, eval_dataset = load_reward_datasets(
        args.data_dir,
        scored_path=args.scored_path,
        margin_source=args.margin_source,
    )
    tokenizer, model = load_reward_model(args.model, load_in_4bit=args.load_in_4bit)

    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from trl import RewardConfig, RewardTrainer

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
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
        modules_to_save=["score"],
    )

    training_args = RewardConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=25,
        report_to="none",
        center_rewards_coefficient=0.01,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def load_reward_datasets(
    data_dir: Path,
    *,
    scored_path: Path,
    margin_source: str,
) -> tuple[Any, Any]:
    from datasets import load_dataset

    train_path = data_dir / "reward_model_pairs_train.jsonl"
    dev_path = data_dir / "reward_model_pairs_dev.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run rl_setup export-trl first.")
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing {dev_path}. Run rl_setup export-trl first.")

    score_index = load_score_index(scored_path) if margin_source == "proposal" else {}
    if margin_source == "proposal" and not score_index:
        raise FileNotFoundError(
            f"Could not load proposal reward scores from {scored_path}. "
            "Upload scored_episodes.jsonl to Colab or rerun with --margin-source safe."
        )

    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    eval_dataset = load_dataset("json", data_files=str(dev_path), split="train")
    return (
        normalize_reward_dataset(train_dataset, score_index=score_index, margin_source=margin_source),
        normalize_reward_dataset(eval_dataset, score_index=score_index, margin_source=margin_source),
    )


def normalize_reward_dataset(
    dataset: Any,
    *,
    score_index: dict[str, float],
    margin_source: str,
) -> Any:
    def normalize(row: dict[str, Any]) -> dict[str, Any]:
        chosen_score, rejected_score = pair_scores(row, score_index, margin_source)
        margin = max(0.0, min(4.0, chosen_score - rejected_score))
        return {
            "chosen": str(row["chosen"]),
            "rejected": str(row["rejected"]),
            "margin": margin,
        }

    return dataset.map(normalize, remove_columns=dataset.column_names)


def pair_scores(
    row: dict[str, Any],
    score_index: dict[str, float],
    margin_source: str,
) -> tuple[float, float]:
    if margin_source == "proposal":
        chosen_id = str(row.get("chosen_instance_id") or row.get("winner_instance_id") or "")
        rejected_id = str(row.get("rejected_instance_id") or row.get("loser_instance_id") or "")
        return score_index.get(chosen_id, 0.0), score_index.get(rejected_id, 0.0)
    return _float(row.get("chosen_reward_safe")), _float(row.get("rejected_reward_safe"))


def load_score_index(scored_path: Path) -> dict[str, float]:
    if not scored_path.exists():
        return {}
    index: dict[str, float] = {}
    config = ProposalRewardConfig()
    for line_no, line in enumerate(scored_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected object on line {line_no} of {scored_path}")
        instance_id = str(row.get("instance_id") or "")
        if instance_id:
            index[instance_id] = proposal_episode_reward(row, config)
    return index


def load_reward_model(model_id: str, *, load_in_4bit: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "num_labels": 1,
        "problem_type": "regression",
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

    model = AutoModelForSequenceClassification.from_pretrained(model_id, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def _float(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


if __name__ == "__main__":
    main()
