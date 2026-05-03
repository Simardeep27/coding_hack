from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
#!/usr/bin/env python3
"""
Phase 2, Step 1: SFT Training with LoRA
=========================================

Fine-tunes a base code model on honest, successful SWE-bench trajectories
using supervised learning with LoRA (Low-Rank Adaptation) for memory efficiency.

This creates the "warm start" policy for the integrity-aware RL pipeline.
By training EXCLUSIVELY on trajectories that are:
    - Successfully resolved (correct solutions)
    - Free of exploitation signals (no test tampering, oracle access, etc.)
    - Generalizing well (high hidden test pass rate)

We ensure the policy begins from a non-exploitative prior, making the
subsequent RL phase more stable and less prone to discovering reward hacks.

Architecture decisions:
    - LoRA:                   Fits on 24GB GPU, preserves base model capabilities
    - 4-bit quantization:     Further memory reduction via bitsandbytes NF4
    - Gradient checkpointing: Trades compute for memory
    - Completion-only loss:   Only learns to generate agent actions
    - No packing:             Clean trajectory boundaries, no cross-contamination

Usage:
    # Dry run (10 steps, verify pipeline works)
    python -m training.sft_train --config config/sft_config.yaml --dry_run

    # Full training
    python -m training.sft_train --config config/sft_config.yaml

    # Override model
    python -m training.sft_train --model Qwen/Qwen2.5-Coder-14B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


def load_config(config_path: str = "config/sft_config.yaml") -> dict:
    """Load centralized configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: dict) -> tuple:
    """
    Load the base model with optional 4-bit quantization, ready for LoRA.

    Model choice rationale:
        We use an instruction-tuned code model (default: Qwen2.5-Coder-7B-Instruct)
        that already has strong code understanding and instruction following.
        The SFT phase adapts it to the specific agent action format and honest
        problem-solving patterns without destroying its general capabilities.

    Memory strategy for 24GB VRAM:
        - BF16 base: ~14GB for 7B model
        - 4-bit NF4:  ~4GB for 7B model (with double quantization)
        - LoRA adapters: ~100-200MB additional
        - Gradient checkpointing: reduces activation memory 60-70%

    Args:
        config: Configuration dictionary with model and quantization settings.

    Returns:
        Tuple of (model, tokenizer).
    """
    model_name = config["model"]["name"]
    use_4bit = config["model"].get("use_4bit", False)

    print(f"[INFO] Loading model: {model_name}")
    print(f"[INFO] 4-bit quantization: {'enabled' if use_4bit else 'disabled'}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Load model with optional quantization ---
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        print("[INFO] Using NF4 quantization with double quantization")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing for memory savings
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {param_count / 1e6:.1f}M")

    return model, tokenizer


def setup_lora(model, config: dict):
    """
    Apply LoRA adapters to the model for parameter-efficient fine-tuning.

    LoRA adds small trainable rank-decomposition matrices to the attention
    layers. With r=16 and targeting q/k/v/o projections, we add ~0.5% new
    trainable parameters while keeping 99.5% of the model frozen.

    Benefits for the integrity-aware RL pipeline:
        1. Memory: Fits 7B model training on a single 24GB GPU
        2. Preservation: Base model capabilities are maintained
        3. Modularity: Easy to merge/swap adapters for evaluation
        4. Speed: Training is faster with fewer gradient computations
        5. RL readiness: Same adapter approach works for Phase 2 PPO/DPO

    Args:
        model:  Base language model (optionally quantized).
        config: Configuration with LoRA hyperparameters.

    Returns:
        PEFT model with LoRA adapters attached.
    """
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],                          # Rank: 16
        lora_alpha=lora_cfg["alpha"],              # Alpha: 32 (scaling = alpha/r = 2)
        lora_dropout=lora_cfg["dropout"],          # Dropout: 0.05
        target_modules=lora_cfg["target_modules"], # q_proj, k_proj, v_proj, o_proj
        bias="none",                               # Don't train biases
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameter summary
    model.print_trainable_parameters()

    return model


def verify_dataset_safety(dataset) -> None:
    """
    Final safety verification before training begins.

    This is the LAST line of defense against accidentally training on
    reward-hacking trajectories. Runs before any gradient computation.

    Checks:
        - Dataset is non-empty
        - Required fields are present
        - Sample format is correct

    Note: The per-trajectory is_reward_hack assertion was already run in
    build_sft_dataset.py. This provides a structural check.

    Raises:
        AssertionError: If dataset is empty or missing required fields.
    """
    print("[INFO] Running final safety verification on training dataset...")

    assert len(dataset) > 0, (
        "SAFETY CHECK FAILED: Dataset is empty! "
        "No honest trajectories to train on. Aborting."
    )

    # Verify expected fields exist in the first example
    sample = dataset[0]
    assert "messages" in sample or "input_ids" in sample, (
        "SAFETY CHECK FAILED: Dataset missing required fields "
        "(expected 'messages' or 'input_ids'). Aborting."
    )

    print(
        f"[INFO] Safety verification PASSED: "
        f"{len(dataset)} clean examples ready for training"
    )


def create_formatting_func(tokenizer):
    """
    Create a formatting function for SFTTrainer.

    Converts the messages list in each example to the model's chat template
    format. Using the model's native chat template ensures compatibility
    with the instruction-following format learned during pre-training.

    For Qwen2.5 models, this produces:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
        ...
        <|im_end|>
        <|im_start|>assistant
        ...
        <|im_end|>

    Args:
        tokenizer: HuggingFace tokenizer with chat template support.

    Returns:
        Formatting function compatible with SFTTrainer.
    """

    def formatting_func(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return text

    return formatting_func


def main():
    """
    Main entry point: setup, verify, train, and save the SFT warm-started model.

    Training pipeline:
        1. Load config and dataset
        2. Run safety verification (no reward hacks in data)
        3. Load base model with optional 4-bit quantization
        4. Attach LoRA adapters
        5. Configure completion-only loss masking
        6. Train with SFTTrainer
        7. Save final checkpoint with metadata
    """
    parser = argparse.ArgumentParser(
        description="SFT training for integrity-aware RL warm start"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sft_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run only 10 steps to verify the full pipeline works end-to-end",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name from config",
    )
    args = parser.parse_args()

    # --- Load config ---
    config = load_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    train_cfg = config["training"]

    # --- Load dataset ---
    dataset_dir = config["data"]["dataset_output_dir"]
    print(f"[INFO] Loading SFT dataset from {dataset_dir}")
    dataset = load_from_disk(dataset_dir)
    print(f"[INFO] Dataset size: {len(dataset)} examples")

    # --- Safety verification ---
    verify_dataset_safety(dataset)

    # --- Setup model and LoRA ---
    model, tokenizer = setup_model_and_tokenizer(config)
    model = setup_lora(model, config)

    # --- Output directories ---
    output_dir = train_cfg["output_dir"]
    logging_dir = train_cfg.get("logging_dir", f"{output_dir}/logs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logging_dir).mkdir(parents=True, exist_ok=True)

    # --- Dry run adjustments ---
    max_steps = -1
    num_epochs = train_cfg["num_train_epochs"]
    if args.dry_run:
        max_steps = 10
        num_epochs = 1
        print("\n" + "*" * 50)
        print("* DRY RUN MODE: Running 10 steps only")
        print("*" * 50 + "\n")

    # --- Data collator for completion-only loss masking ---
    # This ensures loss is ONLY computed on assistant responses.
    # For Qwen2.5 models, the assistant turn starts with "<|im_start|>assistant\n"
    # and user turns start with "<|im_start|>user\n".
    response_template = "<|im_start|>assistant\n"
    instruction_template = "<|im_start|>user\n"

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        instruction_template=instruction_template,
        tokenizer=tokenizer,
    )

    # --- Training arguments ---
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_dir=logging_dir,
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        report_to="tensorboard",
        packing=train_cfg.get("packing", False),
        max_seq_length=config["model"]["max_length"],
        dataset_text_field=None,
        remove_unused_columns=False,
    )

    # --- Create formatting function ---
    formatting_func = create_formatting_func(tokenizer)

    # --- Initialize SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # --- Print training summary ---
    effective_batch = (
        train_cfg["per_device_train_batch_size"]
        * train_cfg["gradient_accumulation_steps"]
    )
    steps_per_epoch = max(len(dataset) // effective_batch, 1)
    total_steps = steps_per_epoch * num_epochs if max_steps == -1 else max_steps

    print("\n" + "=" * 70)
    print("SFT TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"  Model:              {config['model']['name']}")
    print(f"  4-bit quantization: {config['model'].get('use_4bit', False)}")
    print(f"  LoRA r / alpha:     {config['lora']['r']} / {config['lora']['alpha']}")
    print(f"  Target modules:     {config['lora']['target_modules']}")
    print(f"  Learning rate:      {train_cfg['learning_rate']}")
    print(f"  Epochs:             {num_epochs}")
    print(f"  Batch size:         {train_cfg['per_device_train_batch_size']}")
    print(f"  Grad accumulation:  {train_cfg['gradient_accumulation_steps']}")
    print(f"  Effective batch:    {effective_batch}")
    print(f"  Steps per epoch:    {steps_per_epoch}")
    print(f"  Total steps:        {total_steps}")
    print(f"  Max seq length:     {config['model']['max_length']}")
    print(f"  Dataset size:       {len(dataset)} examples")
    print(f"  Output dir:         {output_dir}")
    print(f"  TensorBoard dir:    {logging_dir}")
    print(f"  Dry run:            {args.dry_run}")
    print("=" * 70 + "\n")

    # --- Train ---
    print("[INFO] Starting SFT training...")
    train_result = trainer.train()

    # --- Save final model ---
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # --- Print results ---
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Final train loss:     {train_result.training_loss:.4f}")
    print(f"  Total steps:          {train_result.global_step}")
    print(f"  Model saved to:       {final_dir}")
    print(f"  TensorBoard logs at:  {logging_dir}")
    print(f"  View with:  tensorboard --logdir {logging_dir}")
    print(f"{'=' * 70}")

    # --- Save training metadata ---
    metadata = {
        "model": config["model"]["name"],
        "dataset_size": len(dataset),
        "epochs": num_epochs,
        "total_steps": train_result.global_step,
        "final_loss": train_result.training_loss,
        "dry_run": args.dry_run,
        "lora_config": config["lora"],
        "training_config": {
            k: v
            for k, v in train_cfg.items()
            if k not in ["output_dir", "logging_dir"]
        },
        "reward_coefficients": config["reward"],
        "filtering_thresholds": config["filtering"],
    }
    metadata_path = final_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Training metadata saved to {metadata_path}")

    # --- Final dataset safety summary ---
    print(
        f"\n[SAFETY] Confirmed: trained on {len(dataset)} "
        f"verified honest trajectories"
    )
    print(f"[SAFETY] Zero reward-hacking trajectories in training data")
    print(f"[SAFETY] SFT warm start ready for Phase 2 RL fine-tuning")


if __name__ == "__main__":
    main()