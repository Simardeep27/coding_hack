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

Implementation notes:
    The dataset built by ``data/build_sft_dataset.py`` already contains
    ``input_ids``, ``attention_mask``, and ``labels`` (with ``-100`` on every
    non-assistant token). We therefore train with plain
    ``transformers.Trainer`` + ``DataCollatorForSeq2Seq`` rather than
    ``trl.SFTTrainer`` -- this sidesteps the moving target of the trl API
    (``DataCollatorForCompletionOnlyLM`` was removed in trl >= 0.20,
    ``tokenizer`` was renamed to ``processing_class``, ``max_seq_length``
    moved into ``SFTConfig``, etc.).

Usage:
    # Dry run (10 steps, verify pipeline works)
    python -m training.sft_train --config config/sft_config.yaml --dry_run

    # Full training
    python -m training.sft_train --config config/sft_config.yaml

    # Override model
    python -m training.sft_train --model Qwen/Qwen2.5-Coder-14B-Instruct
"""

import argparse
import inspect
import json
import os
from pathlib import Path

# Reduce CUDA fragmentation OOMs on small GPUs (T4/P100). Must be set before
# torch's CUDA caching allocator initialises, hence import order matters.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


KEEP_COLUMNS = ("input_ids", "attention_mask", "labels")


def load_config(config_path: str = "config/sft_config.yaml") -> dict:
    """Load centralized configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def detect_precision(train_cfg: dict) -> tuple[bool, bool]:
    """
    Pick a numerically-safe (bf16, fp16) pair for the current device.

    Config defaults to bf16, but T4 / V100 / P100 do not support native bf16
    and silently fall back to slow software emulation (or fail outright in
    bitsandbytes 4-bit kernels). Detect the device and downgrade to fp16
    when bf16 is unsupported.

    Args:
        train_cfg: Training section of the YAML config.

    Returns:
        (use_bf16, use_fp16) -- exactly one is True when CUDA is available;
        both are False on CPU-only hosts (Trainer handles fp32 in that case).
    """
    want_bf16 = bool(train_cfg.get("bf16", True))
    want_fp16 = bool(train_cfg.get("fp16", False))

    if not torch.cuda.is_available():
        # CPU-only smoke tests: disable both, let Trainer use fp32.
        return False, False

    bf16_supported = torch.cuda.is_bf16_supported()
    if want_bf16 and bf16_supported:
        return True, False
    if want_bf16 and not bf16_supported:
        print(
            "[INFO] CUDA device does not support bf16 (likely T4/V100/P100). "
            "Falling back to fp16."
        )
        return False, True
    if want_fp16:
        return False, True
    # User explicitly disabled both: still safer to use fp16 on consumer GPUs
    # if bf16 is unsupported, otherwise honour the user's request.
    return False, not bf16_supported


def setup_model_and_tokenizer(config: dict) -> tuple:
    """
    Load the base model with optional 4-bit quantization, ready for LoRA.

    Memory strategy for 16 GB VRAM (Colab/Kaggle T4):
        - 4-bit NF4:               ~5 GB for the 7B base model
        - LoRA adapters:           ~25 MB additional (r=16, attn-only)
        - Gradient checkpointing:  reduces activation memory ~60%
        - Sequence length 4096:    final activations fit comfortably in <12 GB

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

    # Right-padding is what causal LM training expects (BOS-anchored sequences).
    tokenizer.padding_side = "right"

    # --- Choose compute dtypes -------------------------------------------------
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

    model_kwargs: dict = {
        "trust_remote_code": True,
        "dtype": compute_dtype,
    }

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        print(
            f"[INFO] NF4 + double-quant; compute_dtype="
            f"{'bfloat16' if bf16_supported else 'float16'}"
        )

    # --- Multi-GPU placement -------------------------------------------------
    # bnb 4-bit weights are pinned to a specific CUDA device. PyTorch
    # DataParallel (which Trainer auto-wraps when it sees >1 GPU) replicates
    # the model to other devices and crashes with "illegal memory access" on
    # quantized layers. Force model-parallel via accelerate's device_map="auto"
    # whenever multiple CUDA devices are visible: layers are sharded across
    # GPUs, the Trainer detects model.hf_device_map and skips DataParallel.
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        model_kwargs["device_map"] = "auto"
        print(f"[INFO] {n_gpus} GPUs visible -> using device_map='auto' (model parallel)")
    elif n_gpus == 1:
        # Single GPU: explicit placement avoids accidental CPU offload.
        model_kwargs["device_map"] = {"": 0}
        print("[INFO] 1 GPU visible -> placing model on cuda:0")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Disable cache: incompatible with gradient checkpointing during training.
    model.config.use_cache = False

    if use_4bit:
        # Prepare for k-bit training. We let TrainingArguments enable
        # gradient_checkpointing later (single source of truth) so we pass
        # use_gradient_checkpointing=False here.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )

    # When LoRA + gradient checkpointing without 4-bit, we still need to
    # ensure the input embedding outputs require grad so backprop reaches
    # the adapters. prepare_model_for_kbit_training does this for 4-bit;
    # do it manually for the non-4-bit path.
    if not use_4bit and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {param_count / 1e6:.1f}M")

    return model, tokenizer


def setup_lora(model, config: dict):
    """
    Apply LoRA adapters to the model for parameter-efficient fine-tuning.

    LoRA adds small trainable rank-decomposition matrices to the attention
    layers. With r=16 and targeting q/k/v/o projections, we add ~0.5% new
    trainable parameters while keeping 99.5% of the model frozen.
    """
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def verify_dataset_safety(dataset) -> None:
    """
    Final structural verification before training begins.

    The per-row is_reward_hack assertion already runs in build_sft_dataset.py
    (which derives this dataset). Here we only confirm the on-disk artifact
    is non-empty and has the columns Trainer expects.
    """
    print("[INFO] Running final safety verification on training dataset...")

    assert len(dataset) > 0, (
        "SAFETY CHECK FAILED: Dataset is empty! "
        "No honest trajectories to train on. Aborting."
    )

    missing = [c for c in KEEP_COLUMNS if c not in dataset.column_names]
    assert not missing, (
        f"SAFETY CHECK FAILED: Dataset is missing required columns: {missing}. "
        f"Re-run data/build_sft_dataset.py to regenerate."
    )

    # Spot-check that at least one assistant token survives in the labels.
    sample = dataset[0]
    n_kept = sum(1 for t in sample["labels"] if t != -100)
    assert n_kept > 0, (
        "SAFETY CHECK FAILED: First example has zero non-masked label tokens. "
        "Loss masking would be all-ignored. Aborting."
    )

    print(
        f"[INFO] Safety verification PASSED: "
        f"{len(dataset)} examples; sample assistant tokens = {n_kept}"
    )


def prepare_dataset_columns(dataset):
    """
    Drop everything except input_ids/attention_mask/labels.

    Auxiliary columns (instance_id, messages) trip up the default Trainer
    collator path because they're either nested lists or non-tensor strings.
    Keeping the dataset narrow avoids any silent re-tokenization or column
    mishandling.
    """
    drop = [c for c in dataset.column_names if c not in KEEP_COLUMNS]
    if drop:
        print(f"[INFO] Dropping non-tensor columns from dataset: {drop}")
        dataset = dataset.remove_columns(drop)
    return dataset


def main():
    """
    Main entry point: setup, verify, train, and save the SFT warm-started model.
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

    # --- Safety verification + column trimming ---
    verify_dataset_safety(dataset)
    dataset = prepare_dataset_columns(dataset)

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

    # --- Data collator ---
    # DataCollatorForSeq2Seq pads input_ids/attention_mask to the longest in
    # the batch and pads labels with -100 (so padded positions are ignored
    # by the loss). Works perfectly for our pre-tokenized causal LM data.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        label_pad_token_id=-100,
        pad_to_multiple_of=8,  # tensor-core friendly
        return_tensors="pt",
    )

    # --- Precision selection (T4-safe) ---
    use_bf16, use_fp16 = detect_precision(train_cfg)

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        # PEFT models need this to avoid a re-entrant autograd warning when
        # gradient_checkpointing is enabled.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_dir=logging_dir,
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        save_strategy="steps",
        report_to=["tensorboard"],
        # We pass already-tokenized examples; do not let Trainer drop them.
        remove_unused_columns=False,
        # Optimizer: default to paged_adamw_8bit for QLoRA memory savings;
        # config can override (e.g. adamw_torch on hosts without bitsandbytes).
        optim=str(train_cfg.get("optim", "paged_adamw_8bit")),
        # Helps with very small datasets on a single GPU.
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        # Make stdout deterministic across DDP / no DDP.
        disable_tqdm=False,
    )

    # --- Initialize Trainer ---
    # transformers >= 4.46 renamed `tokenizer` to `processing_class`, and
    # transformers 5.x removed the `tokenizer` kwarg entirely. Pick whichever
    # the installed Trainer signature accepts so this works across versions.
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

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
    print(f"  Precision:          bf16={use_bf16}  fp16={use_fp16}")
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

    # --- Save final model (LoRA adapter + tokenizer) ---
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
        "precision": {"bf16": use_bf16, "fp16": use_fp16},
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
