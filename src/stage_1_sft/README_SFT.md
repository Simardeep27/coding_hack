# Phase 2, Step 1: Supervised Warm Start (SFT)

Fine-tunes a base code model on honest, successful SWE-bench trajectories
to create a non-exploitative prior for the integrity-aware RL pipeline.

## Overview

The SFT warm start ensures the policy begins from a state that:
1. Knows how to solve coding problems (learned from successful trajectories)
2. Does NOT exploit tests, hardcode answers, or tamper with evaluation
3. Generalizes well (high hidden test pass rate, low overfit gap)

This is critical for Phase 2 (online RL): starting from an honest prior
prevents the policy from falling into reward-hacking attractors.

## Prerequisites

- Python 3.11+
- NVIDIA GPU with 24GB+ VRAM (e.g., RTX 4090, A100 40GB)
- CUDA toolkit installed and compatible with PyTorch
- `data/trajectories.jsonl` in the repository (output from Phase 1)

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify all imports work
python -c "from transformers import AutoTokenizer; from peft import LoraConfig; from trl import SFTTrainer; print('All imports OK')"