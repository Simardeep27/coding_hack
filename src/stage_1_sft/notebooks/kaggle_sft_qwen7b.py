"""
Kaggle driver for the SFT warm-start pipeline (Qwen2.5-Coder-7B-Instruct).

How to use:
    1. Create a new Kaggle notebook. Settings -> Accelerator: "GPU T4 x2"
       (preferred) or "GPU P100". Persistence: "Files only".
    2. Upload `trajectories.jsonl` and `trajectories2.jsonl` as a Kaggle
       dataset (e.g., "coding-hack-trajectories"), then attach it to the
       notebook. They will appear at /kaggle/input/<dataset-slug>/.
    3. In a single cell, paste this script. Adjust REPO_URL / DATASET_SLUG
       to your values.
    4. Generation/eval against the SWE-bench harness CANNOT run on Kaggle
       (no Docker). After training, download
       /kaggle/working/coding_hack/src/stage_1_sft/checkpoints/sft_warmstart/final/
       and run eval_sft.py on a Linux box with Docker.

The pipeline is the same as Colab; only the install + paths differ.
"""

import os, subprocess, sys, shutil

REPO_URL = os.environ.get("REPO_URL", "https://github.com/<your-user>/coding_hack.git")
BRANCH   = os.environ.get("BRANCH", "smitha_patch")
REPO_DIR = "/kaggle/working/coding_hack"
DATASET_SLUG = os.environ.get("DATASET_SLUG", "coding-hack-trajectories")

# -- Clone repo
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)

# -- Copy trajectories from Kaggle dataset into the repo data dir
SRC = f"/kaggle/input/{DATASET_SLUG}"
DATA_DIR = os.path.join(REPO_DIR, "src/stage_1_sft/data")
os.makedirs(DATA_DIR, exist_ok=True)
for fname in ("trajectories.jsonl", "trajectories2.jsonl"):
    src = os.path.join(SRC, fname)
    dst = os.path.join(DATA_DIR, fname)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Missing {src}. Add it to the Kaggle dataset {DATASET_SLUG} and "
            f"attach it to the notebook."
        )
    if not os.path.exists(dst):
        shutil.copy(src, dst)
print("Trajectories present at", DATA_DIR)

# -- Install pinned deps. Kaggle ships some preinstalled, but force-reinstall
#    bitsandbytes / peft / trl to known versions.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch>=2.1.0",
     "transformers>=4.45.0",
     "datasets>=3.0.0",
     "accelerate>=1.0.0",
     "tokenizers>=0.20.0",
     "peft>=0.13.0",
     "trl>=0.12.0",
     "bitsandbytes>=0.44.0",
     "pyyaml>=6.0",
     "tensorboard>=2.15.0",
     "sentencepiece>=0.2.0",
     "protobuf>=4.25.0"],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", REPO_DIR], check=True)

# -- Pipeline: merge -> filter+split -> tokenize -> train
os.chdir(os.path.join(REPO_DIR, "src/stage_1_sft"))
subprocess.run(
    [sys.executable, "-m", "data.merge_trajectories", "--config", "config/sft_config.yaml"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "data.process_trajectories", "--config", "config/sft_config.yaml"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "data.build_sft_dataset",
     "--config", "config/sft_config.yaml", "--max_length", "4096"],
    check=True,
)

# Kaggle T4 x2: enable device_map="auto" via accelerate; the existing trainer
# already supports gradient checkpointing + LoRA. P100 (16 GB) also fits at
# max_length=4096 with the default per-device batch size of 1.
extra = ["--dry_run"] if os.environ.get("DRY_RUN") == "1" else []
subprocess.run(
    [sys.executable, "-m", "training.sft_train",
     "--config", "config/sft_config.yaml", *extra],
    check=True,
)

# -- Surface the LoRA adapter at /kaggle/working/sft_warmstart_final so it
#    appears in the notebook's "Output" tab and can be downloaded.
src = os.path.join(REPO_DIR, "src/stage_1_sft/checkpoints/sft_warmstart/final")
dst = "/kaggle/working/sft_warmstart_final"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print("Adapter copied to /kaggle/working/sft_warmstart_final (visible in Output panel).")
print("Done. For real eval, download the adapter and run eval_sft.py on a Docker host.")
