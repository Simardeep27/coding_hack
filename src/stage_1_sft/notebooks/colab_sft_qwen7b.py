"""
Colab driver for the SFT warm-start pipeline (Qwen2.5-Coder-7B-Instruct).

How to use:
    1. Open https://colab.research.google.com and start a fresh notebook.
    2. Runtime -> Change runtime type -> Hardware accelerator: T4 GPU
       (or A100 if you have Colab Pro+; T4 16 GB is the supported budget here).
    3. Copy the cells below into the notebook in order, or upload this file
       and run `!python colab_sft_qwen7b.py` after providing the trajectories
       at /content/coding_hack/src/stage_1_sft/data/.
    4. Generation/eval against the SWE-bench harness CANNOT run on Colab
       (no Docker). After training, download
       checkpoints/sft_warmstart/final/ and run eval_sft.py on a Linux box
       with Docker. Use `--predictions-only` here if you only want to dry-run
       the agent loop (no hidden_rate/visible_rate).

The script is also valid as a single .py invocation; each block is a clear,
copy-pasteable cell.
"""

# =============================================================================
# CELL 1 -- Repo + deps
# =============================================================================
import os, subprocess, sys

REPO_URL = os.environ.get("REPO_URL", "https://github.com/<your-user>/coding_hack.git")
REPO_DIR = "/content/coding_hack"
BRANCH   = os.environ.get("BRANCH", "smitha_patch")

if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
print("CWD:", os.getcwd())

# Install pinned deps. bitsandbytes is needed for 4-bit QLoRA on a T4.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch>=2.1.0",
     "transformers>=4.45.0",
     "datasets>=3.0.0",
     "accelerate>=1.0.0",
     "tokenizers>=0.20.0",
     "peft>=0.13.0",
     "bitsandbytes>=0.44.0",
     "pyyaml>=6.0",
     "tensorboard>=2.15.0",
     "sentencepiece>=0.2.0",
     "protobuf>=4.25.0"],
    check=True,
)
# Also install the local benchmark_agents package so the eval CLI is on PATH.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", REPO_DIR], check=True)

# =============================================================================
# CELL 2 -- Trajectory inputs
# =============================================================================
# Either the trajectory files were committed to the repo (default), or you
# upload them via the Colab Files panel into:
#   /content/coding_hack/src/stage_1_sft/data/trajectories.jsonl
#   /content/coding_hack/src/stage_1_sft/data/trajectories2.jsonl
DATA_DIR = os.path.join(REPO_DIR, "src/stage_1_sft/data")
for f in ["trajectories.jsonl", "trajectories2.jsonl"]:
    p = os.path.join(DATA_DIR, f)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing {p}. Upload it via the Colab Files panel before continuing."
        )
print("Trajectories present.")

# =============================================================================
# CELL 3 -- Merge + filter + split
# =============================================================================
os.chdir(os.path.join(REPO_DIR, "src/stage_1_sft"))
subprocess.run(
    [sys.executable, "-m", "data.merge_trajectories", "--config", "config/sft_config.yaml"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "data.process_trajectories", "--config", "config/sft_config.yaml"],
    check=True,
)

# =============================================================================
# CELL 4 -- Tokenize (use 4096 max-length on T4)
# =============================================================================
subprocess.run(
    [sys.executable, "-m", "data.build_sft_dataset",
     "--config", "config/sft_config.yaml", "--max_length", "4096"],
    check=True,
)

# =============================================================================
# CELL 5 -- Train (LoRA + 4-bit QLoRA)
# =============================================================================
# T4 / 16 GB: keep per_device_train_batch_size=1, gradient_accumulation_steps=8,
# max_length=4096. The default config matches.
# Set DRY_RUN=1 in env for a 10-step smoke test before the full run.
extra = ["--dry_run"] if os.environ.get("DRY_RUN") == "1" else []
subprocess.run(
    [sys.executable, "-m", "training.sft_train",
     "--config", "config/sft_config.yaml", *extra],
    check=True,
)

# =============================================================================
# CELL 6 -- Save the adapter to Drive (optional)
# =============================================================================
# Mount Drive and persist the LoRA adapter so you can download it for eval
# on a Docker-equipped box.
try:
    from google.colab import drive   # type: ignore
    drive.mount("/content/drive")
    import shutil
    src = os.path.join(REPO_DIR, "src/stage_1_sft/checkpoints/sft_warmstart/final")
    dst = "/content/drive/MyDrive/sft_warmstart_final"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Adapter copied to: {dst}")
except Exception as e:
    print(f"[skip] Drive copy unavailable: {e}")

# =============================================================================
# CELL 7 -- (Optional) predictions-only dry run on Colab
# =============================================================================
# Real eval requires Docker (mini-SWE-agent + SWE-bench harness). Colab does
# not provide Docker, so the harness step is unavailable. The block below
# only generates predictions for inspection; it does NOT compute hidden_rate
# / visible_rate. Use a Linux+Docker box for the full eval.
SKIP_PREDICTIONS = os.environ.get("SKIP_PREDICTIONS", "1") == "1"
if not SKIP_PREDICTIONS:
    print("WARNING: predictions-only mode -- hidden/visible rates will be unavailable.")
    subprocess.run(
        [sys.executable, "-m", "evaluation.eval_sft",
         "--config", "config/sft_config.yaml",
         "--predictions-only"],
        check=False,
    )
print("Done. Download the adapter (or read from Drive) and run real eval on a Docker host.")
