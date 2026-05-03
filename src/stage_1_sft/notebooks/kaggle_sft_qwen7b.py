"""
Kaggle driver for the SFT warm-start pipeline (Qwen2.5-Coder-7B-Instruct).

How to use
==========
1. New Kaggle notebook -> Settings -> Accelerator: "GPU T4 x2" (preferred)
   or "GPU P100". Persistence: "Files only".
2. Upload `trajectories.jsonl` and `trajectories2.jsonl` as a Kaggle dataset
   (e.g. "coding-hack-trajectories"), then attach it to the notebook. They
   appear at /kaggle/input/<dataset-slug>/.
3. Edit REPO_URL and DATASET_SLUG in CELL 1 below to your values.
4. Copy each CELL block into its OWN Kaggle cell, in order. Each cell
   starts at column 0 -- copy from the first line of the cell, not from
   inside it, otherwise Jupyter will throw IndentationError.
5. Real eval (mini-SWE-agent + SWE-bench harness) cannot run on Kaggle
   either (no Docker). After training, the LoRA adapter is surfaced in
   /kaggle/working/ so you can download it and run eval_sft.py on a Linux
   box with Docker.

Why `!` magic + `python -u`
---------------------------
- `!cmd` streams the child's stdout straight into the cell.
- `python -u` keeps the child Python unbuffered.

This file is documentation: the cells are inert triple-quoted strings.
"""

CELL_1_CLONE = r"""
REPO_URL = "https://github.com/Simardeep27/coding_hack.git"
BRANCH   = "smitha_patch"
DATASET_SLUG = "coding-hack-trajectories"
!rm -rf /kaggle/working/coding_hack
!git clone --depth 1 -b $BRANCH $REPO_URL /kaggle/working/coding_hack
%cd /kaggle/working/coding_hack
"""

CELL_2_COPY_TRAJECTORIES = r"""
import os, shutil
SRC = f"/kaggle/input/{DATASET_SLUG}"
DATA_DIR = "/kaggle/working/coding_hack/src/stage_1_sft/data"
os.makedirs(DATA_DIR, exist_ok=True)
for fname in ("trajectories.jsonl", "trajectories2.jsonl"):
    src = os.path.join(SRC, fname)
    dst = os.path.join(DATA_DIR, fname)
    assert os.path.exists(src), f"Missing {src} -- attach the Kaggle dataset"
    if not os.path.exists(dst):
        shutil.copy(src, dst)
print("OK ->", DATA_DIR)
"""

CELL_3_DEPS = r"""
!pip install torch>=2.1.0 "transformers>=4.45.0" "datasets>=3.0.0" "accelerate>=1.0.0" "tokenizers>=0.20.0" "peft>=0.13.0" "bitsandbytes>=0.44.0" "pyyaml>=6.0" "tensorboard>=2.15.0" "sentencepiece>=0.2.0" "protobuf>=4.25.0"
!pip install -e /kaggle/working/coding_hack
"""

CELL_4_MERGE_PROCESS = r"""
%cd /kaggle/working/coding_hack/src/stage_1_sft
!python -u -m data.merge_trajectories  --config config/sft_config.yaml
!python -u -m data.process_trajectories --config config/sft_config.yaml
"""

CELL_5_TOKENIZE = r"""
!python -u -m data.build_sft_dataset --config config/sft_config.yaml --max_length 4096
"""

CELL_6_DRY_RUN = r"""
!rm -rf checkpoints/sft_warmstart
!python -u -m training.sft_train --config config/sft_config.yaml --dry_run
"""

CELL_7_FULL_TRAIN = r"""
!python -u -m training.sft_train --config config/sft_config.yaml
"""

CELL_8_EXPORT_ADAPTER = r"""
import os, shutil
src = "/kaggle/working/coding_hack/src/stage_1_sft/checkpoints/sft_warmstart/final"
dst = "/kaggle/working/sft_warmstart_final"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print("Adapter at /kaggle/working/sft_warmstart_final (Output panel).")
"""


if __name__ == "__main__":
    cells = [
        ("CELL 1 -- Clone repo (edit REPO_URL / DATASET_SLUG)", CELL_1_CLONE),
        ("CELL 2 -- Copy attached trajectories into the repo",  CELL_2_COPY_TRAJECTORIES),
        ("CELL 3 -- Install deps",                              CELL_3_DEPS),
        ("CELL 4 -- Merge + filter + split",                    CELL_4_MERGE_PROCESS),
        ("CELL 5 -- Tokenize SFT dataset",                      CELL_5_TOKENIZE),
        ("CELL 6 -- Wipe checkpoint + dry run",                 CELL_6_DRY_RUN),
        ("CELL 7 -- Full training",                             CELL_7_FULL_TRAIN),
        ("CELL 8 -- Export LoRA adapter to /kaggle/working",    CELL_8_EXPORT_ADAPTER),
    ]
    for title, body in cells:
        print("=" * 70)
        print(title)
        print("=" * 70)
        print(body.strip())
        print()
