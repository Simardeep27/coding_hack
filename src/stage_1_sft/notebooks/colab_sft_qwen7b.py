"""
Colab driver for the SFT warm-start pipeline (Qwen2.5-Coder-7B-Instruct).

How to use
==========
1. Open https://colab.research.google.com, then Runtime -> Change runtime
   type -> Hardware accelerator: T4 GPU.
2. Edit REPO_URL in CELL 1 to point at your fork.
3. Copy each CELL block below into its OWN Colab cell, in order. Each cell
   starts at column 0 -- copy from the line that begins the cell, not from
   inside it, otherwise Jupyter will throw IndentationError.
4. Real eval (mini-SWE-agent + SWE-bench harness) cannot run on Colab
   (no Docker). After training, the final cell copies the LoRA adapter to
   Google Drive so you can download it and run eval_sft.py on a Linux box
   with Docker.

Why `!` magic + `python -u`
---------------------------
- `!cmd` streams the child's stdout straight into the cell. No buffering.
- `python -u` disables Python's stdout buffering inside the child too,
  so `[INFO]` lines and `tqdm` bars appear live.

This file is documentation: the cells are inert triple-quoted strings.
Open it in the editor, paste each block into Colab, and run.
"""

CELL_1_CLONE = r"""
REPO_URL = "https://github.com/Simardeep27/coding_hack.git"
BRANCH   = "smitha_patch"
!rm -rf /content/coding_hack
!git clone --depth 1 -b $BRANCH $REPO_URL /content/coding_hack
%cd /content/coding_hack
"""

CELL_2_DEPS = r"""
!pip install torch>=2.1.0 "transformers>=4.45.0" "datasets>=3.0.0" "accelerate>=1.0.0" "tokenizers>=0.20.0" "peft>=0.13.0" "bitsandbytes>=0.44.0" "pyyaml>=6.0" "tensorboard>=2.15.0" "sentencepiece>=0.2.0" "protobuf>=4.25.0"
!pip install -e /content/coding_hack
"""

CELL_3_CHECK_INPUTS = r"""
import os
DATA_DIR = "/content/coding_hack/src/stage_1_sft/data"
for f in ["trajectories.jsonl", "trajectories2.jsonl"]:
    p = os.path.join(DATA_DIR, f)
    assert os.path.exists(p), f"Missing {p} -- upload via the Files panel"
print("OK")
"""

CELL_4_MERGE_PROCESS = r"""
%cd /content/coding_hack/src/stage_1_sft
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

CELL_8_SAVE_TO_DRIVE = r"""
from google.colab import drive
drive.mount("/content/drive")
import shutil, os
src = "/content/coding_hack/src/stage_1_sft/checkpoints/sft_warmstart/final"
dst = "/content/drive/MyDrive/sft_warmstart_final"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print("Saved:", dst)
"""


if __name__ == "__main__":
    cells = [
        ("CELL 1 -- Clone repo (edit REPO_URL!)", CELL_1_CLONE),
        ("CELL 2 -- Install deps",                CELL_2_DEPS),
        ("CELL 3 -- Verify trajectory inputs",    CELL_3_CHECK_INPUTS),
        ("CELL 4 -- Merge + filter + split",      CELL_4_MERGE_PROCESS),
        ("CELL 5 -- Tokenize SFT dataset",        CELL_5_TOKENIZE),
        ("CELL 6 -- Wipe checkpoint + dry run",   CELL_6_DRY_RUN),
        ("CELL 7 -- Full training",               CELL_7_FULL_TRAIN),
        ("CELL 8 -- Save LoRA adapter to Drive",  CELL_8_SAVE_TO_DRIVE),
    ]
    for title, body in cells:
        print("=" * 70)
        print(title)
        print("=" * 70)
        print(body.strip())
        print()
