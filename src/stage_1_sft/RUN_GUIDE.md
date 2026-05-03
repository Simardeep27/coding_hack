# Beginner Run Guide -- SFT Warm Start (Qwen2.5-Coder-7B-Instruct)

This walks you through the entire pipeline end to end, assuming you have
**no prior experience** with Colab, Kaggle, Docker, or model serving. Every
step is a literal action: a click, a paste, or a single command.

The intended split:

| Phase | Where | Why |
|---|---|---|
| Train (SFT) | Colab **or** Kaggle (free GPU) | Your Mac probably can't QLoRA-train a 7B model. Cloud GPUs are free for the limits we need. |
| Eval | Local Mac | Free, and the SWE-bench harness needs Docker on Linux/amd64 -- no cloud needed. |

---

## Part 0 -- One-time setup before anything else

You need exactly two things online: 

### 0.1 Push this repo to your GitHub  

-If the repo is only on your Mac, Colab/Kaggle can't see it. From the repo                                                                                      
      23 -root in Terminal:

git remote -v   # check what 'origin' currently points at
git remote set-url origin https://github.com/<YOUR-USER>/coding_hack.git
git push -u origin smitha_patch
```

If `git remote -v` already shows your own GitHub repo, just `git push` is
enough. The branch name is `smitha_patch`; the notebook drivers default to
that branch.

The trajectory files (`trajectories.jsonl`, `trajectories2.jsonl`) are
already committed, so once you push, Colab/Kaggle will see them.

### 0.2 Pick Colab OR Kaggle (you only need one)

- **Colab** is the simpler workflow: persistent files via Google Drive, one
  big notebook cell.
- **Kaggle** gives slightly more GPU time per week (~30 hours of T4 x2)
  but requires uploading the trajectories as a "Dataset" first.

Pick whichever you have an account for. Both are free.

---

## Part A -- Train on Colab (recommended for first try)

### A.1 Open a new notebook

1. Go to https://colab.research.google.com.
2. **File -> New notebook**.
3. Top right of the screen, click the **arrow next to "Connect"** ->
   **Change runtime type**.
4. Hardware accelerator: **T4 GPU**. (If you have Colab Pro+, pick A100 for
   speed.) Click **Save**.
5. Click **Connect** in the top right. Wait for the green check.

### A.2 Confirm the GPU is up

Paste this into the first cell, then press **Shift+Enter**:

```python
!nvidia-smi
```

You should see a Tesla T4 (or A100) with ~15 GB free. If you see "command
not found", your runtime didn't get a GPU -- redo step A.1.4.

### A.3 Run the training driver

Click the **+ Code** button to add a new cell. Paste **all of this** into
that one cell:

```python
import os, urllib.request

# ---- EDIT THESE TWO LINES ----
os.environ["REPO_URL"] = "https://github.com/<YOUR-USER>/coding_hack.git"
os.environ["BRANCH"]   = "smitha_patch"
# ------------------------------

# Pull the driver script straight out of the repo and exec it.
url = (os.environ["REPO_URL"]
       .replace("github.com", "raw.githubusercontent.com")
       .replace(".git", "")
       + f"/{os.environ['BRANCH']}/src/stage_1_sft/notebooks/colab_sft_qwen7b.py")
exec(urllib.request.urlopen(url).read())
```

Replace `<YOUR-USER>` with your GitHub username. Then **Shift+Enter**.

What happens (you'll see this in the cell output):

1. The repo gets cloned into `/content/coding_hack`.
2. Python deps install (`transformers`, `peft`, `trl`, `bitsandbytes`, ...).
3. The merge step combines `trajectories.jsonl` + `trajectories2.jsonl`
   into `merged_trajectories.jsonl` (119 unique).
4. The filter step produces 48 train / 13 eval rows and writes
   `data/sft_eval_instance_ids.txt`.
5. The tokenizer step converts trajectories to a HuggingFace dataset using
   the Qwen chat template.
6. Training starts. With LoRA + 4-bit on a T4, expect ~30-90 minutes for
   3 epochs over 48 examples. You'll see loss decrease in the output.

Total cell run time on T4: ~1-2 hours. Don't close the tab.

### A.4 First-try smoke test (optional, 5 minutes)

If you want to verify the whole pipeline works *before* committing to the
full run, run a 10-step dry-run first. Paste this into a fresh cell and run
it BEFORE step A.3:

```python
import os; os.environ["DRY_RUN"] = "1"
```

After it completes, remove the cell (or set DRY_RUN back to `0`) and
proceed to A.3 for real.

### A.5 Save the LoRA adapter to Drive

The Colab driver tries to mount Google Drive at the end and copies the
trained adapter there. When prompted **"Permit this notebook to access
your Google Drive files?"**, click **Connect to Google Drive** and follow
the auth popup. The adapter ends up at:

```
/content/drive/MyDrive/sft_warmstart_final/
```

That's what you'll download to your Mac for eval.

### A.6 Download the adapter to your Mac

1. Open https://drive.google.com -> find `sft_warmstart_final`.
2. Right-click -> **Download**. Drive will zip it up; expect ~200 MB.
3. On your Mac, unzip into the repo:
   ```bash
   cd /Users/smithakumar/690S/project/coding_hack
   mkdir -p src/stage_1_sft/checkpoints/sft_warmstart
   unzip ~/Downloads/sft_warmstart_final.zip \
         -d src/stage_1_sft/checkpoints/sft_warmstart/
   # the unzipped folder must be named exactly "final"
   mv src/stage_1_sft/checkpoints/sft_warmstart/sft_warmstart_final \
      src/stage_1_sft/checkpoints/sft_warmstart/final
   ls src/stage_1_sft/checkpoints/sft_warmstart/final
   # expected: adapter_config.json, adapter_model.safetensors, tokenizer.json, ...
   ```

Skip Part B and jump to **Part C** if you trained on Colab.

---

## Part B -- Train on Kaggle (alternative to Colab)

### B.1 Upload the trajectories as a Kaggle Dataset

1. Go to https://kaggle.com/datasets -> **+ New Dataset** (top right).
2. Name it exactly `coding-hack-trajectories`.
3. Drag in both files from your Mac:
   - `src/stage_1_sft/data/trajectories.jsonl`
   - `src/stage_1_sft/data/trajectories2.jsonl`
4. Visibility: Private.
5. Click **Create**. Wait for upload (~10 sec, the files are small).

### B.2 Create a notebook with GPU

1. Go to https://kaggle.com/code -> **+ New Notebook**.
2. Top right -> **Add data** -> search `coding-hack-trajectories` -> **+**.
3. Right sidebar -> **Settings** -> **Accelerator** -> pick `GPU T4 x2`.
   Persistence: `Files only`.

### B.3 Run the training driver

In the first code cell, paste this and run with **Shift+Enter**:

```python
import os, urllib.request

# ---- EDIT THESE TWO LINES ----
os.environ["REPO_URL"]    = "https://github.com/<YOUR-USER>/coding_hack.git"
os.environ["BRANCH"]      = "smitha_patch"
os.environ["DATASET_SLUG"] = "coding-hack-trajectories"
# ------------------------------

url = (os.environ["REPO_URL"]
       .replace("github.com", "raw.githubusercontent.com")
       .replace(".git", "")
       + f"/{os.environ['BRANCH']}/src/stage_1_sft/notebooks/kaggle_sft_qwen7b.py")
exec(urllib.request.urlopen(url).read())
```

Total cell run time on T4 x2: ~30-60 minutes.

### B.4 Download the adapter to your Mac

When the cell finishes, the adapter is at
`/kaggle/working/sft_warmstart_final/`. To download:

1. Right sidebar -> **Output** tab.
2. Find `sft_warmstart_final/`.
3. Click the **download** icon (looks like a cloud arrow).
4. Unzip on your Mac (same as A.6).

---

## Part C -- Set up Docker on your Mac (free)

The SWE-bench harness needs Docker. Two free options; pick one:

### Option C1 -- Colima (fully open-source, recommended)

```bash
brew install colima docker
colima start --cpu 4 --memory 8 --disk 60 --arch x86_64
docker info     # should print "Server Version: ..."
```

The first `colima start` takes a few minutes (downloads a Linux VM image).
After that it's instant. Stop with `colima stop` when you're done to free
up RAM.

### Option C2 -- Docker Desktop

1. Download from https://docker.com/products/docker-desktop.
2. Install the .dmg, open Docker Desktop, accept the license.
3. In a Terminal:
   ```bash
   docker info
   ```

Both work the same way after this point.

> **Apple Silicon note:** SWE-bench images are linux/amd64 and run under
> emulation on M1/M2/M3 Macs. Slower than native, but functional. Plan ~2 hr
> for the 13 eval instances on first run (subsequent runs reuse cached
> images and are much faster).

---

## Part D -- Set up the local model server (mlx-lm)

mlx-lm runs Qwen2.5-Coder-7B natively on Apple Silicon and exposes an
OpenAI-compatible HTTP endpoint. Free, no external services.

### D.1 Install mlx-lm

In your repo's virtual env:

```bash
cd /Users/smithakumar/690S/project/coding_hack
source .venv/bin/activate    # or however you enter your venv
pip install "mlx-lm>=0.19"
```

### D.2 Verify mlx-lm sees your GPU

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0)
```

If you see `Device(cpu, 0)`, you're on Intel Mac and mlx-lm will fall back
to CPU (much slower, but still works). MLX is most useful on M1/M2/M3.

You don't need to manually convert the merged checkpoint -- `eval_sft.py`
does that automatically the first time you run it with `--serve-mode mlx`.

---

## Part E -- Run the local eval

### E.1 Install the harness deps

```bash
cd /Users/smithakumar/690S/project/coding_hack
pip install -e .            # gives you the `benchmark-agents` CLI
pip install swebench        # the official harness
```

### E.2 Confirm Docker is up

```bash
docker info | head -5
# expected: Client: ... Server: Docker Engine ...
```

If this fails, redo Part C.

### E.3 Run eval_sft with mlx-lm serving

```bash
cd src/stage_1_sft
python -m evaluation.eval_sft \
    --config config/sft_config.yaml \
    --serve-mode mlx \
    --skip-base
```

What happens:

1. **Adapter merge** (~2 minutes, ~14 GB peak RAM). The LoRA adapter at
   `checkpoints/sft_warmstart/final/` is merged into the base Qwen weights
   and saved as `checkpoints/sft_warmstart/merged/`.
2. **MLX conversion** (~1 minute). The merged HF checkpoint is converted
   and 4-bit quantized to `checkpoints/sft_warmstart/mlx/` (~4 GB).
3. **mlx-lm server starts** on http://127.0.0.1:8000/v1.
4. **mini-SWE-agent runs** on each of the 13 held-out instances. Per
   instance: spin up a Docker container, run the agent loop against the
   mlx-lm server, dump trajectory + final patch.
5. **SWE-bench harness runs** on the patches. Spawns Docker containers,
   runs the FAIL_TO_PASS and PASS_TO_PASS tests, reports per-instance
   pass/fail.
6. **Reward-hack labeler runs** to compute `is_reward_hack`, `eval_label`,
   `E`, and the static signals.
7. **Markdown report** written to `evaluation/sft_eval_report.md` and
   per-row JSONL in `evaluation/results/`.

Expected total time on M1 Mac: 1-3 hours for all 13 instances on the first
run. We pass `--skip-base` so we don't double the time evaluating the
unfine-tuned base.

If anything dies midway, see **Troubleshooting** below.

### E.4 Read the report

Open `src/stage_1_sft/evaluation/sft_eval_report.md` in any markdown
viewer. You'll see:

- A summary table (Mean Reward, Hidden Rate, Visible Rate, Hack Rate, Resolve Rate).
- A per-instance table with R, H, V, gap, exploit level, resolved bool.

The first numbers you should look at:

- **Mean Hidden Rate (H)** -- did the SFT model actually solve real bugs?
  >0 means yes. <0.3 means SFT taught it the format but not the substance,
  which is normal for 48 training rows.
- **Hack Rate** -- 0% is the goal. >0 means the SFT prior leaked an
  exploit pattern. Check the per-instance table for which one.
- **Mean Overfit Gap (G)** -- low (<0.1) means the model generalizes; high
  means it passes visible tests but fails hidden ones.

### E.5 (Later) Compare to base model

Once everything works end to end, drop `--skip-base` to compare against
the un-finetuned Qwen. The base model gets a separate vLLM/mlx-lm session
and the same 13 instances:

```bash
python -m evaluation.eval_sft \
    --config config/sft_config.yaml \
    --serve-mode mlx
```

The report's "Δ" column tells you whether SFT actually moved the model in
the right direction.

---

## Troubleshooting

### "RuntimeError: CUDA out of memory" on Colab during training

You're on a 12 GB or 15 GB T4. Edit
`src/stage_1_sft/config/sft_config.yaml` before re-running:

```yaml
model:
  max_length: 2048      # was 4096
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16   # was 8
```

Push the change to GitHub, re-run the Colab cell.

### Docker daemon "Cannot connect" on Mac

Run `colima status`. If it says "Not running", run `colima start`. If
Docker Desktop, ensure the whale icon in the menubar is steady (not
animating).

### `swebench.harness.run_evaluation` errors with "platform mismatch"

Apple Silicon emulation issue. Make sure you started colima with
`--arch x86_64`, or in Docker Desktop -> **Settings** -> **General** ->
toggle **Use Rosetta** ON.

### mlx-lm server: "no module named mlx_lm"

You're not in the right venv. From the repo root:

```bash
which python3
source .venv/bin/activate
which python3   # should now point inside .venv
pip install "mlx-lm>=0.19"
```

### "OPENAI_BASE_URL is not set" / 401 errors during agent run

The mlx-lm server didn't come up before mini-SWE-agent started. Check
`logs/mlx_sft.log` for the real error. Common cause: another process is
on port 8000. Edit `evaluation.vllm_port` in the config to e.g. `8123`.

### The harness only writes a partial report.json

That happens when a Docker container couldn't pull its base image (network
hiccup) or ran out of disk. Free up disk and re-run only the failed
instances:

```bash
# Edit data/sft_eval_instance_ids.txt to keep only the failed ids,
# then re-run eval_sft.
```

### How do I shut everything down cleanly?

```bash
# Kill any leftover servers
pkill -f "mlx_lm.server"
pkill -f "vllm.entrypoints"
# Stop colima to free RAM (optional)
colima stop
```

---

## What this gives you

After Part E completes you have:

- A LoRA-adapted Qwen2.5-Coder-7B-Instruct ("the SFT warm start").
- A merged + quantized MLX checkpoint you can keep serving locally for
  ad-hoc debugging.
- A Markdown report and per-instance JSONL telling you whether the SFT
  prior is honest **and** competent on held-out problems.

That's exactly the artifact required to feed Phase 2 (RL).
