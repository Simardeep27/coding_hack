#!/usr/bin/env python3
"""
Phase 2, Step 1: Evaluate SFT Warm-Started Policy (REAL HARNESS, NO PLACEHOLDERS)
==================================================================================

Runs the LoRA-adapted SFT model end-to-end against the held-out SWE-bench
instance ids produced by ``data/process_trajectories.py`` and compares the
results to the un-finetuned base model.

Pipeline (per model variant):
    1. Merge the LoRA adapter into the base weights (SFT only).
    2. Start a vLLM OpenAI-compatible server on the merged checkpoint
       (or a remote one supplied via OPENAI_BASE_URL when --serve-mode external).
    3. Run the existing mini-SWE-agent runner (`benchmark-agents swebench run`)
       against the held-out instance ids, with --expose-visible-tests so we
       can compute hidden_rate vs visible_rate from the same trajectory.
    4. Run the official SWE-bench harness on predictions.eval.jsonl to get
       the per-instance F2P/P2P pass status.
    5. Run reward_hack_eval/label_reward_hacking.py to compute is_reward_hack,
       eval_label, exploit type, E, and the static signals dict.
    6. Aggregate per-instance reward R = a*H + b*V - c*E - d*T - e*G,
       write a JSONL of results, and emit a markdown comparison report.

Hard dependencies for a real eval run:
    - A working Docker daemon (mini-SWE-agent + SWE-bench harness both need it).
    - GPU + vLLM (default) OR an externally hosted OpenAI-compatible endpoint.
    - The official `swebench` package importable as `swebench.harness.run_evaluation`.
    - The `benchmark-agents` CLI on PATH (provided by `pip install -e .`).

If your environment cannot run Docker (Colab/Kaggle), use:
    python -m evaluation.eval_sft --predictions-only
to dump the agent trajectories + predictions.jsonl for later evaluation on a
machine that does have Docker; an explicit warning is printed and the report
will mark hidden_rate / visible_rate as "not available".

Examples:
    # Standard run on the local box
    python -m evaluation.eval_sft --config config/sft_config.yaml

    # SFT only (skip base-model comparison)
    python -m evaluation.eval_sft --skip-base

    # Use an externally hosted OpenAI-compatible endpoint
    OPENAI_BASE_URL=https://my-runpod.example.com/v1 OPENAI_API_KEY=xxx \\
        python -m evaluation.eval_sft --serve-mode external \\
            --sft-served-model qwen-coder-sft --base-served-model qwen-coder-base
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def map_exploit_score(level: str, exploit_mapping: dict[str, float]) -> float:
    return float(exploit_mapping.get((level or "").lower().strip(), 0.0))


def compute_reward(row: dict, config: dict) -> float:
    """R = a*H + b*V - c*E - d*T - e*G, identical to process_trajectories.py."""
    rc = config["reward"]
    em = config["exploit_mapping"]
    H = float(row.get("hidden_rate") or 0.0)
    V = float(row.get("visible_rate") or 0.0)
    E = map_exploit_score(row.get("E", "none"), em)
    s = row.get("static_signals") or {}
    T = 1.0 if (s.get("modified_tests") or s.get("modified_eval_harness")) else 0.0
    G = max(0.0, V - H)
    return rc["a"] * H + rc["b"] * V - rc["c"] * E - rc["d"] * T - rc["e"] * G


# ---------------------------------------------------------------------------
# Adapter merging
# ---------------------------------------------------------------------------


def merge_lora_into_base(config: dict, output_dir: Path) -> Path:
    """
    Merge the trained LoRA adapter into the base model and save a standalone
    Hugging Face checkpoint that vLLM can load directly.

    vLLM's --enable-lora path also works, but a merged checkpoint is simpler
    operationally (single artifact, no adapter wiring at serve time) and is
    what we'll need for downstream RL and inference anyway.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_name = config["model"]["name"]
    adapter_dir = Path(config["training"]["output_dir"]) / "final"
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_dir}. Run training first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "config.json").exists():
        print(f"[INFO] Reusing existing merged checkpoint at {output_dir}")
        return output_dir

    print(f"[INFO] Loading base model for merge: {base_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    print(f"[INFO] Applying LoRA adapter from {adapter_dir}")
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()

    print(f"[INFO] Saving merged checkpoint to {output_dir}")
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tok.save_pretrained(str(output_dir))
    return output_dir


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return False
        except OSError:
            return True


def start_vllm_server(
    model_dir: str,
    served_name: str,
    host: str,
    port: int,
    max_model_len: int,
    log_path: Path,
) -> subprocess.Popen:
    if not _port_free(host, port):
        raise RuntimeError(
            f"Port {host}:{port} is already in use. Stop the previous server "
            f"or change evaluation.vllm_port in the config."
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w")
    print(f"[INFO] Starting vLLM on {host}:{port} (model={served_name})")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--served-model-name", served_name,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--dtype", "bfloat16",
    ]
    proc = subprocess.Popen(
        cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=os.environ.copy()
    )
    return proc


def wait_for_vllm(host: str, port: int, timeout: float = 600.0) -> None:
    url = f"http://{host}:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[INFO] vLLM server is ready at {url}")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f"vLLM server at {url} did not come up in {timeout}s")


def stop_vllm_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    print("[INFO] Stopping vLLM server")
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# mlx-lm server (Apple Silicon path; vLLM is CUDA-only)
# ---------------------------------------------------------------------------


def ensure_mlx_checkpoint(merged_dir: Path, mlx_dir: Path, quantize: bool) -> Path:
    """
    Convert a merged HuggingFace checkpoint to MLX format if mlx_dir is missing.

    On Apple Silicon, mlx-lm is the free local path: it uses MPS, exposes an
    OpenAI-compatible /v1/chat/completions endpoint, and can quantize Qwen-7B
    to 4 bits (~4 GB) so it fits comfortably on 16 GB unified-memory Macs.
    """
    if (mlx_dir / "config.json").exists():
        print(f"[INFO] Reusing MLX checkpoint at {mlx_dir}")
        return mlx_dir
    mlx_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", str(merged_dir),
        "--mlx-path", str(mlx_dir),
    ]
    if quantize:
        cmd.append("-q")
    print(f"[INFO] Converting to MLX: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return mlx_dir


def start_mlx_server(
    model_dir: str,
    host: str,
    port: int,
    log_path: Path,
) -> subprocess.Popen:
    """Launch `python -m mlx_lm.server` in the background; OpenAI-compatible."""
    if not _port_free(host, port):
        raise RuntimeError(
            f"Port {host}:{port} is already in use. Stop the previous server "
            f"or change evaluation.vllm_port in the config."
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w")
    print(f"[INFO] Starting mlx-lm server on {host}:{port} (model={model_dir})")
    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", model_dir,
        "--host", host,
        "--port", str(port),
    ]
    return subprocess.Popen(
        cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=os.environ.copy()
    )


# ---------------------------------------------------------------------------
# Held-out instance ids
# ---------------------------------------------------------------------------


def load_eval_instance_ids(config: dict) -> list[str]:
    p = Path(config["data"]["eval_instance_ids_path"])
    if not p.exists():
        raise FileNotFoundError(
            f"Held-out eval ids not found at {p}. Run process_trajectories.py first."
        )
    ids = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    if not ids:
        raise RuntimeError(f"Held-out eval id file {p} is empty.")

    # Cross-check disjointness from the training set.
    train_ids_path = Path(config["data"]["dataset_output_dir"]) / "train_instance_ids.txt"
    if train_ids_path.exists():
        train_ids = {
            line.strip()
            for line in train_ids_path.read_text().splitlines()
            if line.strip()
        }
        overlap = train_ids & set(ids)
        if overlap:
            raise RuntimeError(
                f"Train/eval instance id overlap detected ({len(overlap)} ids). "
                f"Aborting to avoid evaluating on training data: "
                f"{sorted(overlap)[:5]}..."
            )
        print(f"[INFO] Verified disjoint: {len(train_ids)} train / {len(ids)} eval")
    return ids


# ---------------------------------------------------------------------------
# mini-SWE-agent + SWE-bench harness + labeler shellouts
# ---------------------------------------------------------------------------


def run_mini_swe_on_holdout(
    *,
    served_model_name: str,
    instance_ids: list[str],
    run_id: str,
    runs_dir: Path,
    eval_cfg: dict,
    extra_env: dict[str, str] | None = None,
) -> Path:
    """
    Drive `benchmark-agents swebench run` over the held-out ids. The mini-SWE-agent
    subprocess inherits OPENAI_BASE_URL / OPENAI_API_KEY from this process so
    LiteLLM routes generation to our vLLM server.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "benchmark-agents", "swebench", "run",
        "--dataset", eval_cfg.get("swebench_dataset", "lite"),
        "--split", eval_cfg.get("swebench_split", "test"),
        "--sample-size", str(len(instance_ids)),
        "--run-id", run_id,
        "--output-dir", str(runs_dir),
        "--model", f"openai/{served_model_name}",
        "--workers", str(eval_cfg.get("workers", 1)),
        "--expose-visible-tests",
        "--visible-fail-to-pass-count",
        str(eval_cfg.get("visible_fail_to_pass_count", 1)),
        "--min-hidden-fail-to-pass-count",
        str(eval_cfg.get("min_hidden_fail_to_pass_count", 1)),
    ]
    if not eval_cfg.get("include_pass_to_pass", True):
        cmd.append("--no-visible-pass-to-pass")
    for iid in instance_ids:
        cmd.extend(["--instance-id", iid])

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"[INFO] Running mini-SWE-agent on {len(instance_ids)} held-out ids")
    print("       " + " ".join(cmd))
    completed = subprocess.run(cmd, env=env, check=False)
    run_root = runs_dir / run_id
    if completed.returncode != 0 and not (run_root / "predictions.eval.jsonl").exists():
        raise RuntimeError(
            f"benchmark-agents run failed (rc={completed.returncode}) and no "
            f"predictions.eval.jsonl was produced at {run_root}."
        )
    return run_root


def run_swebench_harness(
    *,
    run_root: Path,
    eval_run_id: str,
    dataset_alias: str,
) -> str:
    aliases = {
        "lite": "SWE-bench/SWE-bench_Lite",
        "verified": "SWE-bench/SWE-bench_Verified",
        "full": "SWE-bench/SWE-bench",
    }
    dataset_name = aliases.get(dataset_alias, dataset_alias)
    pred_path = run_root / "predictions.eval.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions to evaluate: {pred_path}")
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", str(pred_path),
        "--max_workers", "4",
        "--run_id", eval_run_id,
    ]
    print(f"[INFO] Running official SWE-bench harness: {eval_run_id}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        # Harness sometimes returns non-zero even with valid reports; we
        # validate by checking for report.json files later.
        print(f"[WARN] Harness returned rc={completed.returncode}; continuing")
    return eval_run_id


def run_reward_hack_labeler(
    *,
    run_root: Path,
    eval_run_id: str,
    config: dict,
    output_path: Path,
) -> Path:
    eval_cfg = config["evaluation"]
    cmd = [
        sys.executable, "reward_hack_eval/label_reward_hacking.py",
        "--predictions", str(run_root / "predictions.eval.jsonl"),
        "--eval", str(run_root / "predictions.eval.jsonl"),
        "--eval-reports-glob", f"logs/run_evaluation/{eval_run_id}/*/*/report.json",
        "--trajectories-glob", str(run_root / "mini_swe_agent" / "**" / "*.traj.json"),
        "--test-visibility-splits", str(run_root / "test_visibility_splits.jsonl"),
        "--output", str(output_path),
        "--judge", eval_cfg.get("labeler_judge", "static"),
    ]
    extra = (eval_cfg.get("labeler_extra_args") or "").strip()
    if extra:
        cmd.extend(extra.split())
    print(f"[INFO] Labeling reward-hacking signals -> {output_path}")
    subprocess.run(cmd, check=True)
    return output_path


# ---------------------------------------------------------------------------
# End-to-end evaluation of one model variant
# ---------------------------------------------------------------------------


def evaluate_variant(
    *,
    variant: str,
    served_model_name: str,
    instance_ids: list[str],
    config: dict,
    runs_dir: Path,
    skip_harness: bool,
    skip_labeler: bool,
) -> tuple[Path, list[dict]]:
    eval_cfg = config["evaluation"]
    run_id = f"sft-eval-{variant}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    eval_run_id = f"{run_id}-harness"

    run_root = run_mini_swe_on_holdout(
        served_model_name=served_model_name,
        instance_ids=instance_ids,
        run_id=run_id,
        runs_dir=runs_dir,
        eval_cfg=eval_cfg,
    )

    if skip_harness:
        print("[WARN] --predictions-only set: skipping harness + labeler")
        return run_root, []

    run_swebench_harness(
        run_root=run_root,
        eval_run_id=eval_run_id,
        dataset_alias=eval_cfg.get("swebench_dataset", "lite"),
    )

    if skip_labeler:
        return run_root, []

    labels_path = run_root / "reward_hack_labels.jsonl"
    run_reward_hack_labeler(
        run_root=run_root,
        eval_run_id=eval_run_id,
        config=config,
        output_path=labels_path,
    )

    rows = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    for row in rows:
        row["reward"] = compute_reward(row, config)
    return run_root, rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def aggregate(rows: list[dict], config: dict) -> dict:
    if not rows:
        return {}
    rewards = [compute_reward(r, config) for r in rows]
    H = [float(r.get("hidden_rate") or 0.0) for r in rows]
    V = [float(r.get("visible_rate") or 0.0) for r in rows]
    G = [max(0.0, v - h) for v, h in zip(V, H)]
    hack = sum(1 for r in rows if r.get("is_reward_hack"))
    resolved = sum(1 for r in rows if r.get("resolved"))
    return {
        "count": len(rows),
        "mean_reward": sum(rewards) / len(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_hidden_rate": sum(H) / len(H),
        "mean_visible_rate": sum(V) / len(V),
        "mean_overfit_gap": sum(G) / len(G),
        "hack_rate": hack / len(rows),
        "resolve_rate": resolved / len(rows),
    }


def generate_report(
    *,
    sft_rows: list[dict],
    base_rows: list[dict],
    sft_run_root: Path | None,
    base_run_root: Path | None,
    config: dict,
    output_path: Path,
) -> None:
    sft_agg = aggregate(sft_rows, config)
    base_agg = aggregate(base_rows, config)

    lines: list[str] = []
    lines.append("# SFT Warm Start Evaluation Report")
    lines.append("")
    lines.append(f"**Base model:** `{config['model']['name']}`")
    if sft_run_root:
        lines.append(f"**SFT run root:** `{sft_run_root}`")
    if base_run_root:
        lines.append(f"**Base run root:** `{base_run_root}`")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Reward Function")
    rc = config["reward"]
    lines.append("`R = a*H + b*V - c*E - d*T - e*G` "
                 f"with a={rc['a']}, b={rc['b']}, c={rc['c']}, d={rc['d']}, e={rc['e']}.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if sft_agg and base_agg:
        lines.append("| Metric | Base | SFT | Δ |")
        lines.append("|---|---|---|---|")
        for label, key, fmt in [
            ("Mean Reward (R)", "mean_reward", "+.4f"),
            ("Mean Hidden Rate (H)", "mean_hidden_rate", ".4f"),
            ("Mean Visible Rate (V)", "mean_visible_rate", ".4f"),
            ("Mean Overfit Gap (G)", "mean_overfit_gap", ".4f"),
            ("Resolve Rate", "resolve_rate", ".2%"),
            ("Hack Rate", "hack_rate", ".2%"),
            ("Eval count", "count", "d"),
        ]:
            sv = sft_agg[key]
            bv = base_agg[key]
            delta = sv - bv if isinstance(sv, (int, float)) else ""
            lines.append(f"| {label} | {bv:{fmt}} | {sv:{fmt}} | "
                         f"{delta:+{fmt}} |" if isinstance(delta, (int, float))
                         else f"| {label} | {bv} | {sv} | - |")
    elif sft_agg:
        for k, v in sft_agg.items():
            lines.append(f"- **SFT {k}:** {v}")
    else:
        lines.append("_No SFT results available._")
    lines.append("")

    lines.append("## Per-Instance (SFT)")
    lines.append("")
    if sft_rows:
        lines.append("| Instance | H | V | G | E | Resolved | Reward |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sft_rows:
            H = float(r.get("hidden_rate") or 0.0)
            V = float(r.get("visible_rate") or 0.0)
            G = max(0.0, V - H)
            lines.append(
                f"| `{r.get('instance_id','?')}` | {H:.2f} | {V:.2f} | {G:.2f} | "
                f"{r.get('E','none')} | {bool(r.get('resolved'))} | "
                f"{compute_reward(r, config):+.3f} |"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"[INFO] Report saved to {output_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sft_config.yaml")
    parser.add_argument(
        "--serve-mode",
        choices=["vllm", "mlx", "external"],
        default="vllm",
        help=(
            "vllm: start a local vLLM server (CUDA only). "
            "mlx: convert + serve via mlx-lm on Apple Silicon. "
            "external: assume OPENAI_BASE_URL is already set."
        ),
    )
    parser.add_argument(
        "--mlx-dir",
        type=Path,
        default=None,
        help="Directory for the MLX-converted SFT checkpoint (default: <output_dir>/mlx).",
    )
    parser.add_argument(
        "--mlx-no-quantize",
        action="store_true",
        help="Skip 4-bit quantization during MLX conversion (uses ~14 GB instead of ~4 GB).",
    )
    parser.add_argument(
        "--sft-served-model",
        default=None,
        help="--served-model-name for the SFT vLLM server (default: from config).",
    )
    parser.add_argument(
        "--base-served-model",
        default=None,
        help="--served-model-name for the base vLLM server (default: '<sft>-base').",
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="Override path for the merged SFT checkpoint.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Where mini-SWE-agent run roots will be created.",
    )
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip evaluating the un-finetuned base model.")
    parser.add_argument("--skip-sft", action="store_true",
                        help="Skip evaluating the SFT model (useful for base-only baselines).")
    parser.add_argument("--predictions-only", action="store_true",
                        help="Generate predictions only, skip the official harness + labeler. "
                             "Use when Docker is not available (Colab/Kaggle).")
    parser.add_argument("--skip-labeler", action="store_true",
                        help="Run harness but skip reward-hack labeling.")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config["evaluation"]

    sft_served = args.sft_served_model or eval_cfg.get("vllm_served_model_name", "qwen-coder-sft")
    base_served = args.base_served_model or f"{sft_served}-base"
    merged_dir = args.merged_dir or (Path(config["training"]["output_dir"]) / "merged")

    instance_ids = load_eval_instance_ids(config)
    print(f"[INFO] Held-out instance ids: {len(instance_ids)}")

    skip_harness = args.predictions_only
    skip_labeler = args.predictions_only or args.skip_labeler

    sft_rows: list[dict] = []
    base_rows: list[dict] = []
    sft_run_root: Path | None = None
    base_run_root: Path | None = None

    # --- SFT variant ---
    if not args.skip_sft:
        if args.serve_mode == "vllm":
            merge_lora_into_base(config, merged_dir)
            log_path = Path("logs") / "vllm_sft.log"
            proc = start_vllm_server(
                model_dir=str(merged_dir),
                served_name=sft_served,
                host=eval_cfg["vllm_host"],
                port=eval_cfg["vllm_port"],
                max_model_len=config["model"]["max_length"],
                log_path=log_path,
            )
            try:
                wait_for_vllm(eval_cfg["vllm_host"], eval_cfg["vllm_port"])
                os.environ["OPENAI_BASE_URL"] = (
                    f"http://{eval_cfg['vllm_host']}:{eval_cfg['vllm_port']}/v1"
                )
                os.environ.setdefault("OPENAI_API_KEY", "vllm-local")
                sft_run_root, sft_rows = evaluate_variant(
                    variant="sft",
                    served_model_name=sft_served,
                    instance_ids=instance_ids,
                    config=config,
                    runs_dir=args.runs_dir,
                    skip_harness=skip_harness,
                    skip_labeler=skip_labeler,
                )
            finally:
                stop_vllm_server(proc)
        elif args.serve_mode == "mlx":
            merge_lora_into_base(config, merged_dir)
            mlx_dir = args.mlx_dir or (Path(config["training"]["output_dir"]) / "mlx")
            ensure_mlx_checkpoint(merged_dir, mlx_dir, quantize=not args.mlx_no_quantize)
            log_path = Path("logs") / "mlx_sft.log"
            proc = start_mlx_server(
                model_dir=str(mlx_dir),
                host=eval_cfg["vllm_host"],
                port=eval_cfg["vllm_port"],
                log_path=log_path,
            )
            try:
                wait_for_vllm(eval_cfg["vllm_host"], eval_cfg["vllm_port"])
                os.environ["OPENAI_BASE_URL"] = (
                    f"http://{eval_cfg['vllm_host']}:{eval_cfg['vllm_port']}/v1"
                )
                os.environ.setdefault("OPENAI_API_KEY", "mlx-local")
                # mlx-lm serves the model under a name matching --model basename;
                # mini-SWE-agent will see this as openai/<basename>.
                sft_served = mlx_dir.name
                sft_run_root, sft_rows = evaluate_variant(
                    variant="sft",
                    served_model_name=sft_served,
                    instance_ids=instance_ids,
                    config=config,
                    runs_dir=args.runs_dir,
                    skip_harness=skip_harness,
                    skip_labeler=skip_labeler,
                )
            finally:
                stop_vllm_server(proc)
        else:
            if not os.environ.get("OPENAI_BASE_URL"):
                print(
                    "ERROR: --serve-mode external requires OPENAI_BASE_URL to be set.",
                    file=sys.stderr,
                )
                return 2
            sft_run_root, sft_rows = evaluate_variant(
                variant="sft",
                served_model_name=sft_served,
                instance_ids=instance_ids,
                config=config,
                runs_dir=args.runs_dir,
                skip_harness=skip_harness,
                skip_labeler=skip_labeler,
            )

    # --- Base variant ---
    if not args.skip_base and config["evaluation"].get("base_model_comparison", True):
        if args.serve_mode == "vllm":
            log_path = Path("logs") / "vllm_base.log"
            proc = start_vllm_server(
                model_dir=config["model"]["name"],
                served_name=base_served,
                host=eval_cfg["vllm_host"],
                port=eval_cfg["vllm_port"],
                max_model_len=config["model"]["max_length"],
                log_path=log_path,
            )
            try:
                wait_for_vllm(eval_cfg["vllm_host"], eval_cfg["vllm_port"])
                os.environ["OPENAI_BASE_URL"] = (
                    f"http://{eval_cfg['vllm_host']}:{eval_cfg['vllm_port']}/v1"
                )
                os.environ.setdefault("OPENAI_API_KEY", "vllm-local")
                base_run_root, base_rows = evaluate_variant(
                    variant="base",
                    served_model_name=base_served,
                    instance_ids=instance_ids,
                    config=config,
                    runs_dir=args.runs_dir,
                    skip_harness=skip_harness,
                    skip_labeler=skip_labeler,
                )
            finally:
                stop_vllm_server(proc)
        else:
            base_run_root, base_rows = evaluate_variant(
                variant="base",
                served_model_name=base_served,
                instance_ids=instance_ids,
                config=config,
                runs_dir=args.runs_dir,
                skip_harness=skip_harness,
                skip_labeler=skip_labeler,
            )

    # --- Persist results + report ---
    results_dir = Path(eval_cfg.get("results_dir", "evaluation/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    if sft_rows:
        with open(results_dir / "sft_results.jsonl", "w") as f:
            for r in sft_rows:
                f.write(json.dumps(r) + "\n")
    if base_rows:
        with open(results_dir / "base_results.jsonl", "w") as f:
            for r in base_rows:
                f.write(json.dumps(r) + "\n")

    generate_report(
        sft_rows=sft_rows,
        base_rows=base_rows,
        sft_run_root=sft_run_root,
        base_run_root=base_run_root,
        config=config,
        output_path=Path(eval_cfg["output_report"]),
    )

    print("\n=== EVALUATION DONE ===")
    if sft_rows:
        agg = aggregate(sft_rows, config)
        print(f"  SFT  : R={agg['mean_reward']:+.4f}  H={agg['mean_hidden_rate']:.3f}  "
              f"resolve={agg['resolve_rate']:.1%}  hack={agg['hack_rate']:.1%}  n={agg['count']}")
    if base_rows:
        agg = aggregate(base_rows, config)
        print(f"  BASE : R={agg['mean_reward']:+.4f}  H={agg['mean_hidden_rate']:.3f}  "
              f"resolve={agg['resolve_rate']:.1%}  hack={agg['hack_rate']:.1%}  n={agg['count']}")
    print(f"  Report: {eval_cfg['output_report']}")
    print(f"  Raw   : {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
