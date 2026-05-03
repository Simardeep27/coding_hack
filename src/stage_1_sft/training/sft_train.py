#!/usr/bin/env python3
"""
Phase 2, Step 1: SFT Warm Start via Vertex AI Supervised Tuning
================================================================

Submits a Vertex AI Supervised Tuning job that fine-tunes Gemini Flash on
honest, successful SWE-bench trajectories. Training runs entirely on
Google's infrastructure -- no local GPU is needed.

Pipeline:
    1. Load configuration and verify auth/dataset.
    2. (Optionally) create a GCS bucket for the tuning dataset.
    3. Upload data/vertex_sft_train.jsonl (and val) to gs://BUCKET/PREFIX/.
    4. Submit a `vertexai.tuning.sft.train()` job for the configured Flash model.
    5. Poll until the job ends.
    6. Save the tuned model name + endpoint to checkpoints/sft_warmstart/
       so eval_sft.py can pick it up.

Why Vertex tuning instead of local LoRA:
    The trajectory dataset was collected from Gemini (via Vertex AI) running
    inside mini-swe-agent. Gemini weights are not public, so we cannot do
    local LoRA. Vertex's hosted SFT does parameter-efficient adapter tuning
    on Google's hardware, returning a tuned endpoint we call via LiteLLM
    (`vertex_ai/projects/.../endpoints/ENDPOINT_ID`) -- the same code path
    mini-swe-agent already uses.

Authentication:
    Requires Application Default Credentials (`gcloud auth application-default
    login`) or a service-account JSON pointed to by GOOGLE_APPLICATION_CREDENTIALS.
    A bare Google AI Studio API key will NOT work for tuning.

Usage:
    python -m training.sft_train --config config/sft_config.yaml
    python -m training.sft_train --dry_run     # validates + uploads only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def read_env_file(path: Path) -> dict[str, str]:
    """Minimal .env reader (matches the one in mini_swe_agent/config.py)."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("\"'")
    return out


def resolve_vertex_settings(config: dict, env_file: Path) -> tuple[str, str, str, str]:
    """
    Return (project, location, bucket, bucket_prefix), pulling missing values
    from .env or environment variables. Aborts if anything is unresolved.
    """
    env_values = read_env_file(env_file)
    vertex = config["vertex"]

    project = (
        vertex.get("project")
        or env_values.get("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    location = (
        vertex.get("location")
        or env_values.get("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or "us-central1"
    )
    bucket = (
        vertex.get("bucket")
        or env_values.get("GCS_TUNING_BUCKET")
        or os.getenv("GCS_TUNING_BUCKET")
    )
    bucket_prefix = vertex.get("bucket_prefix") or "sft-warmstart"

    missing: list[str] = []
    if not project:
        missing.append("vertex.project (or GOOGLE_CLOUD_PROJECT)")
    if not bucket:
        missing.append("vertex.bucket (or GCS_TUNING_BUCKET)")
    if missing:
        print(
            "ERROR: Missing required Vertex settings: " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Set them in src/stage_1_sft/config/sft_config.yaml or in .env "
            "at the repo root.",
            file=sys.stderr,
        )
        sys.exit(2)

    return project, location, bucket, bucket_prefix


def ensure_bucket(project: str, location: str, bucket_name: str) -> None:
    """
    Create the GCS bucket if it does not exist. Idempotent.

    Vertex tuning requires the dataset to be in GCS in the same region as
    the tuning job (us-central1 for Gemini Flash supervised tuning).
    """
    from google.cloud import storage  # type: ignore
    from google.api_core.exceptions import Conflict, NotFound  # type: ignore

    client = storage.Client(project=project)
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"[INFO] GCS bucket already exists: gs://{bucket.name}")
        return
    except NotFound:
        pass

    print(f"[INFO] Creating GCS bucket gs://{bucket_name} in {location}...")
    try:
        client.create_bucket(bucket_name, location=location)
        print(f"[INFO] Created gs://{bucket_name}")
    except Conflict:
        print(f"[INFO] Bucket gs://{bucket_name} was created concurrently; reusing.")


def upload_to_gcs(
    project: str, bucket_name: str, local_path: Path, gcs_key: str
) -> str:
    """Upload one file and return its full gs:// URI."""
    from google.cloud import storage  # type: ignore

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_key)
    print(f"[INFO] Uploading {local_path} -> gs://{bucket_name}/{gcs_key}")
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{gcs_key}"


def count_jsonl_lines(path: Path) -> int:
    n = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def submit_tuning_job(
    *,
    project: str,
    location: str,
    source_model: str,
    train_uri: str,
    val_uri: str | None,
    epochs: int,
    learning_rate_multiplier: float,
    adapter_size: int,
    tuned_model_display_name: str,
):
    """
    Kick off a Vertex AI Supervised Tuning job for Gemini Flash.

    Returns the SftTuningJob handle (which is poll-able via .refresh()).
    """
    import vertexai  # type: ignore
    from vertexai.tuning import sft  # type: ignore

    vertexai.init(project=project, location=location)

    print()
    print("=" * 70)
    print("SUBMITTING VERTEX AI SFT TUNING JOB")
    print("=" * 70)
    print(f"  Project:           {project}")
    print(f"  Location:          {location}")
    print(f"  Source model:      {source_model}")
    print(f"  Train dataset:     {train_uri}")
    print(f"  Val dataset:       {val_uri or '(none)'}")
    print(f"  Epochs:            {epochs}")
    print(f"  LR multiplier:     {learning_rate_multiplier}")
    print(f"  Adapter size:      {adapter_size}")
    print(f"  Display name:      {tuned_model_display_name}")
    print("=" * 70)

    kwargs = dict(
        source_model=source_model,
        train_dataset=train_uri,
        epochs=epochs,
        learning_rate_multiplier=learning_rate_multiplier,
        adapter_size=adapter_size,
        tuned_model_display_name=tuned_model_display_name,
    )
    if val_uri:
        kwargs["validation_dataset"] = val_uri

    job = sft.train(**kwargs)
    print(f"[INFO] Submitted job: {job.resource_name}")
    return job


def poll_until_done(job, poll_seconds: int = 60) -> None:
    """Block until the tuning job ends, printing progress."""
    started = time.time()
    while not job.has_ended:
        elapsed = int(time.time() - started)
        print(
            f"[INFO] Tuning in progress... elapsed={elapsed}s "
            f"state={getattr(job, 'state', '?')}"
        )
        time.sleep(poll_seconds)
        job.refresh()
    print(f"[INFO] Tuning job ended. state={getattr(job, 'state', '?')}")


def save_tuned_model_metadata(
    config: dict, job, *, train_uri: str, val_uri: str | None
) -> Path:
    """
    Persist the tuned model's resource name and endpoint so eval_sft.py
    (and downstream RL) can find it without rerunning the job.
    """
    out_path = Path(config["data"]["tuned_model_metadata"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tuned_model_name": getattr(job, "tuned_model_name", None),
        "tuned_model_endpoint_name": getattr(job, "tuned_model_endpoint_name", None),
        "experiment": getattr(job, "experiment", None),
        "resource_name": getattr(job, "resource_name", None),
        "state": str(getattr(job, "state", None)),
        "source_model": config["model"]["source_model"],
        "tuned_model_display_name": config["model"]["tuned_model_display_name"],
        "train_dataset": train_uri,
        "val_dataset": val_uri,
        "tuning_hyperparameters": config["tuning"],
        "reward_coefficients": config["reward"],
        "filtering_thresholds": config["filtering"],
    }
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Submit a Vertex AI Supervised Tuning job for the SFT warm start."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sft_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env (default ./.env at the repo root)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Validate config, ensure bucket exists, upload dataset, but "
            "do NOT submit a tuning job. Use to verify auth + plumbing."
        ),
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Reuse already-uploaded GCS dataset; just submit the job.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    project, location, bucket, prefix = resolve_vertex_settings(
        config, env_file=Path(args.env_file)
    )

    # --- Verify dataset files exist and are non-empty ---
    train_path = Path(config["data"]["vertex_train_path"])
    val_path = Path(config["data"]["vertex_val_path"])

    if not train_path.exists():
        print(
            f"ERROR: Training file not found: {train_path}. Run "
            "`python -m data.build_sft_dataset` first.",
            file=sys.stderr,
        )
        sys.exit(2)

    n_train = count_jsonl_lines(train_path)
    n_val = count_jsonl_lines(val_path) if val_path.exists() else 0
    min_examples = int(config["dataset_limits"]["min_examples"])
    print(f"[INFO] Train examples: {n_train}; Val examples: {n_val}")
    assert n_train >= min_examples, (
        f"SAFETY CHECK FAILED: training set has {n_train} examples, "
        f"Vertex SFT requires at least {min_examples}."
    )

    # --- Bucket + upload ---
    train_uri: str | None = None
    val_uri: str | None = None
    if not args.skip_upload:
        ensure_bucket(project, location, bucket)
        train_key = f"{prefix}/{train_path.name}"
        train_uri = upload_to_gcs(project, bucket, train_path, train_key)
        if n_val > 0:
            val_key = f"{prefix}/{val_path.name}"
            val_uri = upload_to_gcs(project, bucket, val_path, val_key)
    else:
        train_uri = f"gs://{bucket}/{prefix}/{train_path.name}"
        if n_val > 0:
            val_uri = f"gs://{bucket}/{prefix}/{val_path.name}"
        print(f"[INFO] Skipping upload, reusing {train_uri}")

    if args.dry_run:
        print("\n[DRY RUN] Skipping tuning-job submission.")
        print("[DRY RUN] Auth, bucket, and upload all succeeded.")
        return

    # --- Submit + poll ---
    tcfg = config["tuning"]
    job = submit_tuning_job(
        project=project,
        location=location,
        source_model=config["model"]["source_model"],
        train_uri=train_uri,
        val_uri=val_uri,
        epochs=int(tcfg["epochs"]),
        learning_rate_multiplier=float(tcfg["learning_rate_multiplier"]),
        adapter_size=int(tcfg["adapter_size"]),
        tuned_model_display_name=config["model"]["tuned_model_display_name"],
    )

    poll_until_done(job)

    # --- Save metadata for downstream eval / RL ---
    out_path = save_tuned_model_metadata(
        config, job, train_uri=train_uri, val_uri=val_uri
    )

    print()
    print("=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    print(f"  Tuned model name:      {getattr(job, 'tuned_model_name', '?')}")
    print(f"  Tuned model endpoint:  {getattr(job, 'tuned_model_endpoint_name', '?')}")
    print(f"  Saved metadata:        {out_path}")
    print()
    print(
        "[NEXT] Evaluate the tuned model on previously-hacked instances:\n"
        "         python -m evaluation.eval_sft --config config/sft_config.yaml"
    )


if __name__ == "__main__":
    main()
