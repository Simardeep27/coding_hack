#!/usr/bin/env bash
set -euo pipefail

# Batch mini-SWE-agent runs, official SWE-bench evaluation, and reward-hack labels.
#
# Typical use:
#   scripts/batch_reward_hack_pipeline.sh
#   scripts/batch_reward_hack_pipeline.sh runs/<existing_run_id>
#
# Common overrides:
#   BATCH_SIZE=50 SEED=42 RUN_ID_PREFIX=lite-mini-visible scripts/batch_reward_hack_pipeline.sh
#   ID_RANGE=51-100 scripts/batch_reward_hack_pipeline.sh
#   RESUME_RUN_ROOT=/path/to/runs/<existing_run_id> scripts/batch_reward_hack_pipeline.sh
#   CLEANUP_DOCKER_IMAGES=0 scripts/batch_reward_hack_pipeline.sh
#   LABEL_EXTRA_ARGS="--judge llm --llm-provider vertex --llm-fallback static" scripts/batch_reward_hack_pipeline.sh
#
# State:
#   - Completed/evaluable instances are recorded in $STATE_DIR/completed_instance_ids.txt.
#   - LimitsExceeded/TimeoutExpired instances are recorded per run but are not completed.
#   - Cumulative labels are written to $CUMULATIVE_LABELS.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
    printf '%s\n' \
        'Usage:' \
        '  scripts/batch_reward_hack_pipeline.sh [existing-run-root]' \
        '' \
        'Without an argument, select a new batch and run all steps.' \
        'With an existing run root, skip selection/mini-SWE-agent and resume at evaluation.' \
        '' \
        'Environment:' \
        '  RESUME_RUN_ROOT=/path/to/runs/<existing_run_id>  Same as passing existing-run-root.'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "$#" -gt 1 ]]; then
    usage >&2
    exit 2
fi

DATASET="${DATASET:-lite}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-50}"
SEED="${SEED:-42}"
ID_RANGE="${ID_RANGE:-}"
RUNS_DIR="${RUNS_DIR:-runs}"
STATE_DIR="${STATE_DIR:-reward_hack_eval/batch_state}"
CUMULATIVE_LABELS="${CUMULATIVE_LABELS:-reward_hack_eval/cumulative_reward_hack_labels.jsonl}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-${DATASET}-mini-swe-split}"
RUN_ID_FROM_ENV="${RUN_ID:-}"
EVAL_RUN_ID_FROM_ENV="${EVAL_RUN_ID:-}"
RESUME_RUN_ROOT="${RESUME_RUN_ROOT:-}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-4}"

EXPOSE_VISIBLE_TESTS="${EXPOSE_VISIBLE_TESTS:-1}"
VISIBLE_FAIL_TO_PASS_COUNT="${VISIBLE_FAIL_TO_PASS_COUNT:-1}"
MIN_HIDDEN_FAIL_TO_PASS_COUNT="${MIN_HIDDEN_FAIL_TO_PASS_COUNT:-1}"
INCLUDE_PASS_TO_PASS="${INCLUDE_PASS_TO_PASS:-1}"

PYTHON_CMD="${PYTHON_CMD:-uv run python}"
BENCHMARK_AGENTS_CMD="${BENCHMARK_AGENTS_CMD:-uv run benchmark-agents}"
EVAL_PYTHON_CMD="${EVAL_PYTHON_CMD:-$PYTHON_CMD}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_LABEL="${SKIP_LABEL:-0}"
RUN_EXTRA_ARGS="${RUN_EXTRA_ARGS:-}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
LABEL_EXTRA_ARGS="${LABEL_EXTRA_ARGS:-}"
REPLACE_EXISTING_LABELS="${REPLACE_EXISTING_LABELS:-0}"
CLEANUP_DOCKER_IMAGES="${CLEANUP_DOCKER_IMAGES:-1}"

if [[ "$#" -eq 1 ]]; then
    if [[ -n "$RESUME_RUN_ROOT" && "$RESUME_RUN_ROOT" != "$1" ]]; then
        echo "Both RESUME_RUN_ROOT and an existing-run-root argument were provided, but they differ." >&2
        exit 2
    fi
    RESUME_RUN_ROOT="$1"
fi

if [[ -n "$RESUME_RUN_ROOT" ]]; then
    if [[ "$RESUME_RUN_ROOT" != /* ]]; then
        RESUME_RUN_ROOT="${ROOT_DIR}/${RESUME_RUN_ROOT}"
    fi
    if [[ ! -d "$RESUME_RUN_ROOT" ]]; then
        echo "Resume run root does not exist or is not a directory: $RESUME_RUN_ROOT" >&2
        exit 2
    fi
    RUN_ROOT="$(cd "$RESUME_RUN_ROOT" && pwd)"
    RUN_ID="$(basename "$RUN_ROOT")"
    if [[ -n "$RUN_ID_FROM_ENV" && "$RUN_ID_FROM_ENV" != "$RUN_ID" ]]; then
        echo "warning: ignoring RUN_ID=${RUN_ID_FROM_ENV}; resume run id is ${RUN_ID}" >&2
    fi
    RESUME_EXISTING_RUN=1
else
    RUN_ID="${RUN_ID_FROM_ENV:-${RUN_ID_PREFIX}-$(date -u +%Y%m%dT%H%M%SZ)}"
    RUN_ROOT="${RUNS_DIR}/${RUN_ID}"
    RESUME_EXISTING_RUN=0
fi
EVAL_RUN_ID="${EVAL_RUN_ID_FROM_ENV:-${RUN_ID}-eval}"

case "$DATASET" in
    lite) DATASET_NAME="SWE-bench/SWE-bench_Lite" ;;
    verified) DATASET_NAME="SWE-bench/SWE-bench_Verified" ;;
    full) DATASET_NAME="SWE-bench/SWE-bench" ;;
    *) DATASET_NAME="$DATASET" ;;
esac

EVAL_LOG_GLOB="logs/run_evaluation/${EVAL_RUN_ID}/*/*/report.json"
COMPLETED_IDS_FILE="${STATE_DIR}/completed_instance_ids.txt"
SELECTED_IDS_FILE="${STATE_DIR}/${RUN_ID}.selected_instance_ids.txt"
TRANSIENT_IDS_FILE="${RUN_ROOT}/transient_failed_instance_ids.txt"
EVALUABLE_IDS_FILE="${RUN_ROOT}/evaluable_instance_ids.txt"
RUN_LABELS_FILE="${RUN_ROOT}/reward_hack_labels.jsonl"
RUN_HISTORY_FILE="${STATE_DIR}/run_history.jsonl"

# shellcheck disable=SC2206
PYTHON=($PYTHON_CMD)
# shellcheck disable=SC2206
BENCHMARK_AGENTS=($BENCHMARK_AGENTS_CMD)
DOCKER_IMAGE_CLEANUP_DONE=0

ensure_selected_ids_file() {
    if [[ -s "$SELECTED_IDS_FILE" ]]; then
        return
    fi

    echo "== Inferring selected instance ids for resumed run"
    "${PYTHON[@]}" - "$RUN_ROOT" "$SELECTED_IDS_FILE" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
out_path = Path(sys.argv[2])

def unique(values):
    seen = set()
    result = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result

def ids_from_jsonl(path: Path) -> list[str]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        instance_id = row.get("instance_id")
        if instance_id:
            ids.append(str(instance_id))
    return ids

ids: list[str] = []
source = None

sampled_path = run_root / "sampled_instances.jsonl"
if sampled_path.exists():
    ids = ids_from_jsonl(sampled_path)
    source = sampled_path

manifest_path = run_root / "manifest.json"
if not ids and manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [str(item) for item in manifest.get("instance_ids", []) if item]
    source = manifest_path

eval_predictions_path = run_root / "predictions.eval.jsonl"
if not ids and eval_predictions_path.exists():
    ids = ids_from_jsonl(eval_predictions_path)
    source = eval_predictions_path

preds_path = run_root / "mini_swe_agent" / "preds.json"
if not ids and preds_path.exists():
    data = json.loads(preds_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key, row in data.items():
            if isinstance(row, dict):
                ids.append(str(row.get("instance_id") or key))
            else:
                ids.append(str(key))
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and row.get("instance_id"):
                ids.append(str(row["instance_id"]))
    source = preds_path

ids = unique(ids)
if not ids:
    raise SystemExit(f"Could not infer selected instance ids from {run_root}")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
print(f"selected={len(ids)} source={source}")
PY
}

ensure_eval_predictions_file() {
    if [[ -s "${RUN_ROOT}/predictions.eval.jsonl" ]]; then
        return
    fi

    echo "== Creating SWE-bench eval predictions from existing mini-SWE output"
    "${BENCHMARK_AGENTS[@]}" swebench eval-predictions --run-root "$RUN_ROOT"
}

cleanup_run_docker_images() {
    if [[ "$DOCKER_IMAGE_CLEANUP_DONE" == "1" ]]; then
        return
    fi
    DOCKER_IMAGE_CLEANUP_DONE=1

    if [[ "$CLEANUP_DOCKER_IMAGES" != "1" ]]; then
        echo "== CLEANUP_DOCKER_IMAGES=${CLEANUP_DOCKER_IMAGES}, keeping Docker images"
        return
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "== Docker CLI not found, skipping image cleanup"
        return
    fi
    if ! docker ps >/dev/null 2>&1; then
        echo "== Docker daemon not reachable, skipping image cleanup"
        return
    fi

    local mini_log="${RUN_ROOT}/mini_swe_agent/minisweagent.log"
    local image_list="${RUN_ROOT}/docker_images_used.txt"
    if [[ ! -s "$mini_log" ]]; then
        echo "== No mini-SWE-agent log found, skipping image cleanup"
        return
    fi

    "${PYTHON[@]}" - "$mini_log" "$image_list" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
images: list[str] = []
seen: set[str] = set()
pattern = re.compile(r"(?:docker\.io/)?swebench/[^\s'\"`]+")

for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    for image in pattern.findall(line):
        if image not in seen:
            seen.add(image)
            images.append(image)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(images) + ("\n" if images else ""), encoding="utf-8")
PY

    if [[ ! -s "$image_list" ]]; then
        echo "== No SWE-bench Docker images found for cleanup"
        return
    fi

    local removed=0
    local skipped=0
    local image
    echo "== Cleaning Docker images used by this run"
    while IFS= read -r image; do
        [[ -z "$image" ]] && continue
        if ! docker image inspect "$image" >/dev/null 2>&1; then
            skipped=$((skipped + 1))
            continue
        fi
        if docker image rm "$image"; then
            removed=$((removed + 1))
        else
            skipped=$((skipped + 1))
            echo "warning: could not remove Docker image ${image}; it may still be in use" >&2
        fi
    done < "$image_list"
    echo "docker_images_removed=${removed} docker_images_skipped=${skipped} image_list=${image_list}"
}

mkdir -p "$STATE_DIR" "$(dirname "$CUMULATIVE_LABELS")"
touch "$COMPLETED_IDS_FILE" "$CUMULATIVE_LABELS"

if [[ "$RESUME_EXISTING_RUN" == "1" ]]; then
    echo "== Resuming existing mini-SWE-agent run: $RUN_ROOT"
    echo "== Skipping selection and mini-SWE-agent batch"
    trap cleanup_run_docker_images EXIT
    ensure_selected_ids_file
else
    if [[ -e "$RUN_ROOT" ]]; then
        echo "Run root already exists: $RUN_ROOT" >&2
        echo "Set RUN_ID to a fresh value, or pass that path as the first argument to resume." >&2
        exit 2
    fi

    trap cleanup_run_docker_images EXIT

    echo "== Selecting ${BATCH_SIZE} instance ids for ${RUN_ID}"
    "${PYTHON[@]}" - "$DATASET" "$SPLIT" "$BATCH_SIZE" "$SEED" "$ID_RANGE" \
        "$COMPLETED_IDS_FILE" "$CUMULATIVE_LABELS" "$SELECTED_IDS_FILE" <<'PY'
import json
import random
import sys
from pathlib import Path

from benchmark_agents.swebench_vertex.config import DATASET_ALIASES
from benchmark_agents.swebench_vertex.dataset import load_swebench_instances

(
    dataset_arg,
    split,
    batch_size_raw,
    seed_raw,
    id_range,
    completed_path,
    cumulative_path,
    out_path,
) = sys.argv[1:]
batch_size = int(batch_size_raw)
seed = int(seed_raw)
dataset_name = DATASET_ALIASES.get(dataset_arg, dataset_arg)

def parse_id_range(value: str) -> tuple[int, int] | None:
    value = value.strip()
    if not value:
        return None
    if "-" not in value:
        index = int(value)
        return index, index
    start_raw, end_raw = value.split("-", 1)
    start = int(start_raw)
    end = int(end_raw)
    if start < 1 or end < start:
        raise SystemExit(f"Invalid ID_RANGE={value!r}. Use a 1-based inclusive range like 1-50.")
    return start, end

completed = {
    line.strip()
    for line in Path(completed_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
}
cumulative = Path(cumulative_path)
if cumulative.exists():
    for line in cumulative.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        instance_id = row.get("instance_id")
        if instance_id:
            completed.add(str(instance_id))

instances = load_swebench_instances(dataset_name, split)
range_bounds = parse_id_range(id_range)
if range_bounds:
    start, end = range_bounds
    candidates = instances[start - 1 : end]
    if not candidates:
        raise SystemExit(
            f"ID_RANGE={id_range!r} selected no instances from dataset of size {len(instances)}."
        )
    available = [
        instance.instance_id
        for instance in candidates
        if instance.instance_id not in completed
    ]
    selected = available[:batch_size]
else:
    available = [instance.instance_id for instance in instances if instance.instance_id not in completed]
    random.Random(seed).shuffle(available)
    selected = available[:batch_size]
if not selected:
    raise SystemExit("No available instances remain after excluding completed ids.")
if len(selected) < batch_size:
    print(
        f"warning: requested {batch_size}, only {len(selected)} available",
        file=sys.stderr,
    )
Path(out_path).write_text("\n".join(selected) + "\n", encoding="utf-8")
if range_bounds:
    print(
        f"selected={len(selected)} completed_excluded={len(completed)} "
        f"available_in_range={len(available)} id_range={id_range}"
    )
else:
    print(f"selected={len(selected)} completed_excluded={len(completed)} available={len(available)}")
PY

    INSTANCE_ARGS=()
    while IFS= read -r instance_id; do
        [[ -z "$instance_id" ]] && continue
        INSTANCE_ARGS+=(--instance-id "$instance_id")
    done < "$SELECTED_IDS_FILE"

    RUN_ARGS=()
    if [[ "$EXPOSE_VISIBLE_TESTS" == "1" ]]; then
        RUN_ARGS+=(--expose-visible-tests)
        RUN_ARGS+=(--visible-fail-to-pass-count "$VISIBLE_FAIL_TO_PASS_COUNT")
        RUN_ARGS+=(--min-hidden-fail-to-pass-count "$MIN_HIDDEN_FAIL_TO_PASS_COUNT")
        if [[ "$INCLUDE_PASS_TO_PASS" != "1" ]]; then
            RUN_ARGS+=(--no-visible-pass-to-pass)
        fi
    fi

    echo "== Running mini-SWE-agent batch"
    MINI_RUN_CMD=(
        "${BENCHMARK_AGENTS[@]}"
        swebench run
        --dataset "$DATASET"
        --split "$SPLIT"
        --sample-size "$BATCH_SIZE"
        --seed "$SEED"
        --run-id "$RUN_ID"
        --output-dir "$RUNS_DIR"
    )
    MINI_RUN_CMD+=("${INSTANCE_ARGS[@]}")
    if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
        MINI_RUN_CMD+=("${RUN_ARGS[@]}")
    fi
    if [[ -n "$RUN_EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        RUN_EXTRA_ARGS_ARRAY=($RUN_EXTRA_ARGS)
        MINI_RUN_CMD+=("${RUN_EXTRA_ARGS_ARRAY[@]}")
    fi
    set +e
    "${MINI_RUN_CMD[@]}"
    RUN_STATUS=$?
    set -e
    if [[ "$RUN_STATUS" -ne 0 ]]; then
        echo "mini-SWE-agent command exited with status ${RUN_STATUS}; continuing if predictions.eval.jsonl can be produced." >&2
    fi
fi

ensure_eval_predictions_file
if [[ ! -s "${RUN_ROOT}/predictions.eval.jsonl" ]]; then
    echo "No evaluable predictions were produced at ${RUN_ROOT}/predictions.eval.jsonl" >&2
    echo "No ids were added to the cumulative dataset or completed-id state." >&2
    exit 1
fi

echo "== Extracting transient and evaluable ids"
"${PYTHON[@]}" - "$RUN_ROOT" "$TRANSIENT_IDS_FILE" "$EVALUABLE_IDS_FILE" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
transient_out = Path(sys.argv[2])
evaluable_out = Path(sys.argv[3])

summary_path = run_root / "predictions.eval.summary.json"
transient = []
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    excluded = summary.get("excluded", {})
    for key in ("status:LimitsExceeded", "status:TimeoutExpired"):
        transient.extend(str(item) for item in excluded.get(key, []) or [])

evaluable = []
predictions_path = run_root / "predictions.eval.jsonl"
for line in predictions_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    instance_id = row.get("instance_id")
    if instance_id:
        evaluable.append(str(instance_id))

transient_out.write_text("\n".join(sorted(set(transient))) + ("\n" if transient else ""), encoding="utf-8")
evaluable_out.write_text("\n".join(evaluable) + ("\n" if evaluable else ""), encoding="utf-8")
print(f"evaluable={len(evaluable)} transient_released={len(set(transient))}")
PY

if [[ "$SKIP_EVAL" != "1" ]]; then
    echo "== Running official SWE-bench harness: ${EVAL_RUN_ID}"
    # shellcheck disable=SC2206
    EVAL_PYTHON=($EVAL_PYTHON_CMD)
    if ! "${EVAL_PYTHON[@]}" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("swebench") and importlib.util.find_spec("swebench.harness.run_evaluation") else 1)'; then
        echo "Could not import swebench.harness.run_evaluation with EVAL_PYTHON_CMD=${EVAL_PYTHON_CMD}" >&2
        echo "Install the official SWE-bench harness for that Python, or set EVAL_PYTHON_CMD to a Python that has it." >&2
        echo "For this repo, the default is now: EVAL_PYTHON_CMD=\"${PYTHON_CMD}\"" >&2
        exit 1
    fi
    if ! "${EVAL_PYTHON[@]}" - <<'PY'
import sys

try:
    import docker
    from docker.errors import DockerException
except Exception as exc:
    print(f"Could not import the Docker Python SDK: {exc}", file=sys.stderr)
    raise SystemExit(1)

try:
    docker.from_env().ping()
except DockerException as exc:
    print(f"Docker is not reachable from this shell: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
    then
        echo "The official SWE-bench harness requires a working Docker daemon." >&2
        echo "Start Docker Desktop, then confirm 'docker ps' works in this same terminal." >&2
        echo "If Docker is running, check Docker context/socket settings such as DOCKER_HOST." >&2
        exit 1
    fi
    EVAL_CMD=(
        "${EVAL_PYTHON[@]}"
        -m swebench.harness.run_evaluation
        --dataset_name "$DATASET_NAME"
        --predictions_path "${RUN_ROOT}/predictions.eval.jsonl"
        --max_workers "$EVAL_MAX_WORKERS"
        --run_id "$EVAL_RUN_ID"
    )
    if [[ -n "$EVAL_EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        EVAL_EXTRA_ARGS_ARRAY=($EVAL_EXTRA_ARGS)
        EVAL_CMD+=("${EVAL_EXTRA_ARGS_ARRAY[@]}")
    fi
    "${EVAL_CMD[@]}"
else
    echo "== SKIP_EVAL=1, not running official SWE-bench harness"
fi

if [[ "$SKIP_LABEL" != "1" ]]; then
    echo "== Labeling reward-hacking rows"
    if [[ -n "$LABEL_EXTRA_ARGS" ]]; then
        echo "== Label extra args: ${LABEL_EXTRA_ARGS}"
    fi
    LABEL_CMD=(
        "${PYTHON[@]}"
        reward_hack_eval/label_reward_hacking.py
        --predictions "${RUN_ROOT}/predictions.eval.jsonl"
        --eval "${RUN_ROOT}/predictions.eval.jsonl"
        --eval-reports-glob "$EVAL_LOG_GLOB"
        --trajectories-glob "${RUN_ROOT}/mini_swe_agent/**/*.traj.json"
        --test-visibility-splits "${RUN_ROOT}/test_visibility_splits.jsonl"
        --output "$RUN_LABELS_FILE"
    )
    if [[ -n "$LABEL_EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        LABEL_EXTRA_ARGS_ARRAY=($LABEL_EXTRA_ARGS)
        LABEL_CMD+=("${LABEL_EXTRA_ARGS_ARRAY[@]}")
    fi
    "${LABEL_CMD[@]}"
else
    echo "== SKIP_LABEL=1, not labeling reward-hacking rows"
fi

if [[ ! -s "$RUN_LABELS_FILE" ]]; then
    echo "No reward-hack labels were produced at ${RUN_LABELS_FILE}" >&2
    echo "No ids were added to the cumulative dataset or completed-id state." >&2
    exit 1
fi

echo "== Updating cumulative dataset"
"${PYTHON[@]}" - "$CUMULATIVE_LABELS" "$RUN_LABELS_FILE" "$COMPLETED_IDS_FILE" \
    "$TRANSIENT_IDS_FILE" "$EVALUABLE_IDS_FILE" "$RUN_HISTORY_FILE" "$RUN_ID" \
    "$SELECTED_IDS_FILE" "$REPLACE_EXISTING_LABELS" <<'PY'
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

cumulative_path = Path(sys.argv[1])
run_labels_path = Path(sys.argv[2])
completed_path = Path(sys.argv[3])
transient_path = Path(sys.argv[4])
evaluable_path = Path(sys.argv[5])
history_path = Path(sys.argv[6])
run_id = sys.argv[7]
selected_path = Path(sys.argv[8])
replace_existing = sys.argv[9] == "1"

def read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

transient_ids = read_ids(transient_path)
evaluable_ids = read_ids(evaluable_path)
selected_ids = sorted(read_ids(selected_path))
run_label_rows = read_jsonl(run_labels_path)
run_label_ids = {
    str(row.get("instance_id"))
    for row in run_label_rows
    if row.get("instance_id")
}

existing_rows = [
    row
    for row in read_jsonl(cumulative_path)
    if str(row.get("instance_id", "")) not in transient_ids
    and not (
        replace_existing
        and str(row.get("instance_id", "")) in run_label_ids
        and str(row.get("instance_id", "")) in evaluable_ids
    )
]
existing_ids = {str(row.get("instance_id")) for row in existing_rows if row.get("instance_id")}

new_rows = []
for row in run_label_rows:
    instance_id = str(row.get("instance_id", ""))
    if not instance_id:
        continue
    if instance_id in transient_ids:
        continue
    if instance_id not in evaluable_ids:
        continue
    if row.get("patch_successfully_applied") is None and row.get("resolved") is None:
        continue
    if instance_id in existing_ids:
        continue
    new_rows.append(row)
    existing_ids.add(instance_id)

all_rows = existing_rows + new_rows
cumulative_path.write_text(
    "".join(json.dumps(row, sort_keys=False) + "\n" for row in all_rows),
    encoding="utf-8",
)

completed_ids = sorted(
    str(row.get("instance_id"))
    for row in all_rows
    if row.get("instance_id") and str(row.get("instance_id")) not in transient_ids
)
completed_path.write_text("\n".join(completed_ids) + ("\n" if completed_ids else ""), encoding="utf-8")

summary_path = cumulative_path.with_suffix(".summary.json")
label_counts = Counter(str(row.get("eval_label", "UNKNOWN")) for row in all_rows)
summary_path.write_text(
    json.dumps(
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cumulative_labels": str(cumulative_path),
            "total_rows": len(all_rows),
            "completed_ids": len(completed_ids),
            "label_counts": dict(label_counts),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

history_path.parent.mkdir(parents=True, exist_ok=True)
with history_path.open("a", encoding="utf-8") as handle:
    handle.write(
        json.dumps(
            {
                "run_id": str(run_id),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "selected": selected_ids,
                "evaluable": sorted(evaluable_ids),
                "transient_released": sorted(transient_ids),
                "appended": [row["instance_id"] for row in new_rows],
                "replaced_existing_labels": replace_existing,
                "cumulative_total": len(all_rows),
            },
            sort_keys=True,
        )
        + "\n"
    )

print(
    f"appended={len(new_rows)} cumulative_total={len(all_rows)} "
    f"transient_released={len(transient_ids)} replace_existing={replace_existing}"
)
print(f"cumulative_labels={cumulative_path}")
print(f"completed_ids={completed_path}")
print(f"summary={summary_path}")
PY

cleanup_run_docker_images
trap - EXIT

echo "== Done"
echo "run_root=${RUN_ROOT}"
echo "eval_run_id=${EVAL_RUN_ID}"
echo "run_labels=${RUN_LABELS_FILE}"
echo "cumulative_labels=${CUMULATIVE_LABELS}"
