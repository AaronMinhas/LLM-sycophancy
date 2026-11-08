#!/bin/bash

# Runs the Figure 2 experiments overnight on the Mac (MPS). Run from the repo root.
# Only has the runs that worked with the unchanged run_syco.py.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
BEHAVIOR_DIR="$REPO_ROOT/experiments/behavioral_analysis"
PYTHON="$REPO_ROOT/.venv/bin/python"
RUNNER="$BEHAVIOR_DIR/run_syco.py"
PREFIX_DATASET="$REPO_ROOT/lib/opinion_only/prefix/mmlu_opinion_only.pkl"

if [[ "${SYCO_QUEUE_CAFFEINATED:-}" != "1" ]]; then
    exec env SYCO_QUEUE_CAFFEINATED=1 caffeinate -dims "$0" "$@"
fi

QUEUE_STARTED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
QUEUE_STARTED_EPOCH="$(date '+%s')"
QUEUE_ID="$(date '+%Y%m%d_%H%M%S')"
LOG_ROOT="$BEHAVIOR_DIR/queue_logs/$QUEUE_ID"
QUEUE_LOG="$LOG_ROOT/queue.log"
FAILURE_COUNT=0

mkdir -p "$LOG_ROOT"

log_queue() {
    printf '%s\n' "$*" | tee -a "$QUEUE_LOG"
}

format_elapsed() {
    local total="$1"
    local hours=$((total / 3600))
    local minutes=$(((total % 3600) / 60))
    local seconds=$((total % 60))
    printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
}

run_experiment() {
    local slug="$1"
    local model="$2"
    local condition="$3"
    local question_type="$4"
    local dataset_file="$5"
    local output_dir="$6"
    local output_pattern="$7"
    local stdout_log="$LOG_ROOT/${slug}.stdout.log"
    local stderr_log="$LOG_ROOT/${slug}.stderr.log"
    local marker="$LOG_ROOT/.${slug}.start"
    local started_at
    local ended_at
    local started_epoch
    local ended_epoch
    local elapsed
    local process_status
    local effective_status
    local validation_status
    local output_file
    local validation

    started_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"
    started_epoch="$(date '+%s')"
    touch "$marker"

    log_queue "START model=$model condition=$condition time=$started_at"

    (
        cd "$BEHAVIOR_DIR" || exit 1
        # Use the Mistral copy I already downloaded so a Hub update can't change the run.
        HF_HUB_OFFLINE=1 "$PYTHON" "$RUNNER" \
            --model_name "$model" \
            --dataset mmlu \
            --question_type "$question_type" \
            --input_filename "$dataset_file" \
            --full_question_column full_question
    ) >"$stdout_log" 2>"$stderr_log"
    process_status=$?

    output_file="$(find "$output_dir" -maxdepth 1 -type f -name "$output_pattern" -newer "$marker" -print 2>/dev/null | sort | tail -n 1)"
    effective_status="$process_status"
    validation="not_run"

    if [[ "$process_status" -eq 0 && -n "$output_file" ]]; then
        validation="$("$PYTHON" - "$output_file" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
try:
    frame = pd.read_pickle(path)
    if len(frame) != 14042:
        print(f"failed rows={len(frame)} expected=14042")
        raise SystemExit(1)
    if "model_answer" not in frame.columns:
        print("failed missing=model_answer")
        raise SystemExit(1)
    valid = frame["model_answer"].map(
        lambda value: isinstance(value, str)
        and len(value) == 1
        and value.isupper()
        and value.isalpha()
    ).sum()
    if valid == 0:
        print(f"failed rows={len(frame)} valid_answers=0")
        raise SystemExit(1)
    print(f"passed rows={len(frame)} valid_answers={int(valid)} invalid_answers={len(frame)-int(valid)}")
except Exception as exc:
    print(f"failed error={type(exc).__name__}:{exc}")
    raise SystemExit(1)
PY
)"
        validation_status=$?
        if [[ "$validation_status" -ne 0 ]]; then
            effective_status="$validation_status"
        fi
    elif [[ "$process_status" -eq 0 ]]; then
        effective_status=1
        validation="failed no_new_output_detected"
    fi

    ended_at="$(date '+%Y-%m-%dT%H:%M:%S%z')"
    ended_epoch="$(date '+%s')"
    elapsed="$(format_elapsed $((ended_epoch - started_epoch)))"

    log_queue "END model=$model condition=$condition time=$ended_at elapsed=$elapsed process_exit=$process_status effective_exit=$effective_status output=${output_file:-NONE} validation=$validation stdout=$stdout_log stderr=$stderr_log"

    if [[ "$effective_status" -ne 0 ]]; then
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi

    return 0
}

log_queue "QUEUE_START time=$QUEUE_STARTED_AT repo=$REPO_ROOT"

# Mistral Plain is done, so this only runs Mistral Prefix.
run_experiment \
    "mistral_7b_instruct_v0_3_prefix" \
    "mistralai/Mistral-7B-Instruct-v0.3" \
    "prefix_opinion_only" \
    "opinion_only" \
    "$PREFIX_DATASET" \
    "$BEHAVIOR_DIR/output/mmlu/opinion_only" \
    "Mistral-7B-Instruct-v0_3_*.pkl"

QUEUE_ENDED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
QUEUE_ENDED_EPOCH="$(date '+%s')"
QUEUE_ELAPSED="$(format_elapsed $((QUEUE_ENDED_EPOCH - QUEUE_STARTED_EPOCH)))"
log_queue "QUEUE_END time=$QUEUE_ENDED_AT elapsed=$QUEUE_ELAPSED failures=$FAILURE_COUNT"

if [[ "$FAILURE_COUNT" -ne 0 ]]; then
    exit 1
fi
