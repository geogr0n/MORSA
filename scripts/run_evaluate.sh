#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "$SCRIPT_DIR/../.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/../.venv/bin/python"
elif [ -f "$SCRIPT_DIR/../.venv/Scripts/python.exe" ]; then
    PYTHON="$SCRIPT_DIR/../.venv/Scripts/python.exe"
elif [ -f "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
elif [ -f "$ROOT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

: "${DATA_DIR:=${MORSA_DATA_DIR:-}}"
if [ -z "${MODEL_DIR:-}" ] && [ -z "$DATA_DIR" ]; then
    echo "Error: set DATA_DIR=/abs/path/to/cancer_data or MODEL_DIR=/abs/path/to/output/TCGA" >&2
    exit 1
fi

if [ -n "${MODEL_DIR:-}" ]; then
    MODEL_DIR="$MODEL_DIR"
    OUTPUT_DIR="${OUTPUT_DIR:-$(cd "$MODEL_DIR/.." && pwd)}"
else
    MORSA_DATA_DIR="${MORSA_DATA_DIR:-$DATA_DIR}"
    export MORSA_DATA_DIR
    OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/output}"
    MODEL_DIR="$OUTPUT_DIR/TCGA"
fi
EVAL_SCRIPT="$SCRIPT_DIR/../evaluation/evaluate_model.py"
RESULTS_DIR="$MODEL_DIR/results"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

echo "=== Batch Evaluate Experiments ==="
echo "DATA_DIR: $DATA_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "MODEL_DIR: $MODEL_DIR"
echo "MAX_PARALLEL: $MAX_PARALLEL"
echo ""

EXPERIMENTS=()
for dir in "$MODEL_DIR"/*/; do
    if [ -f "$dir/test_results.pkl" ]; then
        exp_name=$(basename "$dir")
        EXPERIMENTS+=("$exp_name")
    fi
done

echo "Found ${#EXPERIMENTS[@]} experiments: ${EXPERIMENTS[*]}"
echo ""

PIDS=()
for exp in "${EXPERIMENTS[@]}"; do
    echo "Starting: $exp"
    "$PYTHON" "$EVAL_SCRIPT" --experiment "$exp" --model_dir "$MODEL_DIR" &
    PIDS+=($!)
    if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo ""
echo "=== Evaluations completed ==="
echo ""

SUMMARY_FILE="$RESULTS_DIR/summary_metrics.csv"
HEADER_WRITTEN=false

for exp in "${EXPERIMENTS[@]}"; do
    ROW_FILE="$RESULTS_DIR/$exp/summary_row.csv"
    if [ -f "$ROW_FILE" ]; then
        if [ "$HEADER_WRITTEN" = false ]; then
            cat "$ROW_FILE" > "$SUMMARY_FILE"
            HEADER_WRITTEN=true
        else
            tail -n +2 "$ROW_FILE" >> "$SUMMARY_FILE"
        fi
    fi
done

echo "=== Summary Metrics ==="
if [ -f "$SUMMARY_FILE" ]; then
    if command -v column >/dev/null 2>&1; then
        column -t -s',' "$SUMMARY_FILE"
    else
        cat "$SUMMARY_FILE"
    fi
    echo ""
    echo "Saved to: $SUMMARY_FILE"
else
    echo "No summary files found."
fi
