#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif [ -f "$PROJECT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON="$PROJECT_DIR/.venv/Scripts/python.exe"
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
if [ -z "$DATA_DIR" ]; then
    echo "Error: set DATA_DIR=/abs/path/to/cancer_data before running run_closed_form.sh" >&2
    exit 1
fi

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
COHORT="${COHORT:-TCGA}"
CANCER="${CANCER:-$(basename "$DATA_DIR")}"

"$PYTHON" "$PROJECT_DIR/src/run_morsa_mean_closed.py" \
    --ref_file "$DATA_DIR/ref_file.csv" \
    --feature_path "$DATA_DIR/features" \
    --save_dir "$DATA_DIR/output" \
    --cohort "$COHORT" \
    --cancer "$CANCER"

"$PYTHON" "$PROJECT_DIR/evaluation/evaluate_model.py" \
    --experiment morsa_mean_closed \
    --model_dir "$DATA_DIR/output/$COHORT"
