#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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
if [ -z "$DATA_DIR" ]; then
    echo "Error: set DATA_DIR=/abs/path/to/cancer_data before running run_train.sh" >&2
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: data directory not found: $DATA_DIR" >&2
    exit 1
fi

DATA_DIR="$(cd "$DATA_DIR" && pwd)"
DATA_ROOT="${DATA_ROOT:-$(dirname "$DATA_DIR")}"
CANCER="${CANCER:-$(basename "$DATA_DIR")}"

ARGS=(
    --python "$PYTHON"
    --data-root "$DATA_ROOT"
    --cancers "$CANCER"
    --groups benchmark
)
if [ "${DRY_RUN:-0}" -ne 1 ]; then
    ARGS+=(--execute)
fi

exec "$PYTHON" "$SCRIPT_DIR/run_experiments.py" "${ARGS[@]}"
