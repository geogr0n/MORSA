#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

: "${DATA_ROOT:=}"
if [ -z "$DATA_ROOT" ]; then
    echo "Error: set DATA_ROOT=/abs/path/to/data before running run_downstream.sh" >&2
    exit 1
fi

: "${RESOURCE_ROOT:=}"
if [ -z "$RESOURCE_ROOT" ]; then
    echo "Error: set RESOURCE_ROOT=/abs/path/to/downstream_data before running run_downstream.sh" >&2
    exit 1
fi

RESULTS_ROOT="${RESULTS_ROOT:-$RESOURCE_ROOT/results/digital_molecular_reporting}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== MORSA Digital Molecular Reporting Downstream ==="
echo "DATA_ROOT: $DATA_ROOT"
echo "RESOURCE_ROOT: $RESOURCE_ROOT"
echo "RESULTS_ROOT: $RESULTS_ROOT"
echo ""
echo "Assuming prepared downstream resources already exist under: $RESOURCE_ROOT/prepared"
echo ""

cd "$PROJECT_DIR"

"$PYTHON" -m downstream.program_report \
    --data_root "$DATA_ROOT" \
    --resource_root "$RESOURCE_ROOT" \
    --results_root "$RESULTS_ROOT"

"$PYTHON" -m downstream.msi_triage \
    --data_root "$DATA_ROOT" \
    --resource_root "$RESOURCE_ROOT" \
    --results_root "$RESULTS_ROOT"

"$PYTHON" -m downstream.subtype_report \
    --data_root "$DATA_ROOT" \
    --resource_root "$RESOURCE_ROOT" \
    --results_root "$RESULTS_ROOT"

"$PYTHON" -m downstream.reporting_summary --results_root "$RESULTS_ROOT"

echo "All downstream reporting tasks completed."
