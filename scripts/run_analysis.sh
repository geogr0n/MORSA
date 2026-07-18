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

: "${DATA_ROOT:=}"
if [ -z "$DATA_ROOT" ]; then
    echo "Error: set DATA_ROOT=/abs/path/to/data before running run_analysis.sh" >&2
    exit 1
fi

: "${MODEL_ROOT:=}"
if [ -z "$MODEL_ROOT" ]; then
    echo "Error: set MODEL_ROOT=/abs/path/to/cancer/output/TCGA before running run_analysis.sh" >&2
    exit 1
fi

: "${CANCER:=}"
if [ -z "$CANCER" ]; then
    echo "Error: set CANCER=<cancer_code> before running run_analysis.sh" >&2
    exit 1
fi

RESULTS_ROOT="${RESULTS_ROOT:-$MODEL_ROOT/analysis}"
TASKS="${TASKS:-rna_structure,basis_stability,coordinate_recovery,morphology_structure,component_attribution,learnedrank_alignment,efficiency}"
SEED="${SEED:-29}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== MORSA Analysis ==="
echo "PYTHON: $PYTHON"
echo "DATA_ROOT: $DATA_ROOT"
echo "MODEL_ROOT: $MODEL_ROOT"
echo "RESULTS_ROOT: $RESULTS_ROOT"
echo "TASKS: $TASKS"
echo ""

cd "$PROJECT_DIR"

"$PYTHON" -m analysis.structural_analysis \
    --data_root "$DATA_ROOT" \
    --model_root "$MODEL_ROOT" \
    --results_root "$RESULTS_ROOT" \
    --cancer "$CANCER" \
    --tasks "$TASKS" \
    --seed "$SEED"

echo "Analysis completed."
