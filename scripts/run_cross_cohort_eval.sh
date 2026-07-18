#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[run_cross_cohort_eval] failed at line $LINENO" >&2' ERR

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
    echo "Error: set DATA_ROOT=/abs/path/to/data before running run_cross_cohort_eval.sh" >&2
    exit 1
fi

: "${CANCER:=${1:-}}"
: "${CANCER:=}"
if [ -z "$CANCER" ]; then
    echo "Error: set CANCER=<TCGA cancer code> before running run_cross_cohort_eval.sh" >&2
    exit 1
fi
CANCER="$(echo "$CANCER" | tr '[:lower:]' '[:upper:]')"

CROSS_COHORT_ROOT="${CROSS_COHORT_ROOT:-$DATA_ROOT/cross-cohort}"
TCGA_MODEL_DIR="${TCGA_MODEL_DIR:-$DATA_ROOT/$CANCER/output/TCGA}"
CROSS_FOLDS="${CROSS_FOLDS:-5}"
CROSS_DEVICE="${CROSS_DEVICE:-auto}"
CROSS_BATCH_SIZE="${CROSS_BATCH_SIZE:-16}"
CROSS_EXPERIMENTS="${CROSS_EXPERIMENTS:-mean,he2rna,vis,morsa_enc,morsa_mean,morsa_he2rna,morsa_vis,morsa}"

# Mappings from primary TCGA cancer code to external cohort.
# References:
# - CDDP_EAGLE-1: GDC/NCI documents it as lung adenocarcinoma.
# - CGCI-HTMCP-LC: GDC/CGCI documents it as lung cancer (NSCLC).
EAGLE_TCGA_CANCER="${EAGLE_TCGA_CANCER:-LUAD}"
HTMCP_TCGA_CANCER="${HTMCP_TCGA_CANCER:-LUSC}"

if [ ! -d "$TCGA_MODEL_DIR" ]; then
    echo "[skip] missing TCGA model dir for $CANCER: $TCGA_MODEL_DIR"
    exit 0
fi

declare -a TARGET_COHORTS=()

if [ -d "$CROSS_COHORT_ROOT/CPTAC-$CANCER" ]; then
    TARGET_COHORTS+=("CPTAC-$CANCER")
fi
if [ "$CANCER" = "$EAGLE_TCGA_CANCER" ] && [ -d "$CROSS_COHORT_ROOT/CCDP=EAGLE-1" ]; then
    TARGET_COHORTS+=("CCDP=EAGLE-1")
fi
if [ "$CANCER" = "$HTMCP_TCGA_CANCER" ] && [ -d "$CROSS_COHORT_ROOT/HTMCP-LC" ]; then
    TARGET_COHORTS+=("HTMCP-LC")
fi

if [ ${#TARGET_COHORTS[@]} -eq 0 ]; then
    echo "[skip] no mapped cross-cohort dataset for CANCER=$CANCER under $CROSS_COHORT_ROOT"
    exit 0
fi

echo "=== Cross-Cohort Evaluation ==="
echo "PYTHON: $PYTHON"
echo "CANCER: $CANCER"
echo "DATA_ROOT: $DATA_ROOT"
echo "CROSS_COHORT_ROOT: $CROSS_COHORT_ROOT"
echo "TCGA_MODEL_DIR: $TCGA_MODEL_DIR"
echo "TARGET_COHORTS: ${TARGET_COHORTS[*]}"
echo "CROSS_EXPERIMENTS: $CROSS_EXPERIMENTS"
echo ""

for cohort in "${TARGET_COHORTS[@]}"; do
    cohort_dir="$CROSS_COHORT_ROOT/$cohort"
    external_ref="$cohort_dir/ref_file.csv"
    external_feat="$cohort_dir/features"
    output_dir="$cohort_dir/output/${CANCER}_TCGA_models"

    if [ ! -f "$external_ref" ] || [ ! -d "$external_feat" ]; then
        echo "[skip] $cohort is incomplete (missing ref_file/features)"
        continue
    fi

    cmd=(
        "$PYTHON" "$PROJECT_DIR/evaluation/evaluate_cross_cohort.py"
        --tcga_model_dir "$TCGA_MODEL_DIR"
        --external_ref_file "$external_ref"
        --external_feature_path "$external_feat"
        --output_dir "$output_dir"
        --external_cohort "$cohort"
        --folds "$CROSS_FOLDS"
        --device "$CROSS_DEVICE"
        --batch_size "$CROSS_BATCH_SIZE"
    )
    if [ -n "$CROSS_EXPERIMENTS" ]; then
        cmd+=(--experiments "$CROSS_EXPERIMENTS")
    fi

    echo "[run] cancer=$CANCER external_cohort=$cohort"
    "${cmd[@]}"
done

echo "Cross-cohort evaluation completed for CANCER=$CANCER."
