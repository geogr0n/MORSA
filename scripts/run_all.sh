#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[run_all] failed at line $LINENO" >&2' ERR

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

cancer_type=(${CANCER_TYPES:-BRCA LUAD LUSC COAD KIRC KIRP GBM HNSC LIHC PAAD PRAD SKCM STAD THCA UCEC BLCA})
if [ -z "${EVENT_TYPES:-}" ]; then
    echo "Set EVENT_TYPES to one or more supported events before running run_all.sh" >&2
    exit 1
fi
event_type=($EVENT_TYPES)

# This orchestrator keeps portable defaults; override with environment variables.
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/data}"
CROSS_COHORT_ROOT="${CROSS_COHORT_ROOT:-$DATA_ROOT/cross-cohort}"

CROSS_FOLDS="${CROSS_FOLDS:-5}"
CROSS_DEVICE="${CROSS_DEVICE:-auto}"
CROSS_BATCH_SIZE="${CROSS_BATCH_SIZE:-16}"
CROSS_EXPERIMENTS="${CROSS_EXPERIMENTS:-}"

echo "=== MORSA Multi-Cancer Runner ==="
echo "PYTHON: $PYTHON"
echo "DATA_ROOT: $DATA_ROOT"
echo "CROSS_COHORT_ROOT: $CROSS_COHORT_ROOT"
echo "cancer_type: ${cancer_type[*]}"
echo "event_type: ${event_type[*]}"
echo ""

is_supported_event() {
    local event="$1"
    case "$event" in
        run_train|run_evaluate|run_experiments|run_analysis|run_cross_cohort_eval)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

run_event_for_cancer() {
    local event="$1"
    local cancer="$2"
    local data_dir="$DATA_ROOT/$cancer"
    local model_root="$data_dir/output/TCGA"

    if [ ! -d "$data_dir" ]; then
        echo "Data directory missing for $cancer: $data_dir" >&2
        return 1
    fi

    case "$event" in
        run_train)
            echo "[run] $event cancer=$cancer"
            DATA_DIR="$data_dir" COHORT="TCGA" bash "$SCRIPT_DIR/run_train.sh"
            ;;
        run_evaluate)
            echo "[run] $event cancer=$cancer"
            DATA_DIR="$data_dir" MODEL_DIR="$model_root" bash "$SCRIPT_DIR/run_evaluate.sh"
            ;;
        run_experiments)
            echo "[run] $event cancer=$cancer"
            MORSA_DATA_ROOT="$DATA_ROOT" bash "$SCRIPT_DIR/run_experiments.sh" --execute --cancers "$cancer"
            ;;
        run_analysis)
            echo "[run] $event cancer=$cancer"
            DATA_ROOT="$DATA_ROOT" \
            MODEL_ROOT="$model_root" \
            RESULTS_ROOT="$model_root/analysis" \
            CANCER="$cancer" \
            bash "$SCRIPT_DIR/run_analysis.sh"
            ;;
        run_cross_cohort_eval)
            echo "[run] $event cancer=$cancer"
            DATA_ROOT="$DATA_ROOT" \
            CROSS_COHORT_ROOT="$CROSS_COHORT_ROOT" \
            CANCER="$cancer" \
            TCGA_MODEL_DIR="$model_root" \
            CROSS_FOLDS="$CROSS_FOLDS" \
            CROSS_DEVICE="$CROSS_DEVICE" \
            CROSS_BATCH_SIZE="$CROSS_BATCH_SIZE" \
            CROSS_EXPERIMENTS="$CROSS_EXPERIMENTS" \
            bash "$SCRIPT_DIR/run_cross_cohort_eval.sh"
            ;;
    esac
}

for event in "${event_type[@]}"; do
    if ! is_supported_event "$event"; then
        echo "Unsupported event: $event" >&2
        echo "Supported events: run_train run_evaluate run_experiments run_analysis run_cross_cohort_eval" >&2
        exit 1
    fi
done

for cancer in "${cancer_type[@]}"; do
    for event in "${event_type[@]}"; do
        run_event_for_cancer "$event" "$cancer"
    done
done

echo "Multi-cancer execution completed."
