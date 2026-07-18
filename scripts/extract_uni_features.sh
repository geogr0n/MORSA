#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${DATA_DIR:=${MORSA_DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
    echo "Error: set DATA_DIR=/abs/path/to/cancer_data before running extract_uni_features.sh" >&2
    exit 1
fi
MORSA_DATA_DIR="${MORSA_DATA_DIR:-$DATA_DIR}"
export MORSA_DATA_DIR
PYTHON="${PYTHON:-python3}"
UNI_MODEL_PATH="${UNI_MODEL_PATH:-$PROJECT_DIR/pytorch_model.bin}"
export UNI_MODEL_PATH

"$PYTHON" "$PROJECT_DIR/preprocessing/extract_uni_features.py" \
        --ref_file "$DATA_DIR/ref_file.csv" \
        --patch_data_path "$DATA_DIR/Patches_hdf5" \
        --feature_path "$DATA_DIR/features" \
        --max_patch_number "${MAX_PATCH_NUMBER:-6400}" \
        --uni_model_path "$UNI_MODEL_PATH"
