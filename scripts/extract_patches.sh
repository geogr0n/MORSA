#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${DATA_DIR:=${MORSA_DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
    echo "Error: set DATA_DIR=/abs/path/to/cancer_data before running extract_patches.sh" >&2
    exit 1
fi
MORSA_DATA_DIR="${MORSA_DATA_DIR:-$DATA_DIR}"
export MORSA_DATA_DIR
PYTHON="${PYTHON:-python3}"

"$PYTHON" "$PROJECT_DIR/preprocessing/extract_patches.py" \
        --ref_file "$DATA_DIR/ref_file.csv" \
        --wsi_path "$DATA_DIR/HE" \
        --patch_path "$DATA_DIR/Patches_hdf5" \
        --mask_path "$DATA_DIR/Patches_hdf5" \
        --patch_size "${PATCH_SIZE:-256}" \
        --max_patches_per_slide "${MAX_PATCHES_PER_SLIDE:-6400}" \
        --seed "${PATCH_SEED:-5}" \
        --parallel "${PATCH_WORKERS:-20}"
