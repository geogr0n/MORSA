#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${DATA_DIR:=${MORSA_DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
    echo "Error: set DATA_DIR=/abs/path/to/cancer_data before running cluster_patch_features.sh" >&2
    exit 1
fi
MORSA_DATA_DIR="${MORSA_DATA_DIR:-$DATA_DIR}"
export MORSA_DATA_DIR
PYTHON="${PYTHON:-python3}"

"$PYTHON" "$PROJECT_DIR/preprocessing/cluster_patch_features.py" \
        --ref_file "$DATA_DIR/ref_file.csv" \
        --feature_path "$DATA_DIR/features" \
        --num_clusters "${NUM_CLUSTERS:-100}" \
        --seed "${KMEANS_SEED:-0}"
