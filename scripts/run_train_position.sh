#!/bin/bash
set -euo pipefail

export TMPDIR=/project/rf/tmp
mkdir -p "$TMPDIR"

# ---- edit below ----
CONFIG=data/train_digital_twin/config_position_embedding_mouseA.yaml
GPU=6
# ---- end edit ----

# Derive save path from config
SAVE_PATH=$(python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
print(c['save-state']['path'])
")
SAVE_DIR=$(dirname "$SAVE_PATH")

mkdir -p "$SAVE_DIR"
cp "$0"      "$SAVE_DIR/run_train_position.sh"
cp "$CONFIG" "$SAVE_DIR/config_snapshot.yaml"

echo "[run] Config  : $CONFIG"
echo "[run] GPU     : $GPU"
echo "[run] Save to : $SAVE_PATH"
echo

CUDA_VISIBLE_DEVICES=$GPU python scripts/learn_position_embedding.py "$CONFIG"
