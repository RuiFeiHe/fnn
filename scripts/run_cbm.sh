#!/bin/bash
set -euo pipefail

export TMPDIR=/project/rf/tmp
mkdir -p "$TMPDIR"

FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---- edit below ----
CONFIG=data/train_digital_twin/config_cbm_mouseA.yaml
GPUS=4,5,6,7
NPROC=4
# ---- end edit ----

SAVE_BASE=$(python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
print(c['save-state']['directory'])
")
SAVE_DIR="${SAVE_BASE}_${TIMESTAMP}"

mkdir -p "$SAVE_DIR"
cp "$0"      "$SAVE_DIR/run_cbm.sh"
cp "$CONFIG" "$SAVE_DIR/config_snapshot.yaml"

echo "[run] Saving to  : $SAVE_DIR"
echo "[run] Config     : $CONFIG"
echo "[run] GPUs       : $GPUS  (nproc=$NPROC)"
echo "[run] Timestamp  : $TIMESTAMP"
echo

CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NPROC \
    --master_port=$FREE_PORT \
    scripts/train_cbm.py $CONFIG \
    --timestamp $TIMESTAMP
