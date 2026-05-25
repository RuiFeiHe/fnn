#!/usr/bin/env bash
set -euo pipefail

ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-""}

python scripts/generate_concepts_from_stimuli.py \
    --stim-dir  /project/rf/data/sensorium2023_fnn/mouseA/training/stimuli \
    --seed      data/concepts/concepts_visual.yaml \
    --out-yaml  data/concepts/concepts_visual_v2.yaml \
    --out-npy   data/concepts/concepts_visual_v2.npy \
    --n-trials  60 \
    --frames-per-trial 5 \
    --frames-per-call  6 \
    --sim-thresh 0.85 \
    --clip-model      ViT-bigG-14 \
    --clip-pretrained laion2b_s39b_b160k \
    --gpu 0
