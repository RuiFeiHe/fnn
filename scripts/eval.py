#!/usr/bin/env python
"""
Evaluate a trained digital twin by computing CC_abs, CC_max, and CC_norm.

Usage:
    python scripts/eval.py                          # uses default config, final ckpt
    python scripts/eval.py --config path/to/config.yaml
    CUDA_VISIBLE_DEVICES=1 python scripts/eval.py --ckpt /project/rf/data/train_digital_twin/results_ex8/best_state_dict.pth

    # eval the best validation checkpoint                                                                                                                                                                                                                                                                                             
    python scripts/eval.py --ckpt /project/rf/data/train_digital_twin/results_ep100/best_state_dict.pth                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                    
    # custom config
    python scripts/eval.py --config path/to/config.yaml --ckpt path/to/ckpt.pth   

      # evaluate a student checkpoint
  python scripts/eval.py --student --ckpt /project/rf/data/train_digital_twin/results_finetune_student_ex15.ft2/state_dict.pth --config data/train_digital_twin/config_finetune_student.yaml

  # evaluate a full network checkpoint (unchanged behaviour)
  python scripts/eval.py --ckpt /project/rf/data/train_digital_twin/results_ex8/best_state_dict.pth

"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm
from fnn.microns import load_network_from_params
from fnn.microns.build import network_s
from fnn.data import load_evaluation_data
from fnn import evaluate
from fnn.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG = Path('/project/rf/code/fnn/data/train_digital_twin/config.yaml')


def main(args):
    # READ CONFIG
    logger.info(f"Config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # RESOLVE CHECKPOINT PATH
    ckpt_path = Path(args.ckpt) if args.ckpt else (
        Path(config['save-state']['directory']) / config['save-state']['state_dict']
    )
    ckpt_epochs_file = ckpt_path.parent / 'ckpt_epochs.pt'
    if ckpt_epochs_file.exists():
        ckpt_epochs = torch.load(ckpt_epochs_file, map_location='cpu')
        epoch = ckpt_epochs.get(ckpt_path.name)
        epoch_str = f"epoch {epoch}" if epoch is not None else "epoch unknown"
    else:
        epoch_str = "epoch unknown"
    logger.info(f"Checkpoint: {ckpt_path} ({epoch_str})")

    # RESOLVE OUTPUT DIRECTORY
    out_dir = Path(args.out) if args.out else Path(config['save-state']['directory'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # LOAD MODEL
    logger.info("Loading model.")
    student = args.student
    if student:
        params = torch.load(ckpt_path, map_location='cpu')
        n_units = params['readout.feature.weights.0'].shape[0]
        model = network_s(units=n_units)
        model.load_state_dict(params)
        if torch.cuda.is_available():
            model = model.cuda()
        logger.info(f"Loaded as network_s ({n_units} units)")
    else:
        model = load_network_from_params(ckpt_path)
    model.eval()

    # LOAD EVALUATION DATA
    eval_data_dir = Path(config['data-source']['evaluation']['directory'])
    logger.info(f"Loading evaluation data from {eval_data_dir}")
    evaluation_data = load_evaluation_data(eval_data_dir)

    burnin_frames = config['objective']['burnin_frames']

    # CC_MAX — derived purely from the recorded neural responses (no model needed)
    logger.info("Computing CC_max.")
    units_fmt = evaluate.format_responses(
        evaluation_data['units'], burnin_frames=burnin_frames
    )
    cc_max = evaluate.compute_cc_max(units_fmt)

    # MODEL PREDICTIONS
    logger.info("Generating model predictions.")
    s = evaluation_data['stimuli']
    p = evaluation_data['perspectives']
    m = evaluation_data['modulations']

    units_pred = []
    with torch.no_grad():
        for i in tqdm(range(len(s)), desc="Stimuli"):
            repeats_pred = []
            for j in range(len(s[i])):
                repeats_pred.append(
                    model.predict(
                        stimuli=s[i][j],
                        perspectives=p[i][j],
                        modulations=m[i][j],
                    )
                )
            units_pred.append(repeats_pred)

    # CC_ABS — correlation between mean model prediction and mean neural response
    logger.info("Computing CC_abs.")
    units_pred_fmt = evaluate.format_responses(units_pred, burnin_frames=burnin_frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress ConstantInputWarning from untrained/constant outputs
        cc_abs = evaluate.compute_cc_abs(units_pred_fmt, units_fmt)

    # CC_NORM
    cc_norm = cc_abs / cc_max

    def _stats(arr, label):
        valid = arr[np.isfinite(arr)]
        n_valid, n_total = len(valid), len(arr)
        if n_valid == 0:
            logger.info(f"{label} — no valid units (all NaN/Inf); model may be untrained or predictions are constant")
        else:
            logger.info(f"{label} — median: {np.median(valid):.4f}  mean: {np.mean(valid):.4f}  "
                        f"valid units: {n_valid}/{n_total}")

    _stats(cc_max,  "CC_max ")
    _stats(cc_abs,  "CC_abs ")
    _stats(cc_norm, "CC_norm")

    # SAVE RESULTS
    out_stem = ckpt_path.stem  # e.g. "state_dict" or "best_state_dict"
    out_path = out_dir / f"eval_{out_stem}.npz"
    np.savez(out_path, cc_max=cc_max, cc_abs=cc_abs, cc_norm=cc_norm)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained digital twin (CC_abs, CC_max, CC_norm)."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to checkpoint .pth file (default: state_dict from config)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Directory to save eval results (default: save-state directory from config)",
    )
    parser.add_argument(
        "--student", action="store_true", default=False,
        help="Load checkpoint as network_s (student) instead of full network",
    )
    args = parser.parse_args()
    main(args)
