#!/usr/bin/env python
"""
Compute SVD of core feature representations over all training samples.

Following KDEP (https://github.com/CVMI-Lab/KDEP/blob/main/src/gen_svd_weights.py):
  - Extract spatially-pooled core output for every training frame
  - Mean-center the feature matrix
  - Compute SVD: features = U @ diag(S) @ Vh
  - Save Vh (right singular vectors), feat_mean, and singular values S

Usage:
    python scripts/get_svd_weights.py [config] [--checkpoint PATH]
    CUDA_VISIBLE_DEVICES=3 python scripts/get_svd_weights.py /project/rf/code/fnn/data/train_digital_twin/config.yaml  
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from fnn.data import load_training_data
from fnn.microns import load_network_from_params
from fnn import microns
from fnn.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG = Path('/project/rf/code/fnn/data/train_digital_twin/config.yaml')


def collect_core_features(model, dataset):
    """
    Pass every training frame through the model and collect spatially-pooled
    core outputs. Returns a [N_frames_total, C] float32 numpy array.

    Recurrent state is reset at the start of each trial so that features
    reflect the same temporal context used during training.
    """
    model.eval()

    captured = {}

    def hook_fn(module, input, output):
        # output: [N, C, H', W'] — average-pool spatial dims to [N, C]
        captured['feat'] = output.mean(dim=(-2, -1)).detach().cpu().float()

    hook = model.module('core').register_forward_hook(hook_fn)
    all_features = []

    train_keys = dataset.keys(training=True)
    logger.info(f"Processing {len(train_keys)} training trials")

    try:
        for key in tqdm(train_keys, desc="Trials"):
            trial = dataset.load(key)
            stimuli     = trial['stimuli']      # [T, H, W, C] uint8 or [T, H, W]
            perspectives = trial['perspectives'] # [T, P]
            modulations  = trial['modulations']  # [T, M]
            T = stimuli.shape[0]

            model.reset()
            with torch.no_grad():
                for t in range(T):
                    s, p, m, _ = model.to_tensor(
                        stimuli[t],
                        perspectives[t],
                        modulations[t],
                    )
                    model._raw(s, p, m)
                    all_features.append(captured['feat'].numpy())  # [1, C]
    finally:
        hook.remove()

    return np.concatenate(all_features, axis=0)  # [N_frames_total, C]


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    data_dir = config['data-source']['training']['directory']
    max_items = config['data-source']['training'].get('max_items', None)
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_training_data(data_dir, max_items)
    n_train = dataset.df.training.sum()
    s_train = dataset.df.loc[dataset.df.training, 'samples'].sum()
    logger.info(f"Training trials: {n_train} ({s_train} frames)")

    # Load model
    if args.checkpoint:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model = load_network_from_params(args.checkpoint)
    else:
        logger.info("Loading foundation model")
        model, _ = microns.scan(**config['data-source']['foundation-core'])
    model = model.to(device)

    # Collect spatially-pooled core features for every training frame
    logger.info("Extracting core features over all training frames...")
    features = collect_core_features(model, dataset)
    logger.info(f"Feature matrix shape: {features.shape}")  # [N_frames_total, C]

    # Center (subtract per-feature mean, as in KDEP)
    feat_mean = features.mean(axis=0)        # [C]
    features -= feat_mean                    # in-place to save memory

    # SVD via torch.linalg for numerical stability
    logger.info("Computing SVD...")
    _, S, Vh = torch.linalg.svd(
        torch.from_numpy(features).float(),
        full_matrices=False,
    )
    # S:  [K]    singular values, descending
    # Vh: [K, C] right singular vectors (transposed), K = min(N_frames, C)
    logger.info(f"Singular values — max: {S[0].item():.4f}, min: {S[-1].item():.6f}")

    # Save
    save_dir = Path(config['save-state']['directory'])
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / 'svd_VT.npy',              Vh.numpy())   # [K, C]
    np.save(save_dir / 'svd_feat_mean.npy',        feat_mean)   # [C]
    np.save(save_dir / 'svd_singular_values.npy',  S.numpy())   # [K]

    logger.info(f"Saved to {save_dir}")
    logger.info(f"  svd_VT.npy              {Vh.shape}")
    logger.info(f"  svd_feat_mean.npy       {feat_mean.shape}")
    logger.info(f"  svd_singular_values.npy {S.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute SVD of core features over all training samples (KDEP-style)."
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Load model from a .pth checkpoint instead of the foundation model",
    )
    args = parser.parse_args()
    main(args)
