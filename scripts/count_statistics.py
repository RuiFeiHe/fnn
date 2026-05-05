#!/usr/bin/env python
"""
Count value-range statistics of core feature maps before and after SVD projection.

For every training frame, spatially average-pools the raw core output [N,C,H,W] -> [N,C]
and the SVD-projected output [N,K,H,W] -> [N,K], then reports per-channel and global
statistics (min, max, mean, std, and selected percentiles).

Usage:
    python scripts/count_statistics.py [config]
    CUDA_VISIBLE_DEVICES=0 python scripts/count_statistics.py /project/rf/code/fnn/data/train_digital_twin/config_distill.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from fnn.data import load_training_data
from fnn import microns
from fnn.microns.build import network_t, network_t_pool
from fnn.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG = Path('/project/rf/code/fnn/data/train_digital_twin/config_distill.yaml')

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def _stats_report(name, values):
    """
    Print summary statistics for a 2D array [N_frames, C].
    Reports both global (across all channels and frames) and per-channel stats.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{name}  shape={values.shape}  (frames x channels)")
    logger.info(f"{'='*60}")

    # Global stats (flatten everything)
    flat = values.ravel()
    pcts = np.percentile(flat, PERCENTILES)
    logger.info(f"  Global min   : {flat.min():.6f}")
    logger.info(f"  Global max   : {flat.max():.6f}")
    logger.info(f"  Global mean  : {flat.mean():.6f}")
    logger.info(f"  Global std   : {flat.std():.6f}")
    pct_str = "  ".join(f"p{p}={v:.4f}" for p, v in zip(PERCENTILES, pcts))
    logger.info(f"  Percentiles  : {pct_str}")

    # Per-channel stats (mean over frames for each channel)
    ch_mean = values.mean(axis=0)   # [C]
    ch_std  = values.std(axis=0)    # [C]
    ch_min  = values.min(axis=0)    # [C]
    ch_max  = values.max(axis=0)    # [C]
    logger.info(f"\n  Per-channel mean  — min={ch_mean.min():.4f}  max={ch_mean.max():.4f}  "
                f"mean={ch_mean.mean():.4f}  std={ch_mean.std():.4f}")
    logger.info(f"  Per-channel std   — min={ch_std.min():.4f}   max={ch_std.max():.4f}   "
                f"mean={ch_std.mean():.4f}  std={ch_std.std():.4f}")
    logger.info(f"  Per-channel min   — min={ch_min.min():.4f}   max={ch_min.max():.4f}")
    logger.info(f"  Per-channel max   — min={ch_max.min():.4f}   max={ch_max.max():.4f}")


def collect_features(model, dataset, device):
    """
    Iterate all training frames and collect spatially-pooled core features.

    Returns
    -------
    raw_feats  : np.ndarray [N_frames, C_raw]   — before SVD projection
    proj_feats : np.ndarray [N_frames, K_proj]  — after SVD projection
    """
    model.eval()

    # Hook on core module: fires before _project, captures raw [N,C,H,W] -> pool [N,C]
    raw_captured = {}

    def hook_fn(module, input, output):
        raw_captured['feat'] = output.mean(dim=(-2, -1)).detach().cpu().float()

    hook = model.module('core').register_forward_hook(hook_fn)

    raw_feats_list  = []
    proj_feats_list = []

    train_keys = dataset.keys(training=True)
    logger.info(f"Processing {len(train_keys)} training trials")

    try:
        for key in tqdm(train_keys, desc="Trials"):
            trial = dataset.load(key)
            stimuli      = trial['stimuli']       # [T, ...]
            perspectives = trial['perspectives']  # [T, P]
            modulations  = trial['modulations']   # [T, M]
            T = stimuli.shape[0]

            model.reset()
            with torch.no_grad():
                for t in range(T):
                    s, p, m, _ = model.to_tensor(
                        stimuli[t], perspectives[t], modulations[t]
                    )
                    # _forward_core: hook fires on raw core, returns [N,K,H,W] or [N,K] (pooled)
                    proj = model._forward_core(s, p, m)

                    raw_feats_list.append(raw_captured['feat'].numpy())   # [1, C]
                    if proj.dim() == 4:
                        proj = proj.mean(dim=(-2, -1))                    # [1, K]
                    proj_feats_list.append(proj.detach().cpu().float().numpy())
    finally:
        hook.remove()

    return (
        np.concatenate(raw_feats_list,  axis=0),  # [N_frames, C_raw]
        np.concatenate(proj_feats_list, axis=0),  # [N_frames, K_proj]
    )


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    svd_dir         = config.get('svd_dir', '/project/rf/code/fnn/svd')
    pts_cfg         = config.get('pts', {})
    pts_temperature = pts_cfg.get('temperature', 0.1)
    pts_n           = pts_cfg.get('n', 3)
    pooled          = config.get('distillation', {}).get('pooled', False)
    logger.info(f"PTS temperature (T) : {pts_temperature}")
    logger.info(f"PTS exponent (n)    : {pts_n}")
    logger.info(f"Pooled mode         : {pooled}")

    # Load dataset
    data_dir  = config['data-source']['training']['directory']
    max_items = config['data-source']['training'].get('max_items', None)
    logger.info(f"Loading dataset from {data_dir}")
    dataset = load_training_data(data_dir, max_items)
    n_train = dataset.df.training.sum()
    s_train = dataset.df.loc[dataset.df.training, 'samples'].sum()
    logger.info(f"Training trials: {n_train} ({s_train} frames)")

    units = len(dataset.df.units.iloc[0][0])

    # Build network_t with foundation weights
    logger.info("Building network_t and loading foundation model weights.")
    build_teacher = network_t_pool if pooled else network_t
    model = build_teacher(units=units, svd_dir=svd_dir, pts_temperature=pts_temperature, pts_n=pts_n).to(device)
    foundation_model, _ = microns.scan(**config['data-source']['foundation-core'])
    for module_name in ["core", "modulation.lstm"]:
        model.module(module_name).load_state_dict(
            foundation_model.module(module_name).state_dict()
        )
    del foundation_model

    # Collect features
    logger.info("Collecting core features over all training frames...")
    raw_feats, proj_feats = collect_features(model, dataset, device)
    logger.info(f"Raw  feature matrix : {raw_feats.shape}")
    logger.info(f"Proj feature matrix : {proj_feats.shape}")

    # Report statistics
    _stats_report("RAW CORE OUTPUT (before SVD projection)", raw_feats)
    _stats_report("SVD-PROJECTED CORE OUTPUT (after projection)", proj_feats)

    # Optionally save
    if args.save:
        save_path = Path(args.save)
        np.savez(
            save_path,
            raw_feats=raw_feats,
            proj_feats=proj_feats,
        )
        logger.info(f"\nSaved feature arrays to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistics of raw and SVD-projected core feature maps."
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save raw and projected feature arrays as .npz",
    )
    args = parser.parse_args()
    main(args)
