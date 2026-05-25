#!/usr/bin/env python3
"""
Learn a cluster-based 3D position embedding for neurons.

The embedding maps each neuron's (x,y,z) coordinate to a d-dimensional vector
such that neurons with similar firing responses end up with similar embeddings.

Training objective
------------------
  L = α · L_contrast  +  (1-α) · L_pred

  L_contrast : response-correlation loss — MSE between the cosine-similarity
               matrix of embeddings and the Pearson-correlation matrix of
               mean firing responses (encourages functionally similar neurons
               to cluster together in embedding space).

  L_pred     : regression loss — a linear layer on top of the embedding
               predicts each neuron's mean response from its embedding.
               This keeps the embedding grounded in response space.

Usage
-----
  python scripts/learn_position_embedding.py config.yaml
  python scripts/learn_position_embedding.py config.yaml --epochs 1000 --lr 5e-4
  python scripts/learn_position_embedding.py \\
      --training_dir /project/rf/data/sensorium2023_fnn/mouseA/training \\
      --out          /project/rf/data/sensorium2023_fnn/mouseA/position_embedding.pt
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import trange
from scipy.stats import zscore
import yaml

from fnn.model.position_embeddings import (build_cluster_embedding, build_position_embedding,
                                            response_correlation_loss,
                                            precompute_sparse_pairs, sparse_response_correlation_loss,
                                            precompute_triplets, precompute_triplets_hard_neg,
                                            precompute_triplets_rastermap,
                                            triplet_loss)

DEFAULT_CONFIG = Path('data/train_digital_twin/config_position_embedding_mouseA.yaml')


# ---------------------------------------------------------------------------
# Rastermap-based cluster center initialisation
# ---------------------------------------------------------------------------

def get_rastermap_corr(
    traces:     np.ndarray,
    n_clusters: int   = 100,
    n_pcs:      int   = 200,
    locality:   float = 0.75,
    time_lag_window: int = 5,
) -> tuple:
    """
    Run Rastermap and return:
      rmap_order : (N,) 1-D neuron ordering (0 … N-1)
      C_rmap     : (N, N) float32 — cosine similarity in rastermap PC space
                   (denoised correlation; values in [-1, 1], same scale as C_resp)
    """
    from rastermap import Rastermap
    spks = zscore(traces.T, axis=1).astype(np.float32)
    spks = np.nan_to_num(spks)
    model = Rastermap(
        n_clusters      = n_clusters,
        n_PCs           = n_pcs,
        locality        = locality,
        time_lag_window = time_lag_window,
        normalize       = True,
        mean_time       = True,
    ).fit(spks)

    N          = spks.shape[0]
    rmap_order = np.empty(N, dtype=np.float32)
    rmap_order[model.isort] = np.arange(N, dtype=np.float32)

    # Cosine similarity in PC space — exact denoised correlation values
    Usv    = model.Usv.astype(np.float32)                             # (N, n_PCs)
    U_norm = Usv / (np.linalg.norm(Usv, axis=1, keepdims=True) + 1e-8)
    C_rmap = (U_norm @ U_norm.T).astype(np.float32)                  # (N, N)

    return rmap_order, C_rmap


def rastermap_cluster_centers(
    traces:     np.ndarray,
    positions:  np.ndarray,
    n_clusters: int,
    n_pcs:      int   = 200,
    locality:   float = 0.75,
    time_lag_window: int = 5,
) -> np.ndarray:
    """
    Run Rastermap on response traces and return mean xyz position of each cluster.

    Parameters
    ----------
    traces     : (T, N) raw response traces
    positions  : (N, 3) neuron xyz coordinates
    n_clusters : number of rastermap clusters

    Returns
    -------
    centers : (n_clusters, 3) mean xyz per cluster, ordered by rastermap sort
    """
    from rastermap import Rastermap
    spks = zscore(traces.T, axis=1).astype(np.float32)   # (N, T)
    spks = np.nan_to_num(spks)
    model = Rastermap(
        n_clusters      = n_clusters,
        n_PCs           = n_pcs,
        locality        = locality,
        time_lag_window = time_lag_window,
        normalize       = True,
        mean_time       = True,
    ).fit(spks)
    isort      = model.isort
    N          = len(isort)
    boundaries = np.linspace(0, N, n_clusters + 1, dtype=int)
    centers    = np.array([
        positions[isort[boundaries[c]:boundaries[c + 1]]].mean(axis=0)
        for c in range(n_clusters)
    ], dtype=np.float32)
    return centers


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_mean_responses(units_dir: Path, n_train: int = None) -> np.ndarray:
    """
    Load training unit files and return per-neuron mean response (N,).
    Uses all train-flagged trials.

    Returns
    -------
    mean_resp : (N, T_concat) — concatenated responses across training trials
                (trimmed to first min_T frames per trial for memory)
    """
    files = sorted(units_dir.glob('trial*.npy'))
    if n_train:
        files = files[:n_train]

    # First pass: find dimensions
    r0 = np.load(files[0])   # (T, N)
    T_per, N = r0.shape

    # Accumulate sum for mean (memory-efficient)
    total = np.zeros(N, dtype=np.float64)
    count = 0
    for f in files:
        r = np.load(f)   # (T, N)
        total += r.sum(axis=0)
        count += r.shape[0]
    mean_resp = (total / count).astype(np.float32)   # (N,)
    return mean_resp


def load_response_traces(units_dir: Path, max_trials: int = 50) -> np.ndarray:
    """
    Load a subset of trials and return concatenated response traces (T_total, N).
    Used to compute per-neuron response correlation for L_contrast.
    """
    files = sorted(units_dir.glob('trial*.npy'))[:max_trials]
    traces = [np.load(f) for f in files]   # each (T, N)
    return np.concatenate(traces, axis=0)  # (T_total, N)


# ---------------------------------------------------------------------------
# Linear probe head (for L_pred)
# ---------------------------------------------------------------------------

class ResponsePredictor(nn.Module):
    def __init__(self, embed_dim: int, n_neurons: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, n_neurons)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)   # (N, N) — each embedding predicts all neurons?


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Load config if provided, then apply CLI overrides
    cfg = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    def _get(cli_val, *keys, default=None):
        """Return CLI value if set, else walk cfg dict by keys, else default."""
        if cli_val is not None:
            return cli_val
        d = cfg
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d if d is not None else default

    training_dir = Path(_get(args.training_dir, 'data-source', 'training', 'directory',
                             default='/project/rf/data/sensorium2023_fnn/mouseA/training'))
    out_path     = Path(_get(args.out, 'save-state', 'path',
                             default='/project/rf/data/sensorium2023_fnn/mouseA/position_embedding.pt'))
    encoder_type = str(_get(args.encoder_type, 'model', 'encoder_type', default='cluster'))
    n_clusters   = int(_get(args.n_clusters,   'model', 'n_clusters',   default=64))
    embed_dim    = int(_get(args.embed_dim,    'model', 'embed_dim',    default=64))
    multiscale   = bool(_get(args.multiscale or None, 'model', 'multiscale', default=False))
    mlp_hidden   = int(_get(args.mlp_hidden,   'model', 'mlp_hidden',   default=256))
    mlp_layers   = int(_get(args.mlp_layers,   'model', 'mlp_layers',   default=3))
    mlp_use_xy   = bool(_get(None,             'model', 'mlp_use_xy',   default=True))
    top_k_assign   = _get(None, 'model', 'top_k_assign',   default=None)
    # leave as float (fraction) or int (absolute) — _resolve_top_k handles both
    rastermap_init = bool(_get(None, 'model', 'rastermap_init', default=False))
    freeze_centers = bool(_get(None, 'model', 'freeze_centers', default=False))
    epochs       = int(_get(args.epochs,     'training', 'epochs',  default=500))
    lr           = float(_get(args.lr,       'training', 'lr',      default=1e-3))
    alpha        = float(_get(args.alpha,    'training', 'alpha',   default=0.7))
    max_trials        = int(_get(args.max_trials,        'data-source', 'training', 'max_trials', default=50))
    contrastive_mode      = str(_get(args.contrastive_mode,      'training', 'contrastive_mode',      default='full'))
    sparse_top_k          = int(_get(args.sparse_top_k,          'training', 'sparse_top_k',          default=50000))
    n_triplets            = int(_get(args.n_triplets,             'training', 'n_triplets',             default=5000))
    triplet_margin        = float(_get(args.triplet_margin,       'training', 'triplet_margin',         default=0.3))
    triplet_pos_thr       = float(_get(args.triplet_pos_thr,      'training', 'triplet_pos_thr',       default=0.3))
    triplet_neg_thr       = float(_get(args.triplet_neg_thr,      'training', 'triplet_neg_thr',       default=0.0))
    triplet_resample_freq = int(_get(args.triplet_resample_freq,  'training', 'triplet_resample_freq', default=0))
    hard_neg_radius       = _get(args.hard_neg_radius,            'training', 'hard_neg_radius',       default=None)
    if hard_neg_radius is not None:
        hard_neg_radius = float(hard_neg_radius)
    triplet_source        = str(_get(None, 'training', 'triplet_source',        default='cresp'))
    rastermap_pos_window  = float(_get(None, 'training', 'rastermap_pos_window', default=0.05))
    rastermap_neg_window  = float(_get(None, 'training', 'rastermap_neg_window', default=0.30))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_dir = training_dir

    # Load positions
    pos_df = pd.read_csv(train_dir / 'positions.csv')
    positions = pos_df[['x', 'y', 'z']].values.astype(np.float32)   # (N, 3)
    N = len(positions)
    print(f'Neurons: {N}')
    print(f'Position range: x=[{positions[:,0].min():.0f},{positions[:,0].max():.0f}]'
          f'  y=[{positions[:,1].min():.0f},{positions[:,1].max():.0f}]'
          f'  z=[{positions[:,2].min():.0f},{positions[:,2].max():.0f}]')

    # Load responses for computing correlation structure
    print(f'Loading response traces (up to {max_trials} trials)...')
    traces = load_response_traces(train_dir / 'units', max_trials=max_trials)  # (T, N)
    print(f'Trace shape: {traces.shape}')

    # Per-neuron mean response (target for L_pred)
    mean_resp = traces.mean(axis=0)   # (N,)

    # Pre-compute response correlation matrix on CPU in float64 for accuracy
    print('Computing response correlation matrix...')
    r = traces - traces.mean(axis=0, keepdims=True)   # (T, N) zero-mean
    r_norm = r / (np.linalg.norm(r, axis=0, keepdims=True) + 1e-8)
    C_resp = (r_norm.T @ r_norm).astype(np.float32)   # (N, N)
    print(f'C_resp range: [{C_resp.min():.3f}, {C_resp.max():.3f}]')

    # Always compute rastermap PC correlation for evaluation
    print('Running Rastermap for evaluation metric ...')
    rmap_order, C_rmap = get_rastermap_corr(traces)
    print(f'  C_rmap range: [{C_rmap.min():.3f}, {C_rmap.max():.3f}]')

    # Move to device
    pos_t       = torch.tensor(positions,  device=device)
    C_resp_t    = torch.tensor(C_resp,     device=device)
    mean_resp_t = torch.tensor(mean_resp,  device=device)   # (N,)

    # Precompute sparse pair indices (done once; reused every epoch)
    if contrastive_mode == 'sparse':
        print(f'Precomputing sparse pairs (top_k={sparse_top_k} pos + neg)...')
        sp_rows, sp_cols, sp_target = precompute_sparse_pairs(C_resp, sparse_top_k)
        sp_rows_t   = torch.tensor(sp_rows,   dtype=torch.long,    device=device)
        sp_cols_t   = torch.tensor(sp_cols,   dtype=torch.long,    device=device)
        sp_target_t = torch.tensor(sp_target, dtype=torch.float32, device=device)
        print(f'  Selected {len(sp_rows):,} pairs')

    elif contrastive_mode == 'triplet':
        _hard   = hard_neg_radius is not None
        C_train = C_rmap if triplet_source == 'rastermap' else C_resp
        src_lbl = 'C_rmap' if triplet_source == 'rastermap' else 'C_resp'
        print(f'Precomputing triplets from {src_lbl} (n={n_triplets}, '
              f'pos_thr={triplet_pos_thr}, neg_thr={triplet_neg_thr}, margin={triplet_margin}'
              + (f', hard_neg_radius={hard_neg_radius:.0f}µm' if _hard else '') + ')...')
        _triplet_fn     = precompute_triplets_hard_neg if _hard else precompute_triplets
        _triplet_kwargs = dict(
            C_resp=C_train, n_triplets=n_triplets,
            pos_threshold=triplet_pos_thr, neg_threshold=triplet_neg_thr,
        )
        if _hard:
            _triplet_kwargs.update(positions=positions, hard_neg_radius=hard_neg_radius)

        tr_anchors, tr_pos, tr_neg = _triplet_fn(**_triplet_kwargs)
        tr_anchors_t = torch.tensor(tr_anchors, dtype=torch.long, device=device)
        tr_pos_t     = torch.tensor(tr_pos,     dtype=torch.long, device=device)
        tr_neg_t     = torch.tensor(tr_neg,     dtype=torch.long, device=device)
        print(f'  Sampled {len(tr_anchors):,} triplets')

    # Build embedding model
    print(f'Building embedding: encoder_type={encoder_type}  embed_dim={embed_dim}')
    model = build_position_embedding(
        positions    = positions,
        encoder_type = encoder_type,
        embed_dim    = embed_dim,
        n_clusters   = n_clusters,
        multiscale   = multiscale,
        top_k_assign = top_k_assign,
        mlp_hidden   = mlp_hidden,
        mlp_layers   = mlp_layers,
        mlp_use_xy   = mlp_use_xy,
    ).to(device)

    # Rastermap-based cluster center initialisation
    from fnn.model.position_embeddings import MultiScaleClusterEmbedding, ClusterPositionEmbedding
    if rastermap_init and encoder_type == 'cluster':
        print('Initialising cluster centers from Rastermap ...')
        if isinstance(model, MultiScaleClusterEmbedding):
            for i, scale in enumerate(model.scales):
                K = scale.n_clusters
                print(f'  scale {i}: running Rastermap with n_clusters={K} ...')
                centers = rastermap_cluster_centers(traces, positions, n_clusters=K)
                with torch.no_grad():
                    scale.centers.copy_(torch.tensor(centers, device=device))
                print(f'  scale {i}: centers initialised  {centers.shape}')
        elif isinstance(model, ClusterPositionEmbedding):
            print(f'  running Rastermap with n_clusters={model.n_clusters} ...')
            centers = rastermap_cluster_centers(traces, positions, n_clusters=model.n_clusters)
            with torch.no_grad():
                model.centers.copy_(torch.tensor(centers, device=device))
            print(f'  centers initialised  {centers.shape}')

    # Optionally freeze cluster centers
    if freeze_centers and encoder_type == 'cluster':
        if isinstance(model, MultiScaleClusterEmbedding):
            for scale in model.scales:
                scale.centers.requires_grad_(False)
        elif isinstance(model, ClusterPositionEmbedding):
            model.centers.requires_grad_(False)
        print('Cluster centers frozen (requires_grad=False)')

    # Log top-k assignment per scale
    if isinstance(model, MultiScaleClusterEmbedding):
        for i, scale in enumerate(model.scales):
            k = scale.top_k_assign if scale.top_k_assign is not None else scale.n_clusters
            print(f'  scale {i}: top_k={k}/{scale.n_clusters}  ({"global" if scale.top_k_assign is None else f"{k/scale.n_clusters:.0%}"})')
    elif isinstance(model, ClusterPositionEmbedding):
        k = model.top_k_assign if model.top_k_assign is not None else model.n_clusters
        print(f'  top_k={k}/{model.n_clusters}  ({"global" if model.top_k_assign is None else f"{k/model.n_clusters:.0%}"})')

    # Simple linear predictor: embedding → scalar mean response per neuron
    predictor = nn.Linear(embed_dim, 1).to(device)

    params = [p for p in model.parameters() if p.requires_grad] + list(predictor.parameters())
    optimizer = Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f'Training for {epochs} epochs  (α_contrast={alpha})')
    best_loss = float('inf')
    best_state = None

    log_path = out_path.parent / 'train.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, 'w')
    log_file.write('epoch\tloss\tcontrast\tpred\tlr\n')
    print(f'Logging to {log_path}')

    pbar = trange(epochs)
    for epoch in pbar:
        model.train()
        predictor.train()
        optimizer.zero_grad()

        # Forward: position → embedding
        z = model(pos_t)                      # (N, d)

        # Resample triplets periodically
        if (contrastive_mode == 'triplet' and triplet_resample_freq > 0
                and epoch > 0 and epoch % triplet_resample_freq == 0):
            _resample_kwargs = dict(
                C_resp=C_train, n_triplets=n_triplets,
                pos_threshold=triplet_pos_thr, neg_threshold=triplet_neg_thr,
                seed=epoch,
            )
            if hard_neg_radius is not None:
                _resample_kwargs.update(positions=positions, hard_neg_radius=hard_neg_radius)
            tr_anchors, tr_pos, tr_neg = _triplet_fn(**_resample_kwargs)
            tr_anchors_t = torch.tensor(tr_anchors, dtype=torch.long, device=device)
            tr_pos_t     = torch.tensor(tr_pos,     dtype=torch.long, device=device)
            tr_neg_t     = torch.tensor(tr_neg,     dtype=torch.long, device=device)

        # L_contrast: embedding cosine similarity should match response correlation
        if contrastive_mode == 'sparse':
            loss_contrast = sparse_response_correlation_loss(z, sp_rows_t, sp_cols_t, sp_target_t)
        elif contrastive_mode == 'triplet':
            loss_contrast = triplet_loss(z, tr_anchors_t, tr_pos_t, tr_neg_t, margin=triplet_margin)
        else:
            e_norm = F.normalize(z, dim=1)
            C_embed = e_norm @ e_norm.T       # (N, N)
            loss_contrast = F.mse_loss(C_embed, C_resp_t)

        # L_pred: embedding → mean response per neuron
        pred = predictor(z).squeeze(-1)       # (N,)
        loss_pred = F.mse_loss(pred, mean_resp_t)

        loss = alpha * loss_contrast + (1.0 - alpha) * loss_pred
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        log_file.write(f'{epoch}\t{loss.item():.6f}\t{loss_contrast.item():.6f}'
                       f'\t{loss_pred.item():.6f}\t{scheduler.get_last_lr()[0]:.2e}\n')
        log_file.flush()

        if epoch % 50 == 0 or epoch == epochs - 1:
            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                contrast=f'{loss_contrast.item():.4f}',
                pred=f'{loss_pred.item():.4f}',
            )

    log_file.close()

    # Restore best
    model.load_state_dict(best_state)
    model.eval()

    # Save
    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': best_state,
        'model_config': {
            'encoder_type': encoder_type,
            'n_clusters':   n_clusters,
            'embed_dim':    embed_dim,
            'multiscale':   multiscale,
            'top_k_assign': top_k_assign,
            'mlp_hidden':   mlp_hidden,
            'mlp_layers':   mlp_layers,
            'mlp_use_xy':   mlp_use_xy,
        },
        'positions': positions,
        'best_loss': best_loss,
    }, out)
    print(f'Saved → {out}')

    # Sanity check: Pearson r between embedding similarity and functional similarity
    with torch.no_grad():
        z = model(pos_t).cpu().numpy()
    e_norm     = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    C_embed_np = e_norm @ e_norm.T
    tri        = np.triu_indices(N, k=1)

    r_cresp = np.corrcoef(C_embed_np[tri], C_resp[tri])[0, 1]
    print(f'Pearson r(C_embed, C_resp):    {r_cresp:.4f}')

    r_rmap = np.corrcoef(C_embed_np[tri], C_rmap[tri])[0, 1]
    print(f'Pearson r(C_embed, C_rmap):    {r_rmap:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Learn cluster-based 3D position embedding from neural responses.'
    )
    parser.add_argument('config', type=Path, nargs='?', default=DEFAULT_CONFIG,
                        help='YAML config file (CLI flags override config values)')
    parser.add_argument('--training_dir', type=str,   default=None)
    parser.add_argument('--out',          type=str,   default=None)
    parser.add_argument('--encoder_type', type=str,   default=None,
                        help="'cluster' (default) or 'mlp'")
    parser.add_argument('--n_clusters',   type=int,   default=None)
    parser.add_argument('--embed_dim',    type=int,   default=None)
    parser.add_argument('--multiscale',   action='store_true', default=False,
                        help='Use multi-scale (3 σ values) cluster embedding')
    parser.add_argument('--mlp_hidden',   type=int,   default=None)
    parser.add_argument('--mlp_layers',   type=int,   default=None)
    parser.add_argument('--epochs',       type=int,   default=None)
    parser.add_argument('--lr',           type=float, default=None)
    parser.add_argument('--alpha',        type=float, default=None,
                        help='Weight of contrastive loss (0=pred only, 1=contrast only)')
    parser.add_argument('--max_trials',       type=int,   default=None,
                        help='Max training trials used to compute response correlations')
    parser.add_argument('--contrastive_mode', type=str,   default=None,
                        help="'full', 'sparse', or 'triplet'")
    parser.add_argument('--sparse_top_k',     type=int,   default=None,
                        help='Pairs per side when contrastive_mode=sparse')
    parser.add_argument('--n_triplets',        type=int,   default=None,
                        help='Number of triplets when contrastive_mode=triplet')
    parser.add_argument('--triplet_margin',    type=float, default=None,
                        help='Margin for triplet loss')
    parser.add_argument('--triplet_pos_thr',   type=float, default=None,
                        help='C_resp threshold for positive pairs in triplet sampling')
    parser.add_argument('--triplet_neg_thr',      type=float, default=None,
                        help='C_resp threshold for negative pairs in triplet sampling')
    parser.add_argument('--triplet_resample_freq', type=int,   default=None,
                        help='Resample triplets every N epochs (0 = never resample)')
    parser.add_argument('--hard_neg_radius',       type=float, default=None,
                        help='Spatial radius (µm) for hard negative mining (None = random negatives)')
    args = parser.parse_args()
    main(args)
