#!/usr/bin/env python
"""
Train a CBM (Concept Bottleneck Model) readout on neural data.

Architecture summary
--------------------
Core (frozen, from foundation model)
  → [N, S*C, H, W]
  → CBMReadout:
      cross-attention Q=pos_emb[neurons], K/V=spatial features [N, H*W, C]
      → [N, U, D_vlm]  (pos_embed_dim == D_vlm, no projection needed)
      → concept scores [N, U, K]  (dot product with frozen concept vectors)
      → readout_weights → [N, U, 1]
  → Poisson unit → [N, U]

The core weights come from the Microns foundation model (frozen by default).
Concept vectors come from a pre-extracted .npy file (frozen by default).
The position embedding comes from a pre-trained checkpoint (frozen by default).

The only newly trained parameters are:
  readout.cross_attn      — cross-attention weights (Q/K/V projections)
  readout.readout_weights — concept-score → response weights (per-neuron or shared)
  readout.out_bias      — per-neuron bias
  perspective.*         — spatial perspective adaptor (not frozen)
  modulation.*          — gain-modulation adaptor (not frozen)

Launch (single GPU):
    python scripts/train_cbm.py data/train_digital_twin/config_cbm_mouseA.yaml

Launch (multi-GPU DDP):
    torchrun --nproc_per_node=<N> scripts/train_cbm.py config.yaml

Options:
    --timestamp YYYYMMDD_HHMMSS   appended to save-state.directory
"""

import os
import shutil
import warnings
import argparse
import logging as _logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm

from fnn.data import load_training_data, load_evaluation_data
from fnn.microns.build import network_cbm
from fnn.microns import scan as microns_scan
from fnn.model.position_embeddings import build_position_embedding
from fnn.train.schedulers import CosineLr
from fnn.train.optimizers import SgdClip
from fnn.train.loaders import Batches
from fnn.train.objectives import NetworkLoss
from fnn.train.parallel import ParameterGroup
from fnn import evaluate
from fnn.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG = Path('data/train_digital_twin/config_cbm_mouseA.yaml')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pos_embedding(pos_ckpt_path: Path):
    """
    Reconstruct and return the position embedding module from a checkpoint
    saved by learn_position_embedding.py.

    Returns
    -------
    nn.Module  — restored position embedding (eval mode, on CPU)
    np.ndarray — (U, 3) neuron positions in µm
    int        — embedding output dimension
    """
    state = torch.load(pos_ckpt_path, map_location='cpu', weights_only=False)
    positions    = state['positions']          # (U, 3) numpy array
    model_config = state['model_config']
    model_state  = state['model_state_dict']

    pos_emb = build_position_embedding(positions, **model_config)
    pos_emb.load_state_dict(model_state)
    pos_emb.eval()

    # Infer output dim
    if hasattr(pos_emb, 'out_dim'):
        pos_embed_dim = pos_emb.out_dim
    else:
        pos_embed_dim = model_config.get('embed_dim', 96)

    return pos_emb, positions, pos_embed_dim


def rebuild_cbm_network(cbm_ckpt_path: Path, config: dict, device='cpu'):
    """
    Rebuild a CBM network from a saved checkpoint + config dict.
    Used for post-training evaluation.

    Parameters
    ----------
    cbm_ckpt_path : Path  — state_dict.pth saved during training
    config        : dict  — the full YAML config (for paths and hyperparams)
    device        : str

    Returns
    -------
    nn.Module  — loaded CBM network in eval mode on the requested device
    np.ndarray — (U, 3) neuron positions
    """
    pos_ckpt = Path(config['data-source']['position_embedding'])
    pos_emb, positions, pos_embed_dim = load_pos_embedding(pos_ckpt)

    concept_vectors = Path(config['data-source']['concept_vectors'])
    model_cfg      = config.get('model', {})
    n_heads        = int(model_cfg.get('n_heads',       8))
    attn_drop      = float(model_cfg.get('attn_drop',   0.0))
    score_temp     = float(model_cfg.get('score_temp',  1.0))
    per_neuron_out = bool(model_cfg.get('per_neuron_out', False))

    # Infer unit count from state_dict
    sd = torch.load(cbm_ckpt_path, map_location='cpu', weights_only=False)
    units = sd['readout.out_bias'].shape[0]

    net = network_cbm(
        units=units,
        concept_vectors=concept_vectors,
        pos_embedding=pos_emb,
        pos_embed_dim=pos_embed_dim,
        n_heads=n_heads,
        attn_drop=attn_drop,
        score_temp=score_temp,
        freeze_concepts=True,
        freeze_pos=True,
        per_neuron_out=per_neuron_out,
    )
    # strict=False: tolerates checkpoints saved before attn_bias was added
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"rebuild_cbm_network: missing keys (old ckpt?): {missing}")
    net.to(device)
    net.eval()
    return net, positions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(cbm_ckpt_path: Path, config: dict, eval_dir: Path,
                   burnin_frames: int, save_dir: Path, units_scale: float = 1.0):
    logger.info(f"Evaluating {cbm_ckpt_path.name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = rebuild_cbm_network(cbm_ckpt_path, config, device=device)

    logger.info(f"Loading evaluation data from {eval_dir}")
    evaluation_data = load_evaluation_data(eval_dir)

    s = evaluation_data['stimuli']
    p = evaluation_data['perspectives']
    m = evaluation_data['modulations']
    units_gt = evaluation_data['units']

    if units_scale != 1.0:
        units_gt = [[r * units_scale for r in video] for video in units_gt]

    units_pred = []
    with torch.no_grad():
        for i in tqdm(range(len(s)), desc="Stimuli"):
            repeats_pred = []
            for j in range(len(s[i])):
                repeats_pred.append(
                    model.predict(stimuli=s[i][j], perspectives=p[i][j], modulations=m[i][j])
                )
            units_pred.append(repeats_pred)

    def _stats(arr, label):
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            logger.info(f"{label} — no valid units")
        else:
            logger.info(f"{label} — median: {np.median(valid):.4f}  mean: {np.mean(valid):.4f}  "
                        f"valid: {len(valid)}/{len(arr)}")

    def _median(arr):
        valid = arr[np.isfinite(arr)]
        return float(np.median(valid)) if len(valid) > 0 else float('nan')

    def _eval_subset(gt, pred, label):
        gt_fmt   = evaluate.format_responses(gt,   burnin_frames=burnin_frames)
        pred_fmt = evaluate.format_responses(pred, burnin_frames=burnin_frames)
        cc_max = evaluate.compute_cc_max(gt_fmt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cc_abs = evaluate.compute_cc_abs(pred_fmt, gt_fmt)
        cc_norm = cc_abs / cc_max
        logger.info(f"  [{label}]")
        _stats(cc_max,  "  CC_max ")
        _stats(cc_abs,  "  CC_abs ")
        _stats(cc_norm, "  CC_norm")
        return _median(cc_max), _median(cc_abs), _median(cc_norm)

    n_repeats = len(units_gt[0])
    logger.info(f"Evaluation — all {n_repeats} repeats:")
    cc_max_all, cc_abs_all, cc_norm_all = _eval_subset(units_gt, units_pred, f"all {n_repeats}")

    save_kwargs = dict(
        cc_max_all=cc_max_all, cc_abs_all=cc_abs_all, cc_norm_all=cc_norm_all,
        n_repeats_all=n_repeats,
    )
    if n_repeats >= 30:
        n_sub   = 10
        sub_idx = sorted(np.random.choice(n_repeats, n_sub, replace=False))
        logger.info(f"Evaluation — random {n_sub} repeats (indices {sub_idx}):")
        gt_sub   = [[units_gt[i][j]   for j in sub_idx] for i in range(len(units_gt))]
        pred_sub = [[units_pred[i][j] for j in sub_idx] for i in range(len(units_pred))]
        cc_max_sub, cc_abs_sub, cc_norm_sub = _eval_subset(gt_sub, pred_sub, f"random {n_sub}")
        save_kwargs.update(
            cc_max_sub=cc_max_sub, cc_abs_sub=cc_abs_sub, cc_norm_sub=cc_norm_sub,
            n_repeats_sub=n_sub, sub_indices=np.array(sub_idx),
        )

    out = save_dir / f"eval_{cbm_ckpt_path.stem}.npz"
    np.savez(out, **save_kwargs)
    logger.info(f"Saved evaluation results → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # DDP init — works for both single-GPU (no torchrun) and multi-GPU runs
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        ddp = True
    else:
        rank, world_size, local_rank = 0, 1, 0
        ddp = False

    device   = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main  = (rank == 0)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if is_main:
        logger.info(f"Config: {args.config}  |  DDP={ddp}  world_size={world_size}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.timestamp:
        config['save-state']['directory'] += f'_{args.timestamp}'

    train_dir   = Path(config['data-source']['training']['directory'])
    eval_dir    = Path(config['data-source']['evaluation']['directory'])
    max_items   = config['data-source']['training'].get('max_items', None)
    fc_cfg      = config['data-source']['foundation-core']
    pos_ckpt    = Path(config['data-source']['position_embedding'])
    cv_path     = Path(config['data-source']['concept_vectors'])
    model_cfg   = config.get('model', {})
    freeze_core = bool(model_cfg.get('freeze_core', True))
    n_heads     = int(model_cfg.get('n_heads',     8))
    attn_drop   = float(model_cfg.get('attn_drop', 0.0))
    score_temp  = float(model_cfg.get('score_temp', 1.0))
    freeze_concepts  = bool(model_cfg.get('freeze_concepts',  True))
    freeze_pos       = bool(model_cfg.get('freeze_pos',       True))
    per_neuron_out   = bool(model_cfg.get('per_neuron_out',   False))
    burnin      = config['objective']['burnin_frames']
    units_scale = float(config['objective'].get('units_scale', 1.0))

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    if is_main:
        logger.info(f"Loading dataset from {train_dir}")
    dataset = load_training_data(str(train_dir), max_items)
    if is_main:
        n_train = dataset.df.training.sum()
        n_val   = (~dataset.df.training).sum()
        s_train = dataset.df.loc[dataset.df.training,  'samples'].sum()
        s_val   = dataset.df.loc[~dataset.df.training, 'samples'].sum()
        logger.info(f"Dataset: {n_train} train trials ({s_train} frames), "
                    f"{n_val} val trials ({s_val} frames)")

    units = len(dataset.df.units.iloc[0][0])
    if is_main:
        logger.info(f"Units: {units}")

    # -----------------------------------------------------------------------
    # Position embedding
    # -----------------------------------------------------------------------
    if is_main:
        logger.info(f"Loading position embedding from {pos_ckpt}")
    pos_emb, positions, pos_embed_dim = load_pos_embedding(pos_ckpt)
    if is_main:
        logger.info(f"  pos_embed_dim={pos_embed_dim}  neurons={len(positions)}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    torch.cuda.empty_cache()
    model = network_cbm(
        units=units,
        concept_vectors=cv_path,
        pos_embedding=pos_emb,
        pos_embed_dim=pos_embed_dim,
        n_heads=n_heads,
        attn_drop=attn_drop,
        score_temp=score_temp,
        freeze_concepts=freeze_concepts,
        freeze_pos=freeze_pos,
        per_neuron_out=per_neuron_out,
    ).to(device)

    # Cache per-neuron position embeddings
    model.readout.set_pos_embeddings(positions)
    if is_main:
        logger.info("  pos_emb buffer set.")

    # Optionally seed cross-attn spatial bias from a pretrained PositionFeature readout
    spatial_bias_ckpt = config['data-source'].get('spatial_bias_ckpt')
    if spatial_bias_ckpt:
        spatial_bias_ckpt = Path(spatial_bias_ckpt)
        if is_main:
            logger.info(f"  Initialising spatial attn bias from {spatial_bias_ckpt}")
        sd_pf = torch.load(spatial_bias_ckpt, map_location='cpu', weights_only=False)
        mu_2d = sd_pf['readout.position.mu'].numpy()   # [U, 2]
        model.readout.init_spatial_bias(mu_2d)
        if is_main:
            logger.info(f"  Spatial bias initialised from Gaussian.mu {mu_2d.shape}")

    # -----------------------------------------------------------------------
    # Foundation core transfer
    # -----------------------------------------------------------------------
    if is_main:
        logger.info(f"Loading foundation core from {fc_cfg['directory']} "
                    f"(session={fc_cfg['session']}, scan_idx={fc_cfg['scan_idx']})")
    foundation_model, _ = microns_scan(
        session=fc_cfg['session'],
        scan_idx=fc_cfg['scan_idx'],
        directory=fc_cfg['directory'],
        cuda=False,
    )
    core_state = {
        k[len('core.'):]: v
        for k, v in foundation_model.state_dict().items()
        if k.startswith('core.')
    }
    model.module('core').load_state_dict(core_state)
    if is_main:
        logger.info("  Core weights transferred.")

    if freeze_core:
        model.module('core').freeze(True)
        if is_main:
            logger.info("  Core frozen.")
    else:
        if is_main:
            logger.info("  Core trainable (freeze_core=false).")

    # Broadcast parameters from rank 0 to ensure identical init
    for param in model.parameters():
        if ddp:
            dist.broadcast(param.data, src=0)

    # -----------------------------------------------------------------------
    # Training components
    # -----------------------------------------------------------------------
    sched_cfg = dict(config['scheduler'])
    n_cycles  = int(sched_cfg.pop('n_cycles', 1))
    scheduler = CosineLr(**sched_cfg)
    opt_cfg = dict(config['optimizer'])
    core_lr_scale     = float(opt_cfg.pop('core_lr_scale',     1.0))
    attn_bias_lr_scale = float(opt_cfg.pop('attn_bias_lr_scale', 1.0))
    _core_decay       = opt_cfg.pop('core_decay',   None)
    core_decay        = float(_core_decay) if _core_decay is not None else None
    _readout_decay    = opt_cfg.pop('readout_decay', None)
    readout_decay     = float(_readout_decay) if _readout_decay is not None else None
    optimizer = SgdClip(**opt_cfg)
    loader    = Batches(**config['loader'])
    objective = NetworkLoss(**config['objective'])

    loader._init(dataset=dataset)
    objective._init(network=model)

    trainable        = {k: p for k, p in model.named_parameters() if p.requires_grad}
    core_params      = {k: p for k, p in trainable.items() if k.startswith('core.')}
    attn_bias_params = {k: p for k, p in trainable.items() if k == 'readout.attn_bias'}
    readout_params   = {k: p for k, p in trainable.items()
                        if k.startswith('readout.') and k != 'readout.attn_bias'}
    other_params     = {k: p for k, p in trainable.items()
                        if not k.startswith('core.') and not k.startswith('readout.')}

    if is_main:
        n_p = sum(p.numel() for p in trainable.values())
        logger.info(f"Trainable tensors: {len(trainable)}  scalars: {n_p:,}")
        for label, params in [('core', core_params), ('readout', readout_params),
                               ('readout.attn_bias', attn_bias_params), ('other', other_params)]:
            if params:
                n = sum(p.numel() for p in params.values())
                logger.info(f"  {label}: {n:,} params")

    ddp_group = ParameterGroup(trainable) if ddp else None

    # -----------------------------------------------------------------------
    # Save paths — defined on all ranks (needed for cycle-restart reloading)
    # -----------------------------------------------------------------------
    save_dir   = Path(config['save-state']['directory'])
    final_ckpt = save_dir / config['save-state']['state_dict']
    best_ckpt  = save_dir / config['save-state'].get('best_state_dict', 'best_state_dict.pth')
    init_ckpt  = save_dir / config['save-state'].get('init_state_dict', 'init_state_dict.pth')

    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, save_dir / args.config.name)

        log_path = save_dir / 'train.log'
        fh = _logging.FileHandler(log_path, mode='a')
        from fnn.utils.logging import _UTC_FMT
        fh.setFormatter(_UTC_FMT)
        logger.addHandler(fh)

        metrics_csv = save_dir / config['save-state']['metrics_csv']
        metrics_pt  = save_dir / config['save-state']['metrics_tensor']
        torch.save(model.state_dict(), init_ckpt)
        ckpt_epochs = {init_ckpt.name: 0}
        torch.save(ckpt_epochs, save_dir / 'ckpt_epochs.pt')
        logger.info(f"Saving to {save_dir}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    if is_main:
        logger.info(f"Starting training  (n_cycles={n_cycles}).")
    best_val = float('inf')
    epochs, metrics_list = [], []
    csv_initialized = False
    cycle_size = scheduler.cycle_size

    for cycle_idx in range(n_cycles):
        if is_main and n_cycles > 1:
            logger.info(f"--- Cycle {cycle_idx + 1}/{n_cycles} ---")

        scheduler._init(epoch=0, cycle=cycle_idx)
        optimizer._init(scheduler=scheduler)

        while scheduler.step():
            epoch        = scheduler.epoch
            global_epoch = cycle_idx * cycle_size + epoch
            seed         = scheduler.seed + optimizer.seed
            hyperparameters = scheduler(**optimizer.hyperparameters)

            if ddp:
                ddp_group.sync_params()

            for training in [True, False]:
                with torch.random.fork_rng([local_rank] if torch.cuda.is_available() else []):
                    torch.manual_seed(seed)
                    for data in loader(training=training):
                        objective(training=training, **data)
                        if training:
                            if ddp:
                                ddp_group.sync_grads()
                            if other_params:
                                optimizer.step(other_params, **hyperparameters)
                            if readout_params:
                                ro_hp = dict(hyperparameters)
                                if readout_decay is not None:
                                    ro_hp['decay'] = readout_decay
                                optimizer.step(readout_params, **ro_hp)
                            if attn_bias_params:
                                ab_hp = dict(hyperparameters)
                                ab_hp['lr'] = hyperparameters['lr'] * attn_bias_lr_scale
                                if readout_decay is not None:
                                    ab_hp['decay'] = readout_decay
                                optimizer.step(attn_bias_params, **ab_hp)
                            if core_params:
                                core_hp = {**hyperparameters,
                                           'lr': hyperparameters['lr'] * core_lr_scale}
                                if core_decay is not None:
                                    core_hp['decay'] = core_decay
                                optimizer.step(core_params, **core_hp)

            objectives = objective.step()
            info_dict  = dict(seed=seed, **hyperparameters, **objectives)

            if is_main:
                row = {k: v.item() if hasattr(v, 'item') else v for k, v in info_dict.items()}
                epochs.append(global_epoch)
                metrics_list.append(row)

                pd.DataFrame([{"epoch": global_epoch, **row}]).to_csv(
                    metrics_csv,
                    mode='w' if not csv_initialized else 'a',
                    header=not csv_initialized,
                    index=False,
                )
                csv_initialized = True

                torch.save({"epochs": epochs, "metrics": metrics_list}, metrics_pt)
                torch.save(model.state_dict(), final_ckpt)
                ckpt_epochs[final_ckpt.name] = global_epoch
                torch.save(ckpt_epochs, save_dir / 'ckpt_epochs.pt')

                val_obj = row.get("validation_objective")
                if val_obj is not None and val_obj < best_val:
                    best_val = val_obj
                    torch.save(model.state_dict(), best_ckpt)
                    ckpt_epochs[best_ckpt.name] = global_epoch
                    torch.save(ckpt_epochs, save_dir / 'ckpt_epochs.pt')
                    logger.info(f"Epoch {global_epoch}: new best val {best_val:.6f} → {best_ckpt.name}")

                val_keys = [k for k in row if k.startswith("validation_")]
                if val_keys:
                    logger.info(f"Epoch {global_epoch} val: " +
                                "  ".join(f"{k}={row[k]:.6f}" for k in val_keys))
                logger.info(f"Epoch {global_epoch}: {row}")

        if cycle_idx < n_cycles - 1 and is_main:
            logger.info(f"  Cycle {cycle_idx + 1} done (best val={best_val:.6f}). "
                        f"Starting cycle {cycle_idx + 2} from current weights.")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        logger.info("=" * 60)
        logger.info("Training complete. Running evaluation.")
        for ckpt in [best_ckpt, final_ckpt]:
            if ckpt.exists():
                ep = ckpt_epochs.get(ckpt.name, "?")
                logger.info(f"--- {ckpt.name} (epoch {ep}) ---")
                run_evaluation(ckpt, config, eval_dir, burnin, save_dir,
                               units_scale=units_scale)
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CBM+cross-attention readout on neural data."
    )
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Appended to save-state.directory (e.g. 20260513_120000)")
    args = parser.parse_args()
    main(args)
