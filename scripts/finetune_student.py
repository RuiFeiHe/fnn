#!/usr/bin/env python
"""
Fine-tune a distilled student (network_s): load a distilled core checkpoint,
freeze the core (and modulation.lstm), then train the readout on labelled data.

The distilled checkpoint may have been saved with a different unit count, so
only core and modulation.lstm weights are transferred; the readout is
re-initialised from scratch for the units in the target dataset.

Config keys:
    data-source.student.checkpoint   : path to distilled network_s .pth checkpoint
    finetune.freeze_modules          : list of module names to freeze (default: ["core", "modulation.lstm"])

Launch with:
    torchrun --nproc_per_node=<num_gpus> scripts/finetune_student.py [config]
"""

import os
import shutil
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm

from fnn.data import load_training_data, load_evaluation_data
from fnn.microns.build import network_s
from fnn.microns import load_network_from_params
from fnn.train.schedulers import CosineLr
from fnn.train.optimizers import SgdClip
from fnn.train.loaders import Batches
from fnn.train.objectives import NetworkLoss
from fnn.train.parallel import ParameterGroup
from fnn import evaluate
from fnn.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG = Path('/project/rf/code/fnn/data/train_digital_twin/config_finetune_student.yaml')


def run_evaluation(ckpt_path, config, save_dir):
    """Load a checkpoint and compute CC_abs, CC_max, CC_norm on the evaluation set."""
    logger.info(f"Evaluating {ckpt_path.name}")
    model = load_network_from_params(ckpt_path)
    model.eval()

    eval_data_dir = Path(config['data-source']['evaluation']['directory'])
    logger.info(f"Loading evaluation data from {eval_data_dir}")
    evaluation_data = load_evaluation_data(eval_data_dir)

    burnin_frames = config['objective']['burnin_frames']

    units_fmt = evaluate.format_responses(
        evaluation_data['units'], burnin_frames=burnin_frames
    )
    cc_max = evaluate.compute_cc_max(units_fmt)

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

    units_pred_fmt = evaluate.format_responses(units_pred, burnin_frames=burnin_frames)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc_abs = evaluate.compute_cc_abs(units_pred_fmt, units_fmt)

    cc_norm = cc_abs / cc_max

    def _stats(arr, label):
        valid = arr[np.isfinite(arr)]
        n_valid, n_total = len(valid), len(arr)
        if n_valid == 0:
            logger.info(f"{label} — no valid units (all NaN/Inf)")
        else:
            logger.info(f"{label} — median: {np.median(valid):.4f}  mean: {np.mean(valid):.4f}  "
                        f"valid units: {n_valid}/{n_total}")

    _stats(cc_max,  "CC_max ")
    _stats(cc_abs,  "CC_abs ")
    _stats(cc_norm, "CC_norm")

    def _median(arr):
        valid = arr[np.isfinite(arr)]
        return np.median(valid) if len(valid) > 0 else np.nan

    out_path = save_dir / f"eval_{ckpt_path.stem}.npz"
    np.savez(
        out_path,
        cc_max=_median(cc_max),
        cc_abs=_median(cc_abs),
        cc_norm=_median(cc_norm),
    )
    logger.info(f"Evaluation results saved to {out_path}")


def main(args):
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    is_main = rank == 0

    if is_main:
        logger.info(f"Config file: {args.config}")
        logger.info(f"DDP initialized with {world_size} GPUs")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    student_ckpt   = Path(config['data-source']['student']['checkpoint'])
    freeze_modules = config.get('finetune', {}).get(
        'freeze_modules', ['core', 'modulation.lstm']
    )

    if is_main:
        logger.info(f"Student checkpoint : {student_ckpt}")
        logger.info(f"Freeze modules     : {freeze_modules}")

    # LOAD DATASET
    data_dir  = config['data-source']['training'].get('directory', None)
    max_items = config['data-source']['training'].get('max_items', None)
    if is_main:
        logger.info(f"Loading dataset from {data_dir}")
    dataset = load_training_data(data_dir, max_items)
    if is_main:
        n_train = dataset.df.training.sum()
        n_val   = (~dataset.df.training).sum()
        s_train = dataset.df.loc[dataset.df.training,  'samples'].sum()
        s_val   = dataset.df.loc[~dataset.df.training, 'samples'].sum()
        logger.info(f"Dataset: {n_train} training trials ({s_train} frames), "
                    f"{n_val} validation trials ({s_val} frames)")

    units = len(dataset.df.units.iloc[0][0])
    if is_main:
        logger.info(f"Units in dataset: {units}")

    # BUILD MODEL — fresh network_s sized to target dataset units
    if is_main:
        logger.info("Building network_s for target dataset.")
    torch.cuda.empty_cache()
    model = network_s(units=units).to(device)

    # TRANSFER WEIGHTS FROM DISTILLED CHECKPOINT
    # Load all non-readout weights (core, perspective, modulation) so the frozen
    # core receives inputs from the same perspective it co-evolved with during
    # distillation. Only the readout is kept at random init because the unit
    # count may differ between the distillation and finetuning datasets.
    if is_main:
        logger.info(f"Loading distilled checkpoint: {student_ckpt}")
    distilled_state = torch.load(student_ckpt, map_location='cpu')
    non_readout_state = {k: v for k, v in distilled_state.items() if not k.startswith('readout.')}
    missing, unexpected = model.load_state_dict(non_readout_state, strict=False)
    if is_main:
        readout_keys = [k for k in missing if k.startswith('readout.')]
        other_missing = [k for k in missing if not k.startswith('readout.')]
        logger.info(f"  Loaded {len(non_readout_state)} tensors (readout left at random init: {len(readout_keys)})")
        if other_missing:
            logger.warning(f"  Unexpected missing keys: {other_missing}")
        if unexpected:
            logger.warning(f"  Unexpected extra keys: {unexpected}")

    for module_name in freeze_modules:
        model.module(module_name).freeze(True)
        if is_main:
            logger.info(f"  Froze: {module_name}")

    # Broadcast all parameters from rank 0
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # TRAINING COMPONENTS
    if is_main:
        logger.info("Building training components.")
    scheduler = CosineLr(**config['scheduler'])
    optimizer = SgdClip(**config['optimizer'])
    loader    = Batches(**config['loader'])
    objective = NetworkLoss(**config['objective'])

    scheduler._init(epoch=0, cycle=0)
    optimizer._init(scheduler=scheduler)
    loader._init(dataset=dataset)
    objective._init(network=model)

    trainable = {k: p for k, p in model.named_parameters() if p.requires_grad}
    if is_main:
        n_trainable = sum(p.numel() for p in trainable.values())
        logger.info(f"Trainable parameter tensors : {len(trainable)}")
        logger.info(f"Trainable scalar parameters : {n_trainable:,}")
    ddp_group = ParameterGroup(trainable)

    # SAVE PATHS
    if is_main:
        save_dir = Path(config['save-state']['directory'])
        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, save_dir / args.config.name)
        metrics_csv = save_dir / config['save-state']['metrics_csv']
        metrics_pt  = save_dir / config['save-state']['metrics_tensor']
        final_ckpt  = save_dir / config['save-state']['state_dict']
        best_ckpt   = save_dir / config['save-state'].get('best_state_dict', 'best_state_dict.pth')
        init_ckpt   = save_dir / config['save-state'].get('init_state_dict', 'init_state_dict.pth')
        torch.save(model.state_dict(), init_ckpt)
        ckpt_epochs = {init_ckpt.name: 0}
        torch.save(ckpt_epochs, save_dir / 'ckpt_epochs.pt')
        logger.info(f"Saving to {save_dir}")

    # TRAINING LOOP
    if is_main:
        logger.info("Starting fine-tuning.")
    best_val = float('inf')
    epochs, metrics = [], []
    csv_initialized = False

    for epoch, info_dict in optimizer.optimize(
        loader=loader,
        objective=objective,
        parameters=model.named_parameters(),
        groups=[ddp_group],
    ):
        if is_main:
            row = {k: v.item() if hasattr(v, 'item') else v for k, v in info_dict.items()}
            epochs.append(epoch)
            metrics.append(row)

            pd.DataFrame([{"epoch": epoch, **row}]).to_csv(
                metrics_csv,
                mode='w' if not csv_initialized else 'a',
                header=not csv_initialized,
                index=False,
            )
            csv_initialized = True

            torch.save({"epochs": epochs, "metrics": metrics}, metrics_pt)
            torch.save(model.state_dict(), final_ckpt)
            ckpt_epochs[final_ckpt.name] = epoch
            torch.save(ckpt_epochs, save_dir / 'ckpt_epochs.pt')

            val_obj = row.get("validation_objective")
            if val_obj is not None and val_obj < best_val:
                best_val = val_obj
                torch.save(model.state_dict(), best_ckpt)
                ckpt_epochs[best_ckpt.name] = epoch
                torch.save(ckpt_epochs, save_dir / 'ckpt_epochs.pt')
                logger.info(f"Epoch {epoch}: new best val objective {best_val:.6f} → {best_ckpt.name}")

            val_keys = [k for k in row if k.startswith("validation_")]
            if val_keys:
                val_str = "  ".join(f"{k}={row[k]:.6f}" for k in val_keys)
                logger.info(f"Epoch {epoch} validation: {val_str}")

            logger.info(f"Epoch {epoch}: {row}")

    dist.barrier()
    dist.destroy_process_group()

    # EVALUATE (rank 0 only, after DDP teardown)
    if is_main:
        logger.info("=" * 60)
        logger.info("Fine-tuning complete. Running evaluation.")

        for ckpt in [best_ckpt, final_ckpt]:
            if ckpt.exists():
                ep = ckpt_epochs.get(ckpt.name, "?")
                logger.info(f"--- {ckpt.name} (epoch {ep}) ---")
                run_evaluation(ckpt, config, save_dir)

        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune distilled student (network_s) readout on labelled data."
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()
    main(args)
