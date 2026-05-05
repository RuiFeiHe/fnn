#!/usr/bin/env python
"""
Feature-level knowledge distillation: train network_s (student) to match
the SVD-projected core features of network_t (teacher).

Teacher : network_t — full core (128ch/stream) + fixed SVD projection -> 64ch/stream
Student : network_s — smaller core (64ch/stream), trained from random init

Loss = MSE( student._forward_core(), teacher._forward_core() )   [per frame, no GT labels]

No readout training, no validation, no evaluation.
Both teacher and student output [N, S*64, H', W'] = [N, 256, 16, 24].

Config keys:
    data-source.teacher.checkpoint  : path to trained network_t .pth checkpoint
    distillation.alpha              : reserved / unused (loss is pure MSE for now)
    distillation.burnin_frames      : leading frames to skip in each sample (default 0)

Launch with:
    torchrun --nproc_per_node=<num_gpus> scripts/train_distill_ddp.py [config]
"""

import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
import yaml

from fnn.data import load_training_data
from fnn.microns.build import network_t, network_s
from fnn.train.schedulers import CosineLr
from fnn.train.optimizers import SgdClip
from fnn.train.loaders import Batches
from fnn.train.parallel import ParameterGroup
from fnn import microns
from fnn.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG = Path('/project/rf/code/fnn/data/train_digital_twin/config.yaml')


def distill_batch(teacher, student, batch, burnin_frames, stream):
    """
    Compute MSE distillation loss for one batch.

    Iterates frames sequentially to respect recurrent state. Teacher runs
    under no_grad; student accumulates gradients.

    Parameters
    ----------
    teacher : Visual_t   — frozen
    student : Visual     — network_s being trained
    batch   : dict with keys 'stimuli' [T,N,...], 'perspectives' [T,N,P], 'modulations' [T,N,M]
    burnin_frames : int
    stream  : int | None

    Returns
    -------
    Tensor
        scalar MSE loss (mean over valid frames and spatial positions)
    """
    stimuli      = batch['stimuli']       # [T, N, H, W, C] uint8 or similar
    perspectives = batch['perspectives']  # [T, N, P]
    modulations  = batch['modulations']   # [T, N, M]
    T = stimuli.shape[0]

    teacher.reset()
    student.reset()

    frame_losses = []

    for t in range(T):
        s, p, m, _ = student.to_tensor(stimuli[t], perspectives[t], modulations[t])

        with torch.no_grad():
            teacher_feat = teacher._forward_core(s, p, m, stream=stream)

        student_feat = student._forward_core(s, p, m, stream=stream)

        if t >= burnin_frames:
            frame_losses.append(F.mse_loss(student_feat, teacher_feat.detach()))

    return torch.stack(frame_losses).mean()


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

    svd_dir       = config.get('svd_dir', '/project/rf/code/fnn/svd')
    pts_cfg       = config.get('pts', {})
    pts_temperature = pts_cfg.get('temperature', 0.1)
    pts_n           = pts_cfg.get('n', 3)
    burnin_frames = config.get('distillation', {}).get('burnin_frames',
                       config.get('objective', {}).get('burnin_frames', 0))
    sample_stream = config.get('distillation', {}).get('sample_stream', True)

    if is_main:
        logger.info(f"SVD dir            : {svd_dir}")
        logger.info(f"PTS temperature    : {pts_temperature}")
        logger.info(f"PTS n              : {pts_n}")
        logger.info(f"Burnin frames      : {burnin_frames}")

    # LOAD DATASET  (units column used only for dataset size; GT responses ignored)
    data_dir  = config['data-source']['training'].get('directory', None)
    max_items = config['data-source']['training'].get('max_items', None)
    if is_main:
        logger.info(f"Loading dataset from {data_dir}")
    dataset = load_training_data(data_dir, max_items)
    if is_main:
        n_train = dataset.df.training.sum()
        s_train = dataset.df.loc[dataset.df.training, 'samples'].sum()
        logger.info(f"Training trials: {n_train} ({s_train} frames)")

    units = len(dataset.df.units.iloc[0][0])

    # BUILD TEACHER (network_t) — foundation core + SVD projection, fully frozen
    if is_main:
        logger.info("Building teacher (network_t) from foundation model.")
    torch.cuda.empty_cache()
    teacher = network_t(units=units, svd_dir=svd_dir, pts_temperature=pts_temperature, pts_n=pts_n).to(device)
    foundation_model, _ = microns.scan(**config['data-source']['foundation-core'])
    for module_name in ["core", "modulation.lstm"]:
        teacher.module(module_name).load_state_dict(
            foundation_model.module(module_name).state_dict()
        )
    del foundation_model
    teacher.freeze(True)
    teacher.eval()
    for param in teacher.parameters():
        dist.broadcast(param.data, src=0)

    # BUILD STUDENT (network_s) — random init, full network trained
    if is_main:
        logger.info("Building student (network_s) from random init.")
    student = network_s(units=units).to(device)
    for param in student.parameters():
        dist.broadcast(param.data, src=0)

    # TRAINING COMPONENTS
    if is_main:
        logger.info("Building training components.")
    scheduler = CosineLr(**config['scheduler'])
    optimizer = SgdClip(**config['optimizer'])
    loader    = Batches(**config['loader'])

    scheduler._init(epoch=0, cycle=0)
    optimizer._init(scheduler=scheduler)
    loader._init(dataset=dataset)

    trainable = {k: p for k, p in student.named_parameters() if p.requires_grad}
    if is_main:
        logger.info(f"Trainable parameters: {len(trainable)}")
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
        torch.save(student.state_dict(), init_ckpt)
        ckpt_epochs = {init_ckpt.name: 0}
        torch.save(ckpt_epochs, save_dir / 'ckpt_epochs.pt')
        logger.info(f"Saving to {save_dir}")

    # TRAINING LOOP
    if is_main:
        logger.info("Starting distillation training.")

    epochs_log, metrics_log = [], []
    csv_initialized = False
    best_val_loss = float('inf')

    while scheduler.step():
        epoch = scheduler.epoch
        seed  = scheduler.seed + optimizer.seed
        hyperparameters = scheduler(**optimizer.hyperparameters)

        ddp_group_list = [ddp_group]
        for g in ddp_group_list:
            g.sync_params()

        if sample_stream:
            stream = torch.randint(0, student.streams, (1,)).item()
        else:
            stream = None

        # --- training ---
        train_losses = []

        with torch.random.fork_rng([local_rank]):
            torch.manual_seed(seed)

            with student.train_context(True):
                for batch in loader(training=True):
                    loss = distill_batch(teacher, student, batch, burnin_frames, stream)

                    loss.backward()
                    for g in ddp_group_list:
                        g.sync_grads()
                    optimizer.step(trainable, **hyperparameters)

                    train_losses.append(loss.item())

        # --- validation ---
        val_losses = []

        with torch.no_grad():
            with student.train_context(False):
                for batch in loader(training=False):
                    val_loss = distill_batch(teacher, student, batch, burnin_frames, stream)
                    val_losses.append(val_loss.item())

        epoch_train_loss = float(np.mean(train_losses))
        epoch_val_loss   = float(np.mean(val_losses)) if val_losses else float('nan')

        if is_main:
            row = {
                **hyperparameters,
                'training_distill_loss':   epoch_train_loss,
                'validation_distill_loss': epoch_val_loss,
            }
            epochs_log.append(epoch)
            metrics_log.append(row)

            pd.DataFrame([{"epoch": epoch, **row}]).to_csv(
                metrics_csv,
                mode='w' if not csv_initialized else 'a',
                header=not csv_initialized,
                index=False,
            )
            csv_initialized = True

            torch.save({"epochs": epochs_log, "metrics": metrics_log}, metrics_pt)
            torch.save(student.state_dict(), final_ckpt)
            ckpt_epochs[final_ckpt.name] = epoch
            torch.save(ckpt_epochs, save_dir / 'ckpt_epochs.pt')

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(student.state_dict(), best_ckpt)
                ckpt_epochs[best_ckpt.name] = epoch
                torch.save(ckpt_epochs, save_dir / 'ckpt_epochs.pt')
                logger.info(f"Epoch {epoch}: new best val loss {best_val_loss:.6f} → {best_ckpt.name}")

            logger.info(f"Epoch {epoch}: train={epoch_train_loss:.6f}  val={epoch_val_loss:.6f}  lr={hyperparameters['lr']:.6f}")

    dist.barrier()
    dist.destroy_process_group()

    if is_main:
        logger.info("Distillation complete.")
        logger.info(f"Student core weights saved to {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature distillation: train network_s core to match network_t projected features."
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
