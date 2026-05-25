#!/usr/bin/env python3
"""
Extract visual concept vectors from stimulus frames using the CLIP image encoder.

Each stimulus frame is encoded by the CLIP image encoder and treated as a
concept vector.  Near-duplicate frames are removed via greedy cosine-similarity
thresholding, yielding a compact [K, D_clip] matrix that can be used as a
drop-in replacement for text-based concept vectors in the CBM readout.

Optionally merges with an existing text-concept .npy so the readout sees
both image-grounded and text-grounded concepts.

Usage
-----
  python scripts/extract_image_concept_vectors.py \
      --stim-dir   /project/rf/data/sensorium2023_fnn/mouseA/training/stimuli \
      --out-npy    data/concepts/concepts_image_clip.npy \
      --n-trials   100  --frames-per-trial 5 \
      --sim-thresh 0.95 \
      --clip-model ViT-bigG-14  --clip-pretrained laion2b_s39b_b160k \
      --gpu 0

  # merge with existing text concepts
  python scripts/extract_image_concept_vectors.py ... \
      --merge-npy data/concepts/concepts_visual_v2.npy \
      --out-npy   data/concepts/concepts_combined.npy
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_stimulus_paths(stim_dir: Path, n_trials: int, seed: int) -> list[Path]:
    paths = sorted(stim_dir.glob("trial*.npy"))
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:n_trials]


def sample_frames(npy_path: Path, n_frames: int, burnin: int = 30) -> list[np.ndarray]:
    arr = np.load(npy_path)          # [T, H, W, 1]
    T = arr.shape[0]
    usable = arr[burnin: T - burnin, :, :, 0]
    indices = np.linspace(0, len(usable) - 1, n_frames, dtype=int)
    return [usable[i] for i in indices]


def frame_to_pil(frame: np.ndarray, scale: int = 4) -> Image.Image:
    h, w = frame.shape
    img = Image.fromarray(frame, mode='L')
    img = img.resize((w * scale, h * scale), Image.BILINEAR)
    return img.convert('RGB')


def encode_frames(pil_images: list[Image.Image], model, preprocess,
                  device: str, batch_size: int = 64) -> np.ndarray:
    all_feats = []
    for i in range(0, len(pil_images), batch_size):
        batch = torch.stack([preprocess(img) for img in pil_images[i: i + batch_size]]).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().float())
    return torch.cat(all_feats, dim=0).numpy()


def deduplicate(vecs: np.ndarray, threshold: float) -> list[int]:
    kept = []
    for i, v in enumerate(vecs):
        if not kept:
            kept.append(i)
            continue
        if (vecs[kept] @ v).max() < threshold:
            kept.append(i)
    return kept


def main(args):
    import open_clip

    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    print(f"Loading CLIP {args.clip_model} ({args.clip_pretrained}) on {device} …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    model = model.to(device).eval()

    stim_dir = Path(args.stim_dir)
    paths = load_stimulus_paths(stim_dir, args.n_trials, seed=args.random_seed)
    print(f"Sampling {len(paths)} trials from {stim_dir}")

    all_pil: list[Image.Image] = []
    for p in paths:
        try:
            frames = sample_frames(p, args.frames_per_trial, burnin=args.burnin)
            all_pil.extend([frame_to_pil(f) for f in frames])
        except Exception as e:
            print(f"  skip {p.name}: {e}")

    print(f"Collected {len(all_pil)} frames — encoding …")
    vecs = encode_frames(all_pil, model, preprocess, device)
    print(f"  raw embeddings: {vecs.shape}")

    kept_idx = deduplicate(vecs, threshold=args.sim_thresh)
    kept_vecs = vecs[kept_idx]
    print(f"After dedup (threshold={args.sim_thresh}): {len(kept_vecs)} image concepts")

    if args.merge_npy:
        merge_path = Path(args.merge_npy)
        text_vecs = np.load(merge_path).astype(np.float32)
        print(f"Merging with {merge_path.name}: {text_vecs.shape[0]} text concepts")
        combined = np.concatenate([text_vecs, kept_vecs], axis=0)
        all_kept = deduplicate(combined, threshold=args.sim_thresh)
        kept_vecs = combined[all_kept]
        n_text = sum(1 for i in all_kept if i < len(text_vecs))
        n_img  = len(all_kept) - n_text
        print(f"After combined dedup: {len(kept_vecs)} concepts "
              f"({n_text} text, {n_img} image)")

    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, kept_vecs.astype(np.float32))
    print(f"Saved → {out_npy}  shape={kept_vecs.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract CLIP image embeddings from stimulus frames as concept vectors."
    )
    parser.add_argument('--stim-dir',         type=str,
                        default='/project/rf/data/sensorium2023_fnn/mouseA/training/stimuli')
    parser.add_argument('--out-npy',          type=str,
                        default='data/concepts/concepts_image_clip.npy')
    parser.add_argument('--merge-npy',        type=str, default=None,
                        help='Existing .npy to prepend before dedup (e.g. text concepts)')
    parser.add_argument('--n-trials',         type=int, default=100)
    parser.add_argument('--frames-per-trial', type=int, default=5)
    parser.add_argument('--burnin',           type=int, default=30)
    parser.add_argument('--sim-thresh',       type=float, default=0.95,
                        help='Cosine similarity dedup threshold '
                             '(image embeddings cluster tighter than text, use ~0.95)')
    parser.add_argument('--clip-model',       type=str, default='ViT-bigG-14')
    parser.add_argument('--clip-pretrained',  type=str, default='laion2b_s39b_b160k')
    parser.add_argument('--gpu',              type=int, default=0)
    parser.add_argument('--random-seed',      type=int, default=42)
    main(parser.parse_args())
