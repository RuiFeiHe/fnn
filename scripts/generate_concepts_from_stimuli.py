#!/usr/bin/env python3
"""
Discover visual concepts from stimulus videos using a VLM (Claude vision).

Workflow
--------
1. Randomly sample N training stimulus trials.
2. Extract M frames per trial, evenly spaced (skipping burnin/tail).
3. Upscale + convert to RGB and send batches to Claude for concept suggestions.
4. Aggregate all proposed concepts, merge with any seed concept list.
5. Deduplicate via CLIP cosine similarity (concepts within threshold are merged).
6. Save augmented YAML + extract CLIP embeddings (.npy).

Usage
-----
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/generate_concepts_from_stimuli.py \
      --stim-dir  /project/rf/data/sensorium2023_fnn/mouseA/training/stimuli \
      --seed      data/concepts/concepts_visual.yaml \
      --out-yaml  data/concepts/concepts_visual_v2.yaml \
      --out-npy   data/concepts/concepts_visual_v2.npy \
      --n-trials  60  --frames-per-trial 5  --frames-per-call 6 \
      --sim-thresh 0.85 \
      --clip-model ViT-bigG-14  --clip-pretrained laion2b_s39b_b160k \
      --gpu 0
"""

import argparse
import base64
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def load_stimulus_paths(stim_dir: Path, n_trials: int, seed: int) -> list[Path]:
    paths = sorted(stim_dir.glob("trial*.npy"))
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:n_trials]


def sample_frames(npy_path: Path, n_frames: int, burnin: int = 30) -> list[np.ndarray]:
    """Return n_frames uint8 [H, W] arrays sampled evenly from a trial."""
    arr = np.load(npy_path)           # [T, H, W, 1]
    T = arr.shape[0]
    tail = burnin
    usable = arr[burnin: T - tail, :, :, 0]  # [T', H, W]
    indices = np.linspace(0, len(usable) - 1, n_frames, dtype=int)
    return [usable[i] for i in indices]


def frame_to_pil(frame: np.ndarray, scale: int = 4) -> Image.Image:
    """Upscale a grayscale frame and convert to RGB PIL image."""
    h, w = frame.shape
    img = Image.fromarray(frame, mode='L')
    img = img.resize((w * scale, h * scale), Image.BILINEAR)
    return img.convert('RGB')


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.standard_b64encode(buf.getvalue()).decode('utf-8')


# ---------------------------------------------------------------------------
# VLM concept extraction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a vision scientist analyzing grayscale frames from natural-image videos "
    "used in mouse neuroscience experiments. "
    "Your task is to identify the visual concepts present in the provided frames."
)

USER_PROMPT = (
    "These are {n} grayscale frames from different natural-image stimulus videos. "
    "Identify the visual concepts visible across these frames. "
    "Cover a range of levels:\n"
    "  • Low-level: edges, gratings, textures, spatial frequencies, luminance gradients\n"
    "  • Mid-level: shapes, surfaces, depth cues, motion patterns, symmetry\n"
    "  • High-level: object categories (animals, people, vehicles, plants, …), "
    "scene types (landscape, indoor, urban, sky, water, …)\n\n"
    "Return ONLY valid JSON with this schema:\n"
    '  {{"concepts": ["concept phrase 1", "concept phrase 2", ...]}}\n\n'
    "Rules:\n"
    "  - Each entry must be a short descriptive phrase (2–6 words) suitable as a CLIP text prompt.\n"
    "  - Aim for 12–20 distinct concepts per call.\n"
    "  - Do NOT include concepts not clearly visible in the frames.\n"
    "  - No duplicates, no numbering, no explanation — just the JSON."
)


def call_claude(frames: list[Image.Image], model: str, client, retries: int = 3) -> list[str]:
    content = []
    for img in frames:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": pil_to_b64(img)},
        })
    content.append({
        "type": "text",
        "text": USER_PROMPT.format(n=len(frames)),
    })

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            raw = resp.content[0].text.strip()
            # Extract JSON even if model wraps it in markdown fences
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return [str(c).strip() for c in data.get("concepts", []) if c]
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"    [failed] {e}")
                return []


# ---------------------------------------------------------------------------
# Deduplication via CLIP similarity
# ---------------------------------------------------------------------------

def encode_concepts_clip(concepts: list[str], model_name: str, pretrained: str,
                         device: str) -> np.ndarray:
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    tokens = tokenizer(concepts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def deduplicate(concepts: list[str], vecs: np.ndarray, threshold: float) -> list[int]:
    """
    Greedy deduplication: keep a concept if its cosine similarity to every
    already-kept concept is below `threshold`.
    Returns indices of kept concepts (preserving input order).
    """
    kept = []
    for i, v in enumerate(vecs):
        if not kept:
            kept.append(i)
            continue
        sims = vecs[kept] @ v   # cosine similarity (vecs are unit-normed)
        if sims.max() < threshold:
            kept.append(i)
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # --- Seed concepts -----------------------------------------
    seed_concepts: list[str] = []
    if args.seed and Path(args.seed).exists():
        with open(args.seed) as f:
            data = yaml.safe_load(f)
        seed_concepts = [str(c).strip() for c in data.get('concepts', [])]
        print(f"Loaded {len(seed_concepts)} seed concepts from {args.seed}")

    # --- Sample stimulus paths ---------------------------------
    stim_dir = Path(args.stim_dir)
    paths = load_stimulus_paths(stim_dir, args.n_trials, seed=args.random_seed)
    print(f"Sampling {len(paths)} trials from {stim_dir}")

    # --- VLM concept discovery ---------------------------------
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY", args.api_key)
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY or pass --api-key")

    client = anthropic.Anthropic(api_key=api_key)

    # Collect all frames, then group into batches
    all_pil: list[Image.Image] = []
    for p in paths:
        try:
            frames = sample_frames(p, args.frames_per_trial)
            all_pil.extend([frame_to_pil(f) for f in frames])
        except Exception as e:
            print(f"  skip {p.name}: {e}")

    print(f"Collected {len(all_pil)} frames → "
          f"{len(all_pil) // args.frames_per_call + 1} API calls")

    discovered: list[str] = []
    B = args.frames_per_call
    for batch_start in range(0, len(all_pil), B):
        batch = all_pil[batch_start: batch_start + B]
        batch_idx = batch_start // B + 1
        total = len(all_pil) // B + 1
        print(f"  batch {batch_idx}/{total} ({len(batch)} frames) ...", end=' ', flush=True)
        concepts = call_claude(batch, args.claude_model, client)
        print(f"{len(concepts)} concepts")
        discovered.extend(concepts)
        time.sleep(0.3)   # be gentle with the API

    print(f"\nTotal raw concepts from VLM: {len(discovered)}")

    # --- Merge seed + discovered, then deduplicate ------------
    all_concepts = list(dict.fromkeys(seed_concepts + discovered))  # preserve order, drop dups
    print(f"Unique concept strings before similarity dedup: {len(all_concepts)}")

    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    print(f"Encoding {len(all_concepts)} concepts with CLIP ({args.clip_model}) on {device} …")
    vecs = encode_concepts_clip(all_concepts, args.clip_model, args.clip_pretrained, device)

    kept_idx = deduplicate(all_concepts, vecs, threshold=args.sim_thresh)
    kept_concepts = [all_concepts[i] for i in kept_idx]
    kept_vecs     = vecs[kept_idx]

    n_seed_kept = sum(1 for i in kept_idx if i < len(seed_concepts))
    n_new_kept  = len(kept_idx) - n_seed_kept
    print(f"After dedup (threshold={args.sim_thresh}): {len(kept_concepts)} concepts "
          f"({n_seed_kept} from seed, {n_new_kept} new)")

    # --- Save YAML --------------------------------------------
    out_yaml = Path(args.out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)

    # Separate seed-origin from new for YAML organisation
    new_concepts = [kept_concepts[i] for i in range(len(kept_concepts))
                    if kept_idx[i] >= len(seed_concepts)]

    yaml_body = (
        "# Visual concepts for neural encoding CBM readout.\n"
        "# Generated by generate_concepts_from_stimuli.py\n"
        "# Run scripts/extract_concept_vectors.py to produce the .npy embeddings.\n\n"
        "concepts:\n\n"
    )

    if seed_concepts:
        yaml_body += "  # --- Seed concepts (manually curated) ---\n"
        for c in kept_concepts[:n_seed_kept]:
            yaml_body += f'  - "{c}"\n'
        yaml_body += "\n"

    yaml_body += "  # --- VLM-discovered concepts ---\n"
    for c in new_concepts:
        yaml_body += f'  - "{c}"\n'

    out_yaml.write_text(yaml_body)
    print(f"Saved YAML → {out_yaml}")

    # --- Save .npy embeddings ---------------------------------
    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, kept_vecs.astype(np.float32))
    print(f"Saved embeddings → {out_npy}  shape={kept_vecs.shape}")

    # --- Save concept list as .txt for reference --------------
    out_txt = out_npy.with_suffix('.txt')
    out_txt.write_text('\n'.join(kept_concepts) + '\n')
    print(f"Saved concept list → {out_txt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Discover visual concepts from stimuli using Claude vision + CLIP dedup."
    )
    parser.add_argument('--stim-dir', type=str,
                        default='/project/rf/data/sensorium2023_fnn/mouseA/training/stimuli',
                        help='Directory containing trial*.npy stimulus files')
    parser.add_argument('--seed', type=str,
                        default='data/concepts/concepts_visual.yaml',
                        help='Existing concepts YAML to extend (optional)')
    parser.add_argument('--out-yaml', type=str,
                        default='data/concepts/concepts_visual_v2.yaml',
                        help='Output augmented concepts YAML')
    parser.add_argument('--out-npy', type=str,
                        default='data/concepts/concepts_visual_v2.npy',
                        help='Output CLIP embeddings for the augmented concept list')

    parser.add_argument('--n-trials',        type=int,   default=60,
                        help='Number of stimulus trials to sample')
    parser.add_argument('--frames-per-trial', type=int,  default=5,
                        help='Frames to sample from each trial')
    parser.add_argument('--frames-per-call',  type=int,  default=6,
                        help='Images per Claude API call')
    parser.add_argument('--burnin',          type=int,   default=30,
                        help='Frames to skip at start/end of each trial')
    parser.add_argument('--random-seed',     type=int,   default=42)

    parser.add_argument('--claude-model',    type=str,   default='claude-haiku-4-5-20251001',
                        help='Claude model with vision capability')
    parser.add_argument('--api-key',         type=str,   default='',
                        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')

    parser.add_argument('--clip-model',      type=str,   default='ViT-bigG-14')
    parser.add_argument('--clip-pretrained', type=str,   default='laion2b_s39b_b160k')
    parser.add_argument('--sim-thresh',      type=float, default=0.85,
                        help='CLIP cosine similarity threshold for deduplication')
    parser.add_argument('--gpu',             type=int,   default=0,
                        help='GPU index, -1 for CPU')

    main(parser.parse_args())
