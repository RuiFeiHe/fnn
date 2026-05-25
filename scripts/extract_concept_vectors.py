#!/usr/bin/env python3
"""
Extract concept vectors from a VLM text encoder (OpenCLIP).

For each concept string in the YAML file, tokenises and encodes it with the
chosen OpenCLIP text encoder, then saves the resulting embedding matrix as a
numpy array that CBMReadout can load directly.

Usage
-----
  python scripts/extract_concept_vectors.py
  python scripts/extract_concept_vectors.py \
      --concepts data/concepts/concepts_visual.yaml \
      --out      data/concepts/concepts_visual.npy \
      --model    ViT-bigG-14  --pretrained laion2b_s39b_b160k

Available models (some examples):
  ViT-B-32        pretrained: laion2b_s34b_b79k       (D=512)
  ViT-L-14        pretrained: laion2b_s32b_b82k        (D=768)
  ViT-H-14        pretrained: laion2b_s32b_b79k        (D=1024)
  ViT-bigG-14     pretrained: laion2b_s39b_b160k       (D=1280)
  hf-hub:timm/ViT-SO400M-14-SigLIP   pretrained: webli   (D=1152)

Run  python -m open_clip.pretrained  to list all available checkpoints.

Requirements
------------
  pip install open_clip_torch
"""

import argparse
from pathlib import Path

import numpy as np
import yaml


DEFAULT_CONCEPTS = Path(__file__).parent.parent / 'data' / 'concepts' / 'concepts_visual.yaml'
DEFAULT_OUT      = Path(__file__).parent.parent / 'data' / 'concepts' / 'concepts_visual.npy'


def load_concepts(yaml_path: Path) -> list[str]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    concepts = data['concepts']
    # Strip inline comments if any (shouldn't be present in pure YAML)
    return [str(c).strip() for c in concepts]


def extract(concepts: list[str], model_name: str, pretrained: str, device: str) -> np.ndarray:
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip_torch is required.  Install with:  pip install open_clip_torch"
        )

    import torch

    print(f"Loading OpenCLIP model: {model_name}  pretrained={pretrained}")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Encode all concepts in one batch (they're short strings)
    tokens = tokenizer(concepts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)           # [K, D_vlm]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    vectors = text_features.cpu().float().numpy()
    print(f"Encoded {len(concepts)} concepts → shape {vectors.shape}")
    return vectors


def main(args):
    concepts = load_concepts(args.concepts)
    print(f"Loaded {len(concepts)} concepts from {args.concepts}")
    for i, c in enumerate(concepts):
        print(f"  [{i:3d}] {c}")
    print()

    device = args.device
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    vectors = extract(concepts, args.model, args.pretrained, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vectors)
    print(f"Saved → {args.out}")

    # Also save concept strings alongside the vectors for reference
    txt_out = args.out.with_suffix('.txt')
    with open(txt_out, 'w') as f:
        for c in concepts:
            f.write(c + '\n')
    print(f"Saved concept list → {txt_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concepts',   type=Path,  default=DEFAULT_CONCEPTS,
                        help='Path to concepts YAML file')
    parser.add_argument('--out',        type=Path,  default=DEFAULT_OUT,
                        help='Output path for concept vectors (.npy)')
    parser.add_argument('--model',      type=str,   default='ViT-bigG-14',
                        help='OpenCLIP model name')
    parser.add_argument('--pretrained', type=str,   default='laion2b_s39b_b160k',
                        help='OpenCLIP pretrained weights tag')
    parser.add_argument('--device',     type=str,   default='auto',
                        help='Device: auto | cpu | cuda')
    args = parser.parse_args()
    main(args)
