"""
Concept Bottleneck Model (CBM) readout for neural encoding.

Architecture
------------
1. Spatial features:  [N, C, H, W] → flatten → [N, H*W, C]
2. Cross-attention:   Q = pos_emb           [N, U, D_pos=D_vlm]
                      K = V = spatial_feats [N, H*W, C]
                      → [N, U, D_vlm]
3. Concept scores:    [N, U, D_vlm] × [K, D_vlm]ᵀ → [N, U, K]
4. Output:            [N, U, K] → [N, U, R]

Interpretation
--------------
Cross-attention acts as a spatially-aware readout: each neuron's position
embedding (Q) attends over the spatial feature map (K/V) to extract features
relevant to its receptive field.  Because pos_embed_dim == D_vlm, the attended
features live directly in VLM concept space and are scored against frozen
concept directions without any intermediate projection.  The readout weight
W[u, k] directly expresses how much concept k drives neuron u.

References
----------
Post-hoc CBM: https://github.com/mertyg/post-hoc-cbm
"""

import math
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .readouts import Readout
from .parameters import Parameter as FnnParameter


def _convert_to_fnn_params(module: nn.Module):
    """
    Recursively replace bare nn.Parameter instances with fnn Parameter so that
    SgdClip can access .scale / .decay / .norm_dim on every parameter.
    Preserves requires_grad so frozen parameters stay frozen.
    """
    for name, p in list(module._parameters.items()):
        if p is not None and type(p) is torch.nn.Parameter:
            module._parameters[name] = FnnParameter(p.data, requires_grad=p.requires_grad)
    for child in module.children():
        _convert_to_fnn_params(child)


class CBMReadout(Readout):
    """
    Concept Bottleneck Model readout.

    Each neuron's position embedding (Q) attends over the spatial core feature
    map (K/V) via cross-attention, producing a per-neuron feature vector.
    That vector is projected to VLM concept space and scored against frozen
    concept directions, yielding per-neuron concept scores.  A linear readout
    (optionally per-neuron) maps concept scores to predicted responses.

    Parameters
    ----------
    concept_vectors : Tensor | str | Path
        [K, D_vlm] float32 — pre-extracted VLM text embeddings of K concepts.
        Pass a path to a .npy file or a torch Tensor directly.
    pos_embedding : nn.Module
        Pre-trained position embedding: positions (U, 3) → (U, D_pos).
        Loaded from a checkpoint saved by learn_position_embedding.py.
    pos_embed_dim : int
        Output dimension of pos_embedding (must match its output size).
        Also used as embed_dim for the cross-attention.
    n_heads : int
        Number of cross-attention heads.  Must divide pos_embed_dim.
    attn_drop : float
        Dropout rate inside cross-attention (applied during training).
    score_temp : float
        Temperature divisor applied to per-neuron concept scores.
        score_temp > 1 flattens the score distribution; < 1 sharpens it.
    freeze_concepts : bool
        If True (default), concept_vectors are a frozen buffer.
        If False, they become learnable parameters.
    freeze_pos : bool
        If True (default), all parameters of pos_embedding are frozen.
    per_neuron_out : bool
        If True, each neuron gets its own [K, R] readout weight W[u, k, r],
        directly encoding "how much concept k drives neuron u."
        If False, a single shared [K, R] weight is used across all neurons.

    Notes
    -----
    Call set_pos_embeddings(positions) after _init() and after moving the model
    to the target device.  This caches per-neuron position embeddings as a buffer.
    When loading from a checkpoint the buffer is restored automatically.
    """

    def __init__(
        self,
        concept_vectors: Union[str, Path, torch.Tensor],
        pos_embedding: nn.Module,
        pos_embed_dim: int,
        n_heads: int = 8,
        attn_drop: float = 0.0,
        score_temp: float = 1.0,
        freeze_concepts: bool = True,
        freeze_pos: bool = True,
        per_neuron_out: bool = False,
    ):
        super().__init__()

        # --- Concept vectors ---
        if isinstance(concept_vectors, (str, Path)):
            cv = torch.from_numpy(np.load(concept_vectors)).float()
        else:
            cv = torch.as_tensor(concept_vectors, dtype=torch.float32)

        self.K = cv.shape[0]
        self.D_vlm = cv.shape[1]
        self.score_temp = float(score_temp)

        if freeze_concepts:
            self.register_buffer('concept_vectors', cv)
        else:
            self.concept_vectors = nn.Parameter(cv)

        # --- Position embedding ---
        self.pos_embedding = pos_embedding
        if freeze_pos:
            for p in self.pos_embedding.parameters():
                p.requires_grad_(False)

        self.D_pos = int(pos_embed_dim)
        self.n_heads = int(n_heads)
        self.attn_drop = float(attn_drop)
        self.per_neuron_out = bool(per_neuron_out)

    # ------------------------------------------------------------------
    # _init — called by Visual._init
    # ------------------------------------------------------------------

    def _init(self, cores: int, readouts: int, units: int, streams: int):
        self.cores = int(cores)
        self.readouts = int(readouts)
        self.units = int(units)
        self.streams = int(streams)

        if self.D_pos % self.n_heads != 0:
            raise ValueError(
                f"pos_embed_dim={self.D_pos} must be divisible by n_heads={self.n_heads}"
            )

        # Cross-attention: Q = pos_emb [N, U, D_pos], K/V = spatial features [N, H*W, C]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.D_pos,
            num_heads=self.n_heads,
            dropout=self.attn_drop,
            kdim=self.cores,
            vdim=self.cores,
            batch_first=True,
        )

        # LayerNorm stabilises the tiny initial attn_out (which is dominated by
        # global-avg-pool of V when attention weights are near-uniform) and brings
        # scores to O(1), giving a proper gradient signal to readout_weights.
        self.layer_norm = nn.LayerNorm(self.D_pos)

        # With LayerNorm, scores std ≈ 1; scale init down so initial out ≈ 0.
        _w_std = 1.0 / math.sqrt(self.K * self.D_pos)

        # Readout from per-neuron concept scores → responses
        if self.per_neuron_out:
            # W[u, k, r]: each neuron's weight for each concept — the CBM readout matrix
            self.readout_weights = nn.Parameter(
                torch.empty(self.units, self.K, self.readouts)
            )
            nn.init.normal_(self.readout_weights, std=_w_std)
        else:
            # Shared [K, R] weight across all neurons
            self.readout_weights = nn.Parameter(
                torch.empty(self.K, self.readouts)
            )
            nn.init.normal_(self.readout_weights, std=_w_std)

        self.out_bias = nn.Parameter(torch.zeros(self.units, self.readouts))

        # Cached per-unit position embeddings [U, D_pos]; zeros until set_pos_embeddings()
        self.register_buffer('pos_emb', torch.zeros(units, self.D_pos))

        # Learnable spatial bias added to cross-attn logits [U, H*W].
        # At zero init attention is uniform; call init_spatial_bias() to seed
        # it with a Gaussian RF prior (e.g. from a pretrained PositionFeature readout)
        # so that each neuron initially attends to its known receptive-field location.
        # H*W=384 for 128×192 input through the standard 8× downsampling core.
        self.register_buffer('_spatial_hw', torch.tensor(384))
        self.attn_bias = nn.Parameter(torch.zeros(units, 384))

        # Convert all bare nn.Parameter to fnn Parameter so SgdClip can access
        # .scale / .decay / .norm_dim without AttributeError.
        _convert_to_fnn_params(self)
        # The RF prior in attn_bias should not be regularized toward zero.
        self.attn_bias.decay = False

    # ------------------------------------------------------------------
    # Set cached position embeddings
    # ------------------------------------------------------------------

    def set_pos_embeddings(self, positions: np.ndarray):
        """
        Compute and cache per-neuron position embeddings.

        Must be called after moving the model to the target device, and before
        the first forward pass.  Not needed when loading from a checkpoint.

        Parameters
        ----------
        positions : (U, 3) numpy array — neuron xyz coordinates in µm
        """
        device = self.concept_vectors.device
        self.pos_embedding.to(device)
        pos_t = torch.tensor(positions, dtype=torch.float32, device=device)
        with torch.no_grad():
            emb = self.pos_embedding(pos_t)   # [U, D_pos]
        self.pos_emb.copy_(emb.detach())

    # ------------------------------------------------------------------
    # Spatial attention bias initialisation
    # ------------------------------------------------------------------

    def init_spatial_bias(
        self,
        positions_2d: np.ndarray,
        H: int = 16,
        W: int = 24,
        sigma: float = 0.15,
    ):
        """
        Seed the cross-attention spatial bias with a Gaussian RF prior.

        After this call each neuron u attends primarily to the feature-map
        position closest to its known receptive-field centre.  The bias
        remains learnable so training can refine it.

        Parameters
        ----------
        positions_2d : (U, 2) float array — neuron screen positions in [-1, 1]
                       (x, y order, as stored in Gaussian.mu from PositionFeature).
        H, W : int — core feature-map spatial dims (default 16×24 for 128×192 input).
        sigma : float — Gaussian width in normalised [-1, 1] coordinates.
        """
        mu = torch.tensor(positions_2d, dtype=torch.float32)  # [U, 2]
        ys = torch.linspace(-1, 1, H)
        xs = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        # grid[hw] = (x, y) so ordering matches positions_2d (x first)
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # [H*W, 2]

        diff = grid.unsqueeze(0) - mu.unsqueeze(1)   # [U, H*W, 2]
        bias = -(diff ** 2).sum(-1) / (2 * sigma ** 2)  # [U, H*W]

        with torch.no_grad():
            self.attn_bias.copy_(bias.to(self.attn_bias.device))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, core: torch.Tensor, stream=None) -> torch.Tensor:
        """
        Parameters
        ----------
        core : Tensor
            [N, S*C, H, W]  when stream is None
            [N,   C, H, W]  when stream is int
        stream : int | None

        Returns
        -------
        Tensor
            [N, 1, U, R]  when stream is None  (Mean reduce collapses the 1-dim)
            [N,    U, R]  when stream is int
        """
        N = core.shape[0]

        # 1. Spatial features: → [N, C, H, W]
        if stream is None:
            # Average over S streams to get [N, C, H, W]
            _, SC, H, W = core.shape
            feat = core.reshape(N, self.streams, self.cores, H, W).mean(dim=1)
        else:
            feat = core   # already [N, C, H, W]
            H, W = feat.shape[-2], feat.shape[-1]

        # Flatten spatial dims: [N, C, H, W] → [N, H*W, C]
        kv = feat.permute(0, 2, 3, 1).reshape(N, H * W, self.cores)

        # 2. Cross-attention: Q=pos_emb, K/V=spatial features → [N, U, D_vlm]
        q = self.pos_emb.unsqueeze(0).expand(N, -1, -1)   # [N, U, D_vlm]
        # attn_bias [U, H*W] is added to logits — concentrates attention at RF prior
        attn_out, _ = self.cross_attn(q, kv, kv, attn_mask=self.attn_bias, need_weights=False)
        attn_out = self.layer_norm(attn_out)                # normalise to unit variance

        # 3. Concept scores: [N, U, K]
        c_norm = F.normalize(self.concept_vectors, dim=1)    # [K, D_vlm]
        scores = (attn_out @ c_norm.T) / self.score_temp     # [N, U, K]

        # 4. Readout: [N, U, R]
        if self.per_neuron_out:
            out = torch.einsum('nuk,ukr->nur', scores, self.readout_weights) + self.out_bias
        else:
            out = scores @ self.readout_weights + self.out_bias   # [N,U,K]@[K,R] = [N,U,R]

        if stream is None:
            return out.unsqueeze(1)   # [N, 1, U, R] — Mean reduce gives [N, U, R]
        else:
            return out                # [N, U, R]
