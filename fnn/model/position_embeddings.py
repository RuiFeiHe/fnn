"""
Cluster-based 3D position embeddings for neurons.

Design
------
Inspired by KPConv (kernel point convolution) from 3D point cloud learning:
  - K learnable cluster centers {c_k} in 3D anatomical space
  - K learnable embedding vectors {e_k} in R^d
  - Neuron at position p gets embedding z(p) = Σ_k  w_k(p) · e_k

    where  w_k(p) = softmax_k( -‖p - c_k‖² / σ² )   (Gaussian soft assignment)

This combines two ideas from point cloud DL:
  • KPConv  : learnable kernel points as cluster centers, weighted by distance
  • PointNet++  : multi-scale grouping — run ClusterPositionEmbedding at
                  several radii and concatenate for a richer descriptor

Supervision
-----------
The embedding is trained end-to-end via response-prediction loss (plugged into
the readout).  An optional auxiliary contrastive loss can be added to explicitly
pull together clusters whose neurons share correlated responses:

    L_contrast = ‖ C_resp - C_embed ‖_F²

where C_resp[i,j]  = Pearson corr of mean responses for neurons i,j
      C_embed[i,j] = cosine similarity of embeddings z_i, z_j

Classes
-------
ClusterPositionEmbedding   — single-scale RBF cluster encoder
MultiScaleClusterEmbedding — stack of ClusterPositionEmbedding at different σ
"""

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# K-means without sklearn
# ---------------------------------------------------------------------------

def _kmeans_numpy(points: np.ndarray, k: int, n_iter: int = 100, seed: int = 0) -> np.ndarray:
    """
    Simple k-means in NumPy.  Returns (k, d) cluster centers.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), k, replace=False)
    centers = points[idx].copy().astype(np.float64)
    pts = points.astype(np.float64)
    for _ in range(n_iter):
        dists   = np.linalg.norm(pts[:, None] - centers[None], axis=-1)  # (N, k)
        labels  = dists.argmin(axis=1)
        new_c   = np.array([
            pts[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
            for j in range(k)
        ])
        if np.allclose(centers, new_c, atol=1e-6):
            break
        centers = new_c
    return centers.astype(np.float32)


# ---------------------------------------------------------------------------
# Top-k assignment helper
# ---------------------------------------------------------------------------

def _resolve_top_k(top_k_assign, K: int):
    """
    Resolve top_k_assign to an absolute int for a scale with K centers.

    top_k_assign may be:
      None          → None (global assignment, attend to all K centers)
      float (0, 1]  → max(1, round(fraction * K))  — scale-proportional
      int >= 1      → min(top_k_assign, K)          — absolute, clamped to K
    """
    if top_k_assign is None:
        return None
    if isinstance(top_k_assign, float):
        return max(1, round(top_k_assign * K))
    return min(int(top_k_assign), K)


# ---------------------------------------------------------------------------
# Single-scale cluster embedding
# ---------------------------------------------------------------------------

class ClusterPositionEmbedding(nn.Module):
    """
    Map 3D neuron positions to embeddings via soft assignment to K cluster centers.

    Parameters
    ----------
    n_clusters   : K — number of learnable cluster centers
    embed_dim    : d — output embedding dimension
    sigma_init   : initial RBF bandwidth (same units as input coordinates)
    learn_sigma  : if True, σ is a learnable scalar parameter
    out_mlp      : if True, apply a 2-layer MLP after the weighted sum
    top_k_assign : sparse local assignment — None = attend to all K centers;
                   int = attend to nearest k centers (clamped to K);
                   float in (0,1] = attend to nearest fraction*K centers

    Input  : positions  (N, 3)   float32
    Output : embeddings (N, embed_dim)  float32
    """

    def __init__(
        self,
        n_clusters:   int,
        embed_dim:    int,
        sigma_init:   float = 100.0,
        learn_sigma:  bool  = True,
        out_mlp:      bool  = True,
        top_k_assign = None,
    ):
        super().__init__()
        self.n_clusters   = n_clusters
        self.embed_dim    = embed_dim
        self.top_k_assign = _resolve_top_k(top_k_assign, n_clusters)

        # Cluster centers in 3D space — (K, 3)
        self.centers = nn.Parameter(torch.empty(n_clusters, 3))
        nn.init.normal_(self.centers, std=sigma_init)

        # Per-cluster embedding vectors — (K, embed_dim)
        self.embeds = nn.Parameter(torch.empty(n_clusters, embed_dim))
        nn.init.normal_(self.embeds, std=1.0 / math.sqrt(n_clusters))

        # RBF bandwidth
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(sigma_init)), requires_grad=learn_sigma
        )

        # Optional projection after weighted sum
        if out_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.mlp = None

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        positions : (N, 3)

        Returns
        -------
        embeddings : (N, embed_dim)
        """
        # (N, K) squared Euclidean distances to each cluster center
        diff    = positions.unsqueeze(1) - self.centers.unsqueeze(0)  # (N, K, 3)
        sq_dist = (diff ** 2).sum(-1)                                  # (N, K)

        # RBF logits; optionally mask non-top-k nearest centers to -inf
        logits = -sq_dist / (self.sigma ** 2 + 1e-8)                  # (N, K)
        if self.top_k_assign is not None and self.top_k_assign < self.n_clusters:
            # kthvalue(k) returns the k-th smallest distance; mask everything farther
            kth = sq_dist.kthvalue(self.top_k_assign, dim=-1, keepdim=True).values
            logits = logits.masked_fill(sq_dist > kth, float('-inf'))
        weights = F.softmax(logits, dim=-1)                            # (N, K)

        # Weighted sum of cluster embeddings
        z = weights @ self.embeds   # (N, embed_dim)

        if self.mlp is not None:
            z = self.mlp(z)

        return z

    def init_centers_from_positions(self, positions: np.ndarray):
        """
        K-means initialisation of cluster centers from neuron positions.

        Parameters
        ----------
        positions : (N, 3) numpy array
        """
        centers = _kmeans_numpy(positions, self.n_clusters)
        with torch.no_grad():
            self.centers.copy_(torch.tensor(centers, dtype=torch.float32))
        # Set sigma to average nearest-center distance for good initial coverage
        dists = np.linalg.norm(positions[:, None] - centers[None], axis=-1)
        sigma = float(dists.min(axis=1).mean())
        with torch.no_grad():
            self.log_sigma.fill_(math.log(max(sigma, 1e-3)))


# ---------------------------------------------------------------------------
# Multi-scale variant (PointNet++ style)
# ---------------------------------------------------------------------------

class MultiScaleClusterEmbedding(nn.Module):
    """
    Run ClusterPositionEmbedding at multiple σ values and concatenate.

    Captures both fine-grained local structure (small σ) and coarse
    functional areas (large σ).

    Parameters
    ----------
    n_clusters_per_scale : list of K values, one per scale
    embed_dim_per_scale  : list of d values, one per scale
    sigmas               : list of initial σ values (one per scale)
    learn_sigma          : whether σ is learnable at each scale
    out_dim              : final projection dim (None = sum of per-scale dims)
    """

    def __init__(
        self,
        n_clusters_per_scale: Sequence[int],
        embed_dim_per_scale:  Sequence[int],
        sigmas:               Sequence[float],
        learn_sigma:          bool = True,
        out_dim:              Optional[int] = None,
        top_k_assign:         Optional[int] = None,
    ):
        super().__init__()
        assert len(n_clusters_per_scale) == len(embed_dim_per_scale) == len(sigmas)
        self.scales = nn.ModuleList([
            ClusterPositionEmbedding(
                n_clusters   = k,
                embed_dim    = d,
                sigma_init   = s,
                learn_sigma  = learn_sigma,
                out_mlp      = False,
                top_k_assign = top_k_assign,  # resolved per-scale inside ClusterPositionEmbedding
            )
            for k, d, s in zip(n_clusters_per_scale, embed_dim_per_scale, sigmas)
        ])
        total_dim = sum(embed_dim_per_scale)
        if out_dim is not None and out_dim != total_dim:
            self.proj = nn.Linear(total_dim, out_dim)
            self.out_dim = out_dim
        else:
            self.proj = None
            self.out_dim = total_dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        positions : (N, 3)

        Returns
        -------
        embeddings : (N, out_dim)
        """
        parts = [scale(positions) for scale in self.scales]
        z = torch.cat(parts, dim=-1)   # (N, total_dim)
        if self.proj is not None:
            z = self.proj(z)
        return z

    def init_centers_from_positions(self, positions: np.ndarray):
        for scale in self.scales:
            scale.init_centers_from_positions(positions)


# ---------------------------------------------------------------------------
# Auxiliary contrastive loss
# ---------------------------------------------------------------------------

def response_correlation_loss(
    embeddings: torch.Tensor,
    mean_responses: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Pull cluster embeddings towards response-correlation structure.

    Minimises  ‖ C_resp - C_embed ‖²_F  where:
      C_resp[i,j]  = Pearson correlation of mean_responses[i] and mean_responses[j]
      C_embed[i,j] = cosine similarity of embeddings[i] and embeddings[j]

    Parameters
    ----------
    embeddings    : (N, d)  — position embeddings for N neurons
    mean_responses: (N, T)  — mean response trace per neuron (T frames)
    temperature   : scale applied to cosine similarities before MSE

    Returns
    -------
    scalar loss
    """
    # Response correlation matrix
    r = mean_responses - mean_responses.mean(dim=1, keepdim=True)
    r_norm = r / (r.norm(dim=1, keepdim=True) + 1e-8)
    C_resp = r_norm @ r_norm.T    # (N, N)

    # Embedding cosine similarity matrix
    e_norm = F.normalize(embeddings, dim=1)
    C_embed = e_norm @ e_norm.T   # (N, N)

    return F.mse_loss(C_embed / temperature, C_resp / temperature)


# ---------------------------------------------------------------------------
# Sparse contrastive loss
# ---------------------------------------------------------------------------

def precompute_sparse_pairs(
    C_resp: np.ndarray,
    top_k: int,
) -> tuple:
    """
    Select the top-K most correlated and top-K least correlated neuron pairs
    from the upper triangle of C_resp.

    Returns
    -------
    rows, cols : (2K,) int64 numpy arrays — pair indices
    target     : (2K,) float32 numpy array — C_resp values for those pairs
    """
    N = C_resp.shape[0]
    triu_r, triu_c = np.triu_indices(N, k=1)
    c_flat = C_resp[triu_r, triu_c]

    k = min(top_k, len(c_flat) // 2)
    pos_idx = np.argpartition(c_flat, -k)[-k:]
    neg_idx = np.argpartition(c_flat,  k)[:k]
    sel = np.concatenate([pos_idx, neg_idx])

    return triu_r[sel], triu_c[sel], c_flat[sel]


def sparse_response_correlation_loss(
    embeddings:  torch.Tensor,
    pair_rows:   torch.Tensor,
    pair_cols:   torch.Tensor,
    target_corr: torch.Tensor,
) -> torch.Tensor:
    """
    MSE contrastive loss computed only on pre-selected neuron pairs.

    Parameters
    ----------
    embeddings  : (N, d)
    pair_rows   : (M,) long
    pair_cols   : (M,) long
    target_corr : (M,) float  — C_resp values for the selected pairs
    """
    e_norm = F.normalize(embeddings, dim=1)
    sim = (e_norm[pair_rows] * e_norm[pair_cols]).sum(dim=1)  # (M,)
    return F.mse_loss(sim, target_corr)


# ---------------------------------------------------------------------------
# Triplet loss
# ---------------------------------------------------------------------------

def precompute_triplets(
    C_resp: np.ndarray,
    n_triplets: int,
    pos_threshold: float = 0.3,
    neg_threshold: float = 0.0,
    seed: int = 0,
) -> tuple:
    """
    Sample (anchor, positive, negative) triplets from C_resp.

    For each anchor, positive = a neuron with C_resp > pos_threshold,
    negative = a neuron with C_resp < neg_threshold.
    Only anchors that have at least one valid positive and one valid negative
    are kept.

    Returns
    -------
    anchors   : (T,) int64
    positives : (T,) int64
    negatives : (T,) int64
    """
    rng = np.random.default_rng(seed)
    N = C_resp.shape[0]

    anchor_list, pos_list, neg_list = [], [], []
    anchor_order = rng.permutation(N)

    for a in anchor_order:
        pos_cands = np.where(C_resp[a] > pos_threshold)[0]
        neg_cands = np.where(C_resp[a] < neg_threshold)[0]
        pos_cands = pos_cands[pos_cands != a]
        if len(pos_cands) == 0 or len(neg_cands) == 0:
            continue
        p = rng.choice(pos_cands)
        n = rng.choice(neg_cands)
        anchor_list.append(a)
        pos_list.append(p)
        neg_list.append(n)
        if len(anchor_list) >= n_triplets:
            break

    return (np.array(anchor_list, dtype=np.int64),
            np.array(pos_list,    dtype=np.int64),
            np.array(neg_list,    dtype=np.int64))


def precompute_triplets_hard_neg(
    C_resp:        np.ndarray,
    positions:     np.ndarray,
    n_triplets:    int,
    pos_threshold: float = 0.05,
    neg_threshold: float = 0.0,
    hard_neg_radius: float = None,
    seed:          int = 0,
) -> tuple:
    """
    Like precompute_triplets but negatives are drawn from neurons that are
    spatially close to the anchor (within hard_neg_radius) yet functionally
    uncorrelated (C_resp < neg_threshold).

    These "hard negatives" are the most informative triplets because the model
    must learn to separate nearby-but-different neurons rather than just
    nearby-vs-far neurons.

    If a given anchor has no spatially-close negatives, falls back to a
    random negative from the global pool (C_resp < neg_threshold).

    Parameters
    ----------
    C_resp           : (N, N) response correlation matrix
    positions        : (N, 3) neuron xyz coordinates
    n_triplets       : number of triplets to sample
    pos_threshold    : C_resp > this → valid positive
    neg_threshold    : C_resp < this → valid negative
    hard_neg_radius  : spatial radius for hard negative search (µm).
                       If None, defaults to the median pairwise distance.
    seed             : random seed
    """
    rng = np.random.default_rng(seed)
    N   = C_resp.shape[0]

    if hard_neg_radius is None:
        sample = positions[rng.choice(N, min(500, N), replace=False)]
        dists  = np.linalg.norm(sample[:, None] - sample[None], axis=-1)
        hard_neg_radius = float(np.median(dists[np.triu_indices(len(sample), k=1)]))

    anchor_list, pos_list, neg_list = [], [], []
    anchor_order = rng.permutation(N)

    for a in anchor_order:
        pos_cands = np.where(C_resp[a] > pos_threshold)[0]
        pos_cands = pos_cands[pos_cands != a]
        if len(pos_cands) == 0:
            continue

        # Spatially close neurons
        spatial_dists = np.linalg.norm(positions - positions[a], axis=1)
        nearby = np.where(spatial_dists < hard_neg_radius)[0]
        nearby = nearby[nearby != a]

        # Hard negatives: nearby AND low C_resp
        hard_neg_cands = nearby[C_resp[a, nearby] < neg_threshold]

        if len(hard_neg_cands) > 0:
            neg_cands = hard_neg_cands
        else:
            # Fallback: any neuron with low C_resp
            neg_cands = np.where(C_resp[a] < neg_threshold)[0]
            neg_cands = neg_cands[neg_cands != a]

        if len(neg_cands) == 0:
            continue

        anchor_list.append(a)
        pos_list.append(int(rng.choice(pos_cands)))
        neg_list.append(int(rng.choice(neg_cands)))

        if len(anchor_list) >= n_triplets:
            break

    return (np.array(anchor_list, dtype=np.int64),
            np.array(pos_list,    dtype=np.int64),
            np.array(neg_list,    dtype=np.int64))


def precompute_triplets_rastermap(
    rmap_order:      np.ndarray,
    n_triplets:      int,
    pos_window:      float = 0.05,
    neg_window:      float = 0.30,
    positions:       np.ndarray = None,
    hard_neg_radius: float = None,
    seed:            int = 0,
) -> tuple:
    """
    Sample triplets using rastermap 1-D ordering as the similarity measure.

    Positive pairs  : neurons within pos_window * N positions in rastermap order
                      (functionally similar — nearby on the learned 1-D manifold)
    Negative pairs  : neurons more than neg_window * N positions apart
                      (functionally dissimilar — far on the manifold)
    Hard negatives  : if hard_neg_radius is set, negatives are additionally
                      constrained to be spatially close (within radius µm)

    Parameters
    ----------
    rmap_order      : (N,) float — rastermap position for each neuron (0 … N-1)
    n_triplets      : number of triplets to sample
    pos_window      : fraction of N defining positive neighbourhood
    neg_window      : fraction of N defining negative distance threshold
    positions       : (N, 3) xyz — required if hard_neg_radius is set
    hard_neg_radius : spatial radius for hard negative mining (µm)
    seed            : random seed
    """
    rng = np.random.default_rng(seed)
    N   = len(rmap_order)
    pos_thresh = pos_window * N
    neg_thresh = neg_window * N

    anchor_list, pos_list, neg_list = [], [], []
    anchor_order = rng.permutation(N)

    for a in anchor_order:
        order_dists = np.abs(rmap_order - rmap_order[a])

        pos_cands = np.where(order_dists < pos_thresh)[0]
        pos_cands = pos_cands[pos_cands != a]
        if len(pos_cands) == 0:
            continue

        if hard_neg_radius is not None and positions is not None:
            spatial_dists = np.linalg.norm(positions - positions[a], axis=1)
            nearby        = np.where(spatial_dists < hard_neg_radius)[0]
            nearby        = nearby[nearby != a]
            hard_neg_cands = nearby[order_dists[nearby] > neg_thresh]
            neg_cands = hard_neg_cands if len(hard_neg_cands) > 0 else \
                        np.where(order_dists > neg_thresh)[0]
        else:
            neg_cands = np.where(order_dists > neg_thresh)[0]

        neg_cands = neg_cands[neg_cands != a]
        if len(neg_cands) == 0:
            continue

        anchor_list.append(a)
        pos_list.append(int(rng.choice(pos_cands)))
        neg_list.append(int(rng.choice(neg_cands)))

        if len(anchor_list) >= n_triplets:
            break

    return (np.array(anchor_list, dtype=np.int64),
            np.array(pos_list,    dtype=np.int64),
            np.array(neg_list,    dtype=np.int64))


def triplet_loss(
    embeddings: torch.Tensor,
    anchors:    torch.Tensor,
    positives:  torch.Tensor,
    negatives:  torch.Tensor,
    margin:     float = 0.3,
) -> torch.Tensor:
    """
    Cosine-similarity triplet loss.

    Minimises  mean( max(0, margin - sim(a,p) + sim(a,n)) )

    Parameters
    ----------
    embeddings : (N, d)
    anchors    : (T,) long
    positives  : (T,) long
    negatives  : (T,) long
    margin     : separation margin in cosine similarity space
    """
    e = F.normalize(embeddings, dim=1)
    sim_pos = (e[anchors] * e[positives]).sum(dim=1)   # (T,)
    sim_neg = (e[anchors] * e[negatives]).sum(dim=1)   # (T,)
    loss = F.relu(margin - sim_pos + sim_neg)
    return loss.mean()


# ---------------------------------------------------------------------------
# MLP position embedding (xy only)
# ---------------------------------------------------------------------------

class MLPPositionEmbedding(nn.Module):
    """
    Map neuron positions to embeddings via a multi-layer MLP.

    Input coordinates are z-scored using the training-set mean/std
    (stored as buffers so they are saved with the model state).

    Parameters
    ----------
    in_dim    : 2 for xy-only, 3 for xyz
    embed_dim : output embedding dimension
    hidden    : width of hidden layers
    n_layers  : total number of linear layers (including output)
    """

    def __init__(self, in_dim: int, embed_dim: int,
                 hidden: int = 256, n_layers: int = 3):
        super().__init__()
        self.in_dim = in_dim
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers.append(nn.Linear(d, embed_dim))
        self.net = nn.Sequential(*layers)

        # Input normalisation buffers (set by init_from_positions)
        self.register_buffer('pos_mean', torch.zeros(in_dim))
        self.register_buffer('pos_std',  torch.ones(in_dim))

    def init_from_positions(self, positions: np.ndarray):
        """Set normalisation stats from training positions."""
        pos = positions[:, :self.in_dim].astype(np.float32)
        with torch.no_grad():
            self.pos_mean.copy_(torch.tensor(pos.mean(0)))
            self.pos_std.copy_(torch.tensor(pos.std(0).clip(1e-3)))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        positions : (N, D)  D >= in_dim

        Returns
        -------
        embeddings : (N, embed_dim)
        """
        x = positions[:, :self.in_dim]
        x = (x - self.pos_mean) / self.pos_std
        return self.net(x)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_position_embedding(
    positions:    np.ndarray,
    encoder_type: str  = 'cluster',
    embed_dim:    int  = 96,
    # cluster options
    n_clusters:   int  = 32,
    multiscale:   bool = True,
    top_k_assign: Optional[int] = None,
    # mlp options
    mlp_hidden:   int  = 256,
    mlp_layers:   int  = 3,
    mlp_use_xy:   bool = True,
) -> nn.Module:
    """
    Build and initialise a position embedding model.

    encoder_type : 'cluster' — multi-scale RBF cluster encoder (original)
                   'mlp'     — MLP encoder on xy (or xyz) coordinates
    """
    if encoder_type == 'mlp':
        in_dim = 2 if mlp_use_xy else 3
        model  = MLPPositionEmbedding(in_dim=in_dim, embed_dim=embed_dim,
                                      hidden=mlp_hidden, n_layers=mlp_layers)
        model.init_from_positions(positions)
        return model

    # Default: cluster
    return build_cluster_embedding(positions, embed_dim=embed_dim,
                                   n_clusters=n_clusters, multiscale=multiscale,
                                   top_k_assign=top_k_assign)


def build_cluster_embedding(
    positions:    np.ndarray,
    embed_dim:    int  = 64,
    n_clusters:   int  = 64,
    multiscale:   bool = False,
    top_k_assign: Optional[int] = None,
) -> nn.Module:
    """
    Build and initialise a cluster position embedding from neuron positions.

    Parameters
    ----------
    positions : (N, 3) numpy array of neuron xyz coordinates
    embed_dim : embedding dimension
    n_clusters: number of cluster centers (single-scale)
    multiscale: if True, use 3-scale version with σ at 25th/50th/75th percentile
                of pairwise distances

    Returns
    -------
    Initialised nn.Module ready for training
    """
    pos_t = torch.tensor(positions, dtype=torch.float32)

    # Estimate characteristic length scales from data
    sample = positions[np.random.choice(len(positions), min(500, len(positions)), replace=False)]
    dists = np.linalg.norm(sample[:, None] - sample[None], axis=-1)
    dists_flat = dists[np.triu_indices(len(sample), k=1)]
    s_fine, s_mid, s_coarse = (
        float(np.percentile(dists_flat, 10)),
        float(np.percentile(dists_flat, 30)),
        float(np.percentile(dists_flat, 60)),
    )

    if multiscale:
        K = max(8, n_clusters // 3)
        model = MultiScaleClusterEmbedding(
            n_clusters_per_scale = [K,     K * 2,  K * 4],
            embed_dim_per_scale  = [embed_dim // 3,
                                    embed_dim // 3,
                                    embed_dim - 2 * (embed_dim // 3)],
            sigmas               = [s_fine, s_mid, s_coarse],
            learn_sigma          = True,
            out_dim              = embed_dim,
            top_k_assign         = top_k_assign,
        )
    else:
        model = ClusterPositionEmbedding(
            n_clusters   = n_clusters,
            embed_dim    = embed_dim,
            sigma_init   = s_mid,
            learn_sigma  = True,
            out_mlp      = True,
            top_k_assign = top_k_assign,
        )

    model.init_centers_from_positions(positions)
    return model
