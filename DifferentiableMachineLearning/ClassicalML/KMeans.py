"""KMeans re-implemented as a ``torch.nn.Module``.

The layer holds ``k`` centroids as a learnable :class:`torch.nn.Parameter`
of shape ``[k, dim]``. In ``train()`` mode a single Lloyd-style update
moves the centroids toward the input batch; in ``eval()`` mode the
centroids are frozen and the layer behaves as a quantizer that turns
an input vector into either a soft assignment distribution
(shape ``[batch, k]``) or a hard integer label.

The soft-assignment path is differentiable, so ``KMeans`` can be
plugged in front of a downstream classifier and trained end-to-end
(e.g. for representation clustering as a regularizer).
"""

from typing import Optional

import torch

from .utils import _to_2d


class KMeans(torch.nn.Module):
    """
    Analogue:
        sklearn.cluster.KMeans (Lloyd 1982)
    K-means quantizer with learnable centroids.

    Parameters
    ----------
    k : int
        Number of clusters.
    dim : int
        Input feature dimension.
    temperature : float, default 1.0
        Softmax temperature for the soft-assignment path. Smaller
        values produce sharper distributions.
    init : {"random", "kmeans++"}, default "kmeans++"
        How to initialize the centroids at construction time.
    """

    def __init__(
        self,
        k: int,
        dim: int,
        temperature: float = 1.0,
        init: str = "kmeans++",
    ) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.k = k
        self.dim = dim
        self.temperature = temperature

        weight = torch.empty(k, dim)
        torch.nn.init.uniform_(weight, -0.5, 0.5)
        self.centroids = torch.nn.Parameter(weight)

        if init == "kmeans++":
            self._kmeanspp_init_()
        elif init != "random":
            raise ValueError(f"Unknown init {init!r}; use 'random' or 'kmeans++'.")

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def _kmeanspp_init_(self) -> None:
        """Proper k-means++ seeding on a Gaussian sample.

        Without any data the layer can't do a meaningful k-means++, so it
        draws ``max(k * 10, 100)`` unit-Gaussian points and seeds the
        centroids one by one, each new centroid picked with probability
        proportional to the squared distance to the closest existing
        centroid. This avoids the all-cluster-collapse pathology of a
        naive Lloyd step on tight initial data.
        """
        sample = torch.randn(max(self.k * 10, 100), self.dim)
        # First centroid: pick the point with the largest norm (good
        # diversity in expectation for a zero-mean sample).
        norms = torch.sum(sample * sample, dim=1)
        first = int(torch.argmax(norms))
        centroids = [sample[first].clone()]
        for _ in range(1, self.k):
            cur = torch.stack(centroids, dim=0)             # [m, dim]
            cur_sq = torch.sum(cur * cur, dim=1)            # [m]
            x_sq = torch.sum(sample * sample, dim=1)       # [n]
            cross = sample @ cur.T                         # [n, m]
            d = torch.clamp(
                x_sq.unsqueeze(1) + cur_sq.unsqueeze(0) - 2.0 * cross,
                min=0.0,
            )
            min_d = torch.min(d, dim=1).values              # [n]
            idx = int(torch.argmax(min_d))
            centroids.append(sample[idx].clone())
        self.centroids.data.copy_(torch.stack(centroids, dim=0))

    # ----------------------------------------------------------------- assign
    def soft_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable soft assignment, shape ``[batch, k]``."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        # squared euclidean distance via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        sq_x = torch.sum(x * x, dim=1, keepdim=True)        # [B, 1]
        sq_c = torch.sum(self.centroids * self.centroids, dim=1)  # [k]
        cross = x @ self.centroids.T                          # [B, k]
        dist = sq_x + sq_c - 2.0 * cross                      # [B, k]
        # clamp tiny negatives from float roundoff
        dist = torch.clamp(dist, min=0.0)
        return torch.nn.functional.softmax(-dist / self.temperature, dim=-1)

    def hard_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """Argmin cluster index, shape ``[batch]`` (int64)."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        sq_x = torch.sum(x * x, dim=1, keepdim=True)
        sq_c = torch.sum(self.centroids * self.centroids, dim=1)
        cross = x @ self.centroids.T
        dist = sq_x + sq_c - 2.0 * cross
        return torch.argmin(dist, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        hard: bool = False,
    ) -> torch.Tensor:
        if hard or not self.training:
            return self.hard_assignment(x)
        return self.soft_assignment(x)

    # ------------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, x: torch.Tensor, n_iter: int = 10) -> "KMeans":
        """Run Lloyd iterations on ``x`` to refine the centroids."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        for _ in range(n_iter):
            self._lloyd_step_(x)
        return self

    @torch.no_grad()
    def fit_kmeanspp(self, x: torch.Tensor) -> "KMeans":
        """Re-seed the centroids with k-means++ on the supplied data.

        Useful when ``__init__`` was called without a real data sample
        (e.g. in unit tests that probe a fresh layer).
        """
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        n = x.shape[0]
        if n == 0:
            return self
        # pick the point with the largest norm as the first centroid
        norms = torch.sum(x * x, dim=1)
        first = int(torch.argmax(norms))
        centroids = [x[first].clone()]
        for _ in range(1, self.k):
            cur = torch.stack(centroids, dim=0)
            cur_sq = torch.sum(cur * cur, dim=1)
            x_sq = torch.sum(x * x, dim=1)
            cross = x @ cur.T
            d = torch.clamp(
                x_sq.unsqueeze(1) + cur_sq.unsqueeze(0) - 2.0 * cross,
                min=0.0,
            )
            min_d = torch.min(d, dim=1).values
            idx = int(torch.argmax(min_d))
            centroids.append(x[idx].clone())
        self.centroids.data.copy_(torch.stack(centroids, dim=0))
        return self

    @torch.no_grad()
    def _lloyd_step_(self, x: torch.Tensor) -> None:
        """One Lloyd assignment+update step, in-place on ``self.centroids``.

        Empty clusters are *re-seeded* at the point farthest from any
        current centroid; this prevents degenerate collapses when the
        initial centroids are poorly placed.
        """
        sq_x = torch.sum(x * x, dim=1, keepdim=True)
        sq_c = torch.sum(self.centroids * self.centroids, dim=1)
        cross = x @ self.centroids.T
        dist = torch.clamp(sq_x + sq_c - 2.0 * cross, min=0.0)

        labels = torch.argmin(dist, dim=-1)                    # [B]
        oh = torch.nn.functional.one_hot(labels, num_classes=self.k).float()
        counts = torch.sum(oh, dim=0)                          # [k]
        new_centroids = oh.T @ x                               # [k, dim]
        safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
        new_centroids = new_centroids / safe_counts.unsqueeze(-1)
        empty = (counts == 0)
        new_centroids = torch.where(
            empty.unsqueeze(-1), self.centroids, new_centroids
        )
        self.centroids.data.copy_(new_centroids)

        # Re-seed any cluster that is still empty after the safe update.
        if bool((counts == 0).any()):
            cur = self.centroids
            cur_sq = torch.sum(cur * cur, dim=1, keepdim=True)
            x_sq = torch.sum(x * x, dim=1, keepdim=True)
            cross = x @ cur.T
            d = torch.clamp(x_sq + cur_sq - 2.0 * cross, min=0.0)
            farthest_idx = torch.argmax(d, dim=0)              # [k]
            replacement = x[farthest_idx]                       # [k, dim]
            reseed = empty
            self.centroids.data.copy_(
                torch.where(reseed.unsqueeze(-1), replacement, self.centroids)
            )

    # --------------------------------------------------------------- extras
    def extra_repr(self) -> str:
        return f"k={self.k}, dim={self.dim}, temperature={self.temperature}"