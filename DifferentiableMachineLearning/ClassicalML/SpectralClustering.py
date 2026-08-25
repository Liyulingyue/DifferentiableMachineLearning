"""Spectral Clustering (Shi & Malik 2000 Normalised Cuts; Ng-Jordan-
Weiss 2001) re-implemented as a ``torch.nn.Module``.

Analogue:
    sklearn.cluster.SpectralClustering (which is essentially the
    Ng-Jordan-Weiss variant).

Workflow:
    1. Build an N x N similarity matrix ``W`` from the input
       (RBF kernel, k-NN graph, or pre-computed).
    2. Compute the symmetric normalised graph Laplacian
       ``L_sym = I - D^{-1/2} W D^{-1/2}``.
    3. Take the eigenvectors corresponding to the ``n_clusters``
       smallest eigenvalues; normalise each row to unit length.
    4. Run KMeans in the spectral embedding.

The whole pipeline is non-differentiable (eigendecomposition +
KMeans) and runs under ``@torch.no_grad``; the layer is exposed
as a ``torch.nn.Module`` so the user can call ``fit_predict`` in
the same idiom as the rest of the kit.
"""

import torch

from .KMeans import KMeans
from .utils import _to_2d


def _rbf_affinity(x: torch.Tensor, gamma: float) -> torch.Tensor:
    sq_x = torch.sum(x * x, dim=1, keepdim=True)
    sq_y = torch.sum(x * x, dim=1)
    cross = x @ x.T
    d2 = torch.clamp(sq_x + sq_y.unsqueeze(0) - 2.0 * cross, min=0.0)
    return torch.exp(-gamma * d2)


def _knn_affinity(x: torch.Tensor, k: int) -> torch.Tensor:
    sq_x = torch.sum(x * x, dim=1, keepdim=True)
    sq_y = torch.sum(x * x, dim=1)
    cross = x @ x.T
    d2 = sq_x + sq_y.unsqueeze(0) - 2.0 * cross
    d2 = torch.sqrt(torch.clamp(d2, min=0.0))
    _, idx = torch.topk(d2, k=k + 1, largest=False)
    W = torch.zeros_like(d2)
    for i in range(d2.shape[0]):
        W[i, idx[i, 1:]] = 1.0
    return (W + W.T) * 0.5


class SpectralClustering(torch.nn.Module):
    """Spectral clustering with Ng-Jordan-Weiss normalisation.

    Analogue:
        sklearn.cluster.SpectralClustering (Ng-Jordan-Weiss variant).

    Parameters
    ----------
    n_clusters : int
    gamma : float
        RBF kernel bandwidth (``exp(-gamma ||x - y||^2)``).
    affinity : {"rbf", "knn", "precomputed"}
        ``"rbf"`` uses RBF kernel; ``"knn"`` uses symmetric k-nearest-
        neighbour graph; ``"precomputed"`` treats the input as an
        N x N similarity matrix directly.
    n_neighbors : int
        Number of neighbours for ``affinity="knn"``.
    n_init : int
        Number of KMeans restarts inside the spectral embedding.
    """

    def __init__(
        self,
        n_clusters: int,
        gamma: float = 1.0,
        affinity: str = "rbf",
        n_neighbors: int = 10,
        n_init: int = 10,
    ) -> None:
        super().__init__()
        if affinity not in {"rbf", "knn", "precomputed"}:
            raise ValueError(f"Unknown affinity {affinity!r}")
        if n_clusters <= 1:
            raise ValueError("n_clusters must be > 1")
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.n_init = n_init
        self.register_buffer(
            "labels_", torch.zeros(0, dtype=torch.int64)
        )

    @torch.no_grad()
    def _spectral_embedding(self, W: torch.Tensor) -> torch.Tensor:
        """Compute the Ng-Jordan-Weiss normalised spectral embedding.

        Returns the matrix ``U`` of shape ``[N, n_clusters]`` whose
        rows are the normalised eigenvectors of the smallest
        ``n_clusters`` non-zero eigenvalues of ``L_sym``.
        """
        deg = W.sum(dim=1)
        deg_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(deg, min=1e-12))
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L_sym = torch.eye(W.shape[0], dtype=W.dtype) - D_inv_sqrt @ W @ D_inv_sqrt
        eigvals, eigvecs = torch.linalg.eigh(L_sym)              # ascending
        # Take the bottom ``n_clusters`` eigenvectors (skip the first
        # one whose eigenvalue is ~0, the connected-component axis).
        start = 1
        U = eigvecs[:, start: start + self.n_clusters]
        # Row-normalise
        U_norm = U / torch.clamp(
            torch.norm(U, dim=1, keepdim=True), min=1e-12
        )
        return U_norm

    @torch.no_grad()
    def fit_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run spectral clustering and return integer labels."""
        if self.affinity == "rbf":
            x = _to_2d(x)
            W = _rbf_affinity(x, self.gamma)
        elif self.affinity == "knn":
            x = _to_2d(x)
            W = _knn_affinity(x, self.n_neighbors)
        else:
            W = _to_2d(x)
            if W.shape[0] != W.shape[1]:
                raise ValueError("Precomputed affinity must be square")
        U = self._spectral_embedding(W)
        # KMeans in the spectral embedding; multiple restarts.
        best_loss = float("inf")
        best_labels = None
        for _ in range(self.n_init):
            km = KMeans(k=self.n_clusters, dim=U.shape[1], temperature=0.1)
            km.fit_kmeanspp(U)
            km.fit(U, n_iter=30)
            # KMeans has no public loss, but we can use the per-iteration
            # Lloyd step and pick the run that ends with the lowest
            # in-cluster sum-of-squares.
            sse = float(torch.sum(
                torch.min(torch.sum(
                    (U.unsqueeze(1) - km.centroids.unsqueeze(0)) ** 2,
                    dim=-1,
                ), dim=-1).values
            ).item())
            if sse < best_loss:
                best_loss = sse
                best_labels = km(U, hard=True)
        self.labels_ = best_labels.long()
        return self.labels_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fit_predict(x)

    def extra_repr(self) -> str:
        return (
            f"n_clusters={self.n_clusters}, gamma={self.gamma}, "
            f"affinity={self.affinity!r}, n_neighbors={self.n_neighbors}"
        )