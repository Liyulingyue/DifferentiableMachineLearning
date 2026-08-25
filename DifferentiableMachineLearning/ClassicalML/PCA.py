"""Principal Component Analysis re-implemented as a ``torch.nn.Module``.

The principal axes are stored as a learnable :class:`torch.nn.Parameter`
of shape ``[n_components, dim]``. A Stiefel-manifold projection
(Gram-Schmidt via QR) keeps the rows orthonormal both at construction
and after every gradient step, so the layer behaves like a learned
orthogonal bottleneck. ``fit`` solves the closed-form PCA via SVD on
the centred data.
"""

from typing import Optional

import torch

from .utils import _to_2d


def _orthonormalise(matrix: torch.Tensor) -> torch.Tensor:
    """QR-based orthonormalisation; the first ``min(m, n)`` rows are
    guaranteed orthogonal and unit-norm regardless of input rank.
    """
    q, r = torch.linalg.qr(matrix.T, mode="reduced")  # q: [dim, k]
    sign = torch.sign(torch.diagonal(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(0)
    return q.T  # [k, dim]


class PCA(torch.nn.Module):
    """
    Analogue:
        sklearn.decomposition.PCA (Pearson 1901; Hotelling 1933)
    PCA bottleneck layer with a learnable orthogonal basis.

    Parameters
    ----------
    n_components : int
        Target dimensionality of the projected space.
    dim : int
        Input feature dimension.
    """

    def __init__(self, n_components: int, dim: int) -> None:
        super().__init__()
        if not (0 < n_components <= dim):
            raise ValueError(
                f"0 < n_components={n_components} <= dim={dim} required"
            )

        self.n_components = n_components
        self.dim = dim

        weight = torch.empty(n_components, dim)
        torch.nn.init.uniform_(weight, -0.5, 0.5)
        self.components = torch.nn.Parameter(weight)
        with torch.no_grad():
            self.components.data.copy_(_orthonormalise(weight))

        # Fitted mean (set by fit; subtracts before projection in eval
        # mode). Stored as a buffer; defaults to zero so the layer
        # works without a fit() call.
        self.register_buffer(
            "mean", torch.zeros(dim, dtype=torch.float32)
        )
        self._is_fitted = False

    # ----------------------------------------------------------------- ops
    def _basis(self) -> torch.Tensor:
        """Re-orthonormalised basis (in case the parameter drifted)."""
        return _orthonormalise(self.components)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` onto the principal subspace, ``[batch, n_components]``."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        centred = x - self.mean
        return centred @ self._basis().T

    def reconstruct(self, coords: torch.Tensor) -> torch.Tensor:
        """Reconstruct from low-dimensional coordinates, ``[batch, dim]``."""
        coords = _to_2d(coords)
        if coords.shape[1] != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} coordinates, got {coords.shape[1]}"
            )
        return coords @ self._basis() + self.mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project-then-reconstruct (autoencoder form)."""
        return self.reconstruct(self.project(x))

    # ----------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "PCA":
        """Compute the closed-form PCA on ``x`` (centred, SVD-based)."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        self.mean.data.copy_(torch.mean(x, dim=0))
        xc = x - self.mean
        # Economy SVD; the right singular vectors of X are the PCs.
        _, _, vt = torch.linalg.svd(xc, full_matrices=False)        # [k, dim]
        k = min(self.n_components, vt.shape[0])
        basis = vt[:k]                                              # [k, dim]
        # If n_components < dim we may need to pad with random
        # orthonormal rows so the layer remains a square linear map.
        if k < self.n_components:
            extra = torch.randn(self.n_components - k, self.dim)
            basis = torch.cat([basis, _orthonormalise(extra)], dim=0)
        self.components.data.copy_(basis)
        self._is_fitted = True
        return self

    def extra_repr(self) -> str:
        return f"n_components={self.n_components}, dim={self.dim}, fitted={self._is_fitted}"