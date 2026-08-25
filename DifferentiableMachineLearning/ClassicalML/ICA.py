"""Independent Component Analysis re-implemented as a
``torch.nn.Module``.

Uses the classical FastICA fixed-point algorithm with PCA whitening:
the layer holds a ``[n_components, dim]`` demixing matrix ``W`` that
maps observed signals ``X ∈ R^{B × dim}`` to estimated sources
``S = X W^T``. ``fit`` performs centring, PCA whitening, and then
symmetric (parallel) FastICA iterations. Reconstruction is possible
via the pseudo-inverse of ``W``.
"""

from typing import Optional

import torch

from .utils import _to_2d


def _sym_decorrelate(W: torch.Tensor) -> torch.Tensor:
    """Symmetric decorrelation: return ``(W W^T)^{-1/2} W``.

    ``W W^T`` is symmetric positive semi-definite, so we use the
    eigendecomposition instead of SVD.
    """
    M = W @ W.T
    eigvals, eigvecs = torch.linalg.eigh(M)                        # ascending
    inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals))
    return eigvecs @ inv_sqrt @ eigvecs.T @ W


class ICA(torch.nn.Module):
    """
    Analogue:
        sklearn.decomposition.FastICA (Hyvärinen 1999)
    FastICA source separation layer (symmetric / parallel).

    Parameters
    ----------
    n_components : int
        Number of independent sources to recover. Must be ``<= dim``.
    dim : int
        Observed signal dimension.
    nonlinearity : {"tanh", "exp", "cube"}
        Non-quadratic contrast function used by FastICA. ``tanh``
        is the default and works well for super-Gaussian sources.
    max_iter : int
        Maximum FastICA iterations.
    tol : float
        Convergence tolerance on the unmixing matrix.
    """

    _NL = {
        "tanh": (torch.tanh, lambda x: 1.0 - torch.tanh(x) ** 2),
        "exp": (torch.exp, lambda x: torch.exp(-x ** 2 / 2.0)),
        "cube": (lambda x: x ** 3, lambda x: 3.0 * x ** 2),
    }

    def __init__(
        self,
        n_components: int,
        dim: int,
        nonlinearity: str = "tanh",
        max_iter: int = 200,
        tol: float = 1e-4,
    ) -> None:
        super().__init__()
        if nonlinearity not in self._NL:
            raise ValueError(
                f"Unknown nonlinearity {nonlinearity!r}; pick from {list(self._NL)}"
            )
        if not (0 < n_components <= dim):
            raise ValueError(
                f"0 < n_components={n_components} <= dim={dim} required"
            )
        self.n_components = n_components
        self.dim = dim
        self.nonlinearity = nonlinearity
        self.max_iter = max_iter
        self.tol = tol

        weight = torch.empty(n_components, dim)
        torch.nn.init.uniform_(weight, -0.5, 0.5)
        self.W = torch.nn.Parameter(weight)
        self.register_buffer(
            "mean", torch.zeros(dim, dtype=torch.float32)
        )
        self.register_buffer(
            "whitening", torch.eye(dim, dtype=torch.float32)
        )
        self._is_fitted = False

    # ----------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "ICA":
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        self.mean.data.copy_(torch.mean(x, dim=0))
        xc = x - self.mean
        n = xc.shape[0]

        # PCA whitening.
        cov = (xc.T @ xc) / n
        eigvals, eigvecs = torch.linalg.eigh(cov)                   # ascending
        keep = eigvals > 1e-10
        eigvals = eigvals[keep][: self.n_components]
        eigvecs = eigvecs[:, keep][:, : self.n_components]
        K = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals))         # [dim, k]
        Xw = xc @ K                                                # [n, k]

        # Symmetric FastICA. Start from a random rotation.
        g, _gprime = self._NL[self.nonlinearity]
        W = torch.linalg.qr(torch.randn(self.n_components, self.n_components))[0]
        for it in range(self.max_iter):
            u = Xw @ W.T                                          # [n, k]
            W_new = (Xw.T @ g(u)).T                               # [k, k]
            W_new = _sym_decorrelate(W_new)
            # angular change between W_new and W
            cos = torch.min(
                torch.abs(torch.sum(W * W_new, dim=1))
            )                                                      # smallest |row cosine|
            W = W_new
            if float(1.0 - cos) < self.tol:
                break

        # Combined demixing: sources = Xc @ (K W)^T
        self.W.data.copy_(W @ K.T)                                 # [k, dim]
        self.whitening.data.copy_(K)
        self._is_fitted = True
        return self

    # ----------------------------------------------------------------- ops
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Return the estimated sources ``[batch, n_components]``."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        if not self._is_fitted:
            raise RuntimeError("ICA is not fitted; call fit() first.")
        return (x - self.mean) @ self.W.T

    def inverse_transform(self, sources: torch.Tensor) -> torch.Tensor:
        """Pseudo-inverse reconstruction of observed signals from sources."""
        sources = _to_2d(sources)
        if sources.shape[1] != self.n_components:
            raise ValueError(
                f"Expected {self.n_components} sources, got {sources.shape[1]}"
            )
        if not self._is_fitted:
            raise RuntimeError("ICA is not fitted; call fit() first.")
        W_pinv = torch.linalg.pinv(self.W)                         # [dim, k]
        return sources @ W_pinv.T + self.mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    def extra_repr(self) -> str:
        return (
            f"n_components={self.n_components}, dim={self.dim}, "
            f"nonlinearity={self.nonlinearity!r}, fitted={self._is_fitted}"
        )