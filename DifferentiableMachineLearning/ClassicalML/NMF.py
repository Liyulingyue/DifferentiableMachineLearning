"""Non-negative Matrix Factorisation (Lee & Seung 2001) re-implemented
as a ``torch.nn.Module``.

The model factorises a non-negative data matrix ``X ∈ R^{N×D}_{≥0}``
into ``X ≈ W H`` with ``W ∈ R^{K×D}_{≥0}`` (the "dictionary" or
"topics") and ``H ∈ R^{N×K}_{≥0}`` (the "codes" or "activations").
Both matrices are stored as non-negative parameters; ``fit``
alternates Lee-Seung multiplicative updates that monotonically
decrease the Frobenius reconstruction error, and ``transform``
updates only ``H`` while keeping ``W`` fixed.
"""

import torch

from .utils import _to_2d


class NMF(torch.nn.Module):
    """
    Analogue:
        sklearn.decomposition.NMF (Lee & Seung 2001 'Algorithms for Non-negative Matrix Factorization')
    Non-negative matrix factorisation.

    Parameters
    ----------
    n_components : int
        Rank of the factorisation.
    n_features : int
        Number of features (columns) of the input data.
    init : {"random", "nndsvd"}
        Initialisation scheme. ``random`` uses ``Uniform(0, 1)``;
        ``nndsvd`` uses a deterministic NNDSVD-style seeding.
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        init: str = "random",
    ) -> None:
        super().__init__()
        if n_components <= 0 or n_features <= 0:
            raise ValueError("n_components and n_features must be > 0")
        if init not in {"random", "nndsvd"}:
            raise ValueError(f"Unknown init {init!r}")
        self.n_components = n_components
        self.n_features = n_features
        self.init = init
        W = torch.abs(torch.randn(n_components, n_features, dtype=torch.float32)) + 0.1
        self.W = torch.nn.Parameter(W)
        self.register_buffer(
            "H", torch.zeros(0, n_components, dtype=torch.float32)
        )
        self._is_fitted = False

    def _nndsvd_init_(self, x: torch.Tensor) -> None:
        """Quick deterministic init: SVD on the data, then abs-then-rescale."""
        u, s, vt = torch.linalg.svd(x, full_matrices=False)
        # Take the top n_components singular triples.
        W = torch.abs(vt[: self.n_components]) + 0.1
        H = torch.abs(u[:, : self.n_components] @ torch.diag(s[: self.n_components])) + 0.1
        with torch.no_grad():
            self.W.data.copy_(W)
        self._set_H(H)

    def _set_H(self, H: torch.Tensor) -> None:
        # Replace the buffer with the right shape.
        self.H = H.clone()

    # ----------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, x: torch.Tensor, n_iter: int = 200) -> "NMF":
        """Alternating multiplicative updates for ``W`` and ``H``."""
        x = _to_2d(x)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, got {x.shape[1]}"
            )
        if (x < 0).any():
            raise ValueError("NMF requires non-negative input data")
        N = x.shape[0]
        if self.init == "nndsvd":
            self._nndsvd_init_(x)
        else:
            H = torch.abs(torch.randn(N, self.n_components, dtype=torch.float32)) + 0.1
            self._set_H(H)
        eps = 1e-12
        for _ in range(n_iter):
            # H: [N, K], W: [K, D]. Lee-Seung multiplicative updates
            # preserve non-negativity and monotonically decrease
            # ||X - W^T H||_F^2 (here we treat X as [N, D] and
            # reconstruction as H W = [N, K] @ [K, D] = [N, D]).
            # H <- H * (X W) / (H (W W^T))
            numerator = x @ self.W.T                              # [N, K]
            denominator = self.H @ (self.W @ self.W.T) + eps     # [N, K]
            H_new = self.H * numerator / denominator
            self._set_H(H_new)
            # W <- W * (H^T X) / (H^T H W)
            numerator = self.H.T @ x                              # [K, D]
            denominator = (self.H.T @ self.H) @ self.W + eps     # [K, D]
            W_new = self.W * numerator / denominator
            self.W.data.copy_(W_new)
        self._is_fitted = True
        return self

    @torch.no_grad()
    def transform(self, x: torch.Tensor, n_iter: int = 100) -> torch.Tensor:
        """Update only ``H`` for new data with fixed ``W``."""
        x = _to_2d(x)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, got {x.shape[1]}"
            )
        if (x < 0).any():
            raise ValueError("NMF requires non-negative input data")
        H = torch.abs(torch.randn(x.shape[0], self.n_components, dtype=torch.float32)) + 0.1
        eps = 1e-12
        for _ in range(n_iter):
            numerator = x @ self.W.T                              # [N, K]
            denominator = H @ (self.W @ self.W.T) + eps           # [N, K]
            H = H * numerator / denominator
        return H

    def reconstruct(self, H: torch.Tensor) -> torch.Tensor:
        return H @ self.W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct: encode then decode."""
        H = self.transform(x)
        return H @ self.W

    def extra_repr(self) -> str:
        return f"n_components={self.n_components}, n_features={self.n_features}, init={self.init!r}"