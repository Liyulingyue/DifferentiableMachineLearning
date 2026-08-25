"""Linear Discriminant Analysis (Fisher, multi-class) re-implemented
as a ``torch.nn.Module``.

The projection matrix is stored as a learnable ``[n_components, dim]``
parameter. ``fit`` solves the closed-form generalised eigen problem

    S_b v = lambda S_w v

on the supplied labelled data, picks the top ``n_components``
eigenvectors, and stores them. The forward pass also evaluates the
classical Gaussian discriminant log-likelihoods, returning
``[batch, n_classes]`` logits that are differentiable (and can act
as a classifier head).
"""

import torch

from .utils import _to_2d


class LDA(torch.nn.Module):
    """
    Analogue:
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis (Fisher 1936)
    Multi-class Fisher LDA.

    Parameters
    ----------
    n_components : int
        Target dimensionality. Must satisfy ``1 <= n_components <= min(dim, n_classes-1)``.
    dim : int
        Input feature dimension.
    n_classes : int
        Number of classes; declared up front so the layer can allocate
        class statistics.
    """

    def __init__(self, n_components: int, dim: int, n_classes: int) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError(f"n_components must be > 0, got {n_components}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        max_dim = min(dim, n_classes - 1)
        if n_components > max_dim:
            raise ValueError(
                f"n_components={n_components} exceeds the rank bound "
                f"min(dim, n_classes-1) = {max_dim}"
            )

        self.n_components = n_components
        self.dim = dim
        self.n_classes = n_classes

        weight = torch.empty(n_components, dim)
        torch.nn.init.uniform_(weight, -0.5, 0.5)
        self.components = torch.nn.Parameter(weight)
        # Fitted per-class statistics (set by fit()).
        self.register_buffer(
            "class_means", torch.zeros(n_classes, dim, dtype=torch.float32)
        )
        self.register_buffer(
            "class_inv_cov", torch.tile(
                torch.eye(dim, dtype=torch.float32).unsqueeze(0),
                (n_classes, 1, 1),
            )
        )
        self.register_buffer(
            "class_log_prior", torch.zeros(n_classes, dtype=torch.float32)
        )
        self.register_buffer(
            "global_mean", torch.zeros(dim, dtype=torch.float32)
        )
        self._is_fitted = False

    # ----------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "LDA":
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        y = y.long()
        if y.ndim == 2:
            y = y.squeeze(-1)
        n = x.shape[0]
        mu = torch.mean(x, dim=0)
        Sw = torch.zeros(self.dim, self.dim, dtype=x.dtype)
        Sb = torch.zeros(self.dim, self.dim, dtype=x.dtype)
        class_means = []
        priors = []
        for c in range(self.n_classes):
            mask = (y == c)
            cnt = int(torch.sum(mask))
            if cnt == 0:
                raise ValueError(f"class {c} has no samples in training set")
            xc = x[mask]
            mc = torch.mean(xc, dim=0)
            Sc = (xc - mc).T @ (xc - mc)
            Sw += Sc
            Sb += cnt * (mc - mu).unsqueeze(-1) @ (mc - mu).unsqueeze(0)
            class_means.append(mc)
            priors.append(cnt / n)
        # Add a small ridge for numerical stability.
        Sw = Sw + 1e-6 * torch.eye(self.dim, dtype=Sw.dtype)

        # Generalised eigen problem  Sw^{-1} Sb v = lambda v.
        eigvals, eigvecs = torch.linalg.eigh(
            torch.linalg.solve(Sw, Sb)
        )  # ascending order
        idx = torch.argsort(-eigvals)[: self.n_components]
        basis = eigvecs[:, idx].T                                # [k, dim]
        self.components.data.copy_(basis)

        # Per-class Gaussian discriminant statistics.
        self.global_mean.data.copy_(mu)
        inv_covs = []
        for c in range(self.n_classes):
            xc = x[y == c]
            mc = torch.mean(xc, dim=0)
            d = xc - mc
            cov = (d.T @ d) / max(d.shape[0] - 1, 1) + 1e-6 * torch.eye(
                self.dim, dtype=xc.dtype
            )
            inv_covs.append(torch.linalg.inv(cov))
        self.class_means.data.copy_(torch.stack(class_means, dim=0))
        self.class_inv_cov.data.copy_(torch.stack(inv_covs, dim=0))
        self.class_log_prior.data.copy_(torch.log(torch.tensor(priors)))
        self._is_fitted = True
        return self

    # ----------------------------------------------------------------- ops
    def project(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        return x @ self.components.T

    def predict_log_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable Gaussian-discriminant log ``p(y|x)`` of shape
        ``[batch, n_classes]``."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        if not self._is_fitted:
            raise RuntimeError("LDA is not fitted; call fit() first.")
        diff = x.unsqueeze(1) - self.class_means.unsqueeze(0)    # [B, C, D]
        # Mahalanobis: (x - mu) Sigma^{-1} (x - mu)^T
        ic = self.class_inv_cov.unsqueeze(0)                     # [1, C, D, D]
        m = diff.unsqueeze(-1)                                   # [B, C, D, 1]
        quad = torch.linalg.solve(ic, m)                         # [B, C, D, 1]
        quad = (m * quad).sum(dim=[-1, -2])                      # [B, C]
        log_det = torch.log(
            torch.linalg.det(self.class_inv_cov).clamp(min=1e-30)
        )                                                        # [C]
        log_p = -0.5 * (quad + log_det.unsqueeze(0)) + self.class_log_prior.unsqueeze(0)
        return log_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Gaussian-discriminant logits of shape ``[batch, n_classes]``."""
        return self.predict_log_proba(x)

    def extra_repr(self) -> str:
        return (
            f"n_components={self.n_components}, dim={self.dim}, "
            f"n_classes={self.n_classes}, fitted={self._is_fitted}"
        )