"""Gaussian Mixture Model re-implemented as a ``torch.nn.Module``.

The model stores, per component ``k``:

* a learnable ``means``        parameter of shape ``[k, dim]``
* a learnable ``log_vars``     parameter of shape ``[k, dim]`` (diagonal
  log-variances; the ``spherical`` shape is treated as a length-1
  tensor and broadcast)
* a learnable ``log_weights``  parameter of shape ``[k]`` (logits; the
  softmax gives the mixture coefficients)

The forward pass returns the soft responsibilities
``gamma_{nk} = p(z=k | x_n)`` of shape ``[batch, k]``, fully
differentiable, so the layer can act as a K-dimensional feature
extractor in a downstream classifier.
"""

from typing import Optional

import torch

from .utils import _to_2d


_VALID_COV = {"diag", "spherical", "full"}


def _log_gaussian_diag(
    x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """Per-component log-density, ``[batch, k]``."""
    # x: [B, D], mean: [K, D], log_var: [K, D]
    diff = x.unsqueeze(1) - mean.unsqueeze(0)                  # [B, K, D]
    inv_var = torch.exp(-log_var)                              # [K, D]
    quad = torch.sum(diff * diff * inv_var.unsqueeze(0), dim=-1)  # [B, K]
    log_norm = torch.sum(log_var, dim=-1) + mean.shape[-1] * torch.log(
        torch.tensor(2 * torch.pi, dtype=mean.dtype)
    )                                                          # [K]
    return -0.5 * (quad + log_norm.unsqueeze(0))


def _log_gaussian_spherical(
    x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    diff = x.unsqueeze(1) - mean.unsqueeze(0)                  # [B, K, D]
    sq = torch.sum(diff * diff, dim=-1)                       # [B, K]
    D = mean.shape[-1]
    return -0.5 * (sq * torch.exp(-log_var).squeeze(-1).unsqueeze(0)
                   + D * log_var.squeeze(-1).unsqueeze(0)
                   + D * torch.log(torch.tensor(2 * torch.pi, dtype=mean.dtype)))


def _log_gaussian_full(
    x: torch.Tensor, mean: torch.Tensor, L: torch.Tensor, log_det: torch.Tensor
) -> torch.Tensor:
    """``L`` is the lower-triangular Cholesky factor, ``log_det`` its
    log-determinant of the corresponding covariance."""
    diff = x.unsqueeze(1) - mean.unsqueeze(0)                  # [B, K, D]
    B, K, D = diff.shape
    # Solve L_k z_bk = diff_bk per component k
    diff_flat = diff.reshape(B * K, D, 1)                       # [B*K, D, 1]
    L_rep = L.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * K, D, D)
    z_flat = torch.linalg.solve_triangular(
        L_rep, diff_flat, upper=False
    )                                                          # [B*K, D, 1]
    z = z_flat.reshape(B, K, D)
    quad = torch.sum(z * z, dim=-1)                             # [B, K]
    return -0.5 * (quad + log_det.unsqueeze(0) + mean.shape[-1] * torch.log(
        torch.tensor(2 * torch.pi, dtype=mean.dtype)
    ))


class GMM(torch.nn.Module):
    """
    Analogue:
        sklearn.mixture.GaussianMixture (Dempster et al. 1977 EM)
    Gaussian Mixture with learnable parameters and EM fitting.

    Parameters
    ----------
    k : int
        Number of components.
    dim : int
        Feature dimension.
    covariance_type : {"diag", "spherical", "full"}
    reg : float
        Diagonal regularisation added to covariances during the
        forward pass (for numerical stability).
    """

    def __init__(
        self,
        k: int,
        dim: int,
        covariance_type: str = "diag",
        reg: float = 1e-6,
    ) -> None:
        super().__init__()
        if k <= 0 or dim <= 0:
            raise ValueError("k and dim must be > 0")
        if covariance_type not in _VALID_COV:
            raise ValueError(
                f"Unknown covariance_type {covariance_type!r}; "
                f"pick from {_VALID_COV}"
            )

        self.k = k
        self.dim = dim
        self.covariance_type = covariance_type
        self.reg = reg

        self.means = torch.nn.Parameter(
            torch.empty(k, dim).uniform_(-0.5, 0.5)
        )
        if covariance_type == "spherical":
            self.log_vars = torch.nn.Parameter(
                torch.zeros(k, 1, dtype=torch.float32)
            )
        elif covariance_type == "diag":
            self.log_vars = torch.nn.Parameter(
                torch.zeros(k, dim, dtype=torch.float32)
            )
        else:  # full
            self.log_vars = torch.nn.Parameter(
                torch.zeros(k, dim, dim, dtype=torch.float32)
            )
        self.log_weights = torch.nn.Parameter(
            torch.zeros(k, dtype=torch.float32)
        )

    # ------------------------------------------------------- probabilities
    def _log_resp(self, x: torch.Tensor) -> torch.Tensor:
        """Log responsibilities ``[batch, k]`` (differentiable)."""
        if self.covariance_type == "full":
            L = torch.linalg.cholesky(
                torch.exp(self.log_vars) + self.reg * torch.eye(
                    self.dim, dtype=self.log_vars.dtype
                ).unsqueeze(0)
            )
            log_det = 2.0 * torch.sum(
                torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1
            )
            log_p = _log_gaussian_full(x, self.means, L, log_det)
        elif self.covariance_type == "spherical":
            log_p = _log_gaussian_spherical(x, self.means, self.log_vars)
        else:
            log_p = _log_gaussian_diag(x, self.means, self.log_vars)

        log_w = torch.nn.functional.log_softmax(self.log_weights, dim=-1)
        log_p = log_p + log_w.unsqueeze(0)
        return torch.nn.functional.log_softmax(log_p, dim=-1)

    def responsibilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._log_resp(x))

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample log marginal likelihood ``[batch]``."""
        return torch.logsumexp(self._log_resp(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft responsibilities ``[batch, k]`` (differentiable)."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        return self.responsibilities(x)

    # ------------------------------------------------------------------- EM
    @torch.no_grad()
    def fit_em(self, x: torch.Tensor, n_iter: int = 50) -> "GMM":
        """Closed-form maximum-likelihood EM updates."""
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        for _ in range(n_iter):
            self._em_step_(x)
        return self

    @torch.no_grad()
    def _em_step_(self, x: torch.Tensor) -> None:
        # E-step
        log_r = self._log_resp(x)                               # [B, K]
        r = torch.exp(log_r)                                     # [B, K]
        N = r.shape[0]

        N_k = torch.sum(r, dim=0)                                # [K]
        weights = N_k / N

        # M-step
        new_means = (r.T @ x) / N_k.unsqueeze(-1)                # [K, D]
        diff = x.unsqueeze(1) - new_means.unsqueeze(0)           # [B, K, D]

        if self.covariance_type == "diag":
            new_var = torch.sum(
                r.unsqueeze(-1) * diff * diff, dim=0
            ) / N_k.unsqueeze(-1) + self.reg                    # [K, D]
            self.log_vars.data.copy_(torch.log(new_var))
        elif self.covariance_type == "spherical":
            sq = torch.sum(diff * diff, dim=-1)                  # [B, K]
            new_var = torch.sum(r * sq, dim=0) / (N_k * self.dim) + self.reg
            self.log_vars.data.copy_(torch.log(new_var).unsqueeze(-1))
        else:  # full
            new_cov = torch.zeros_like(self.log_vars)            # [K, D, D]
            for k in range(self.k):
                w = r[:, k].unsqueeze(-1).unsqueeze(-1)          # [B, 1, 1]
                d = diff[:, k, :].unsqueeze(-1)                  # [B, D, 1]
                new_cov[k] = (w * d @ d.transpose(1, 2)).sum(dim=0) / N_k[k]
            new_cov = new_cov + self.reg * torch.eye(
                self.dim, dtype=new_cov.dtype
            ).unsqueeze(0)
            self.log_vars.data.copy_(new_cov)

        self.means.data.copy_(new_means)
        # log_weights via log of weights (with a tiny floor to avoid -inf)
        safe_w = torch.clamp(weights, min=1e-12)
        self.log_weights.data.copy_(torch.log(safe_w))

    def extra_repr(self) -> str:
        return (
            f"k={self.k}, dim={self.dim}, covariance={self.covariance_type!r}, "
            f"reg={self.reg}"
        )