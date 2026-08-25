"""Mixture of Probabilistic PCA (Tipping & Bishop 1999) re-implemented
as a ``torch.nn.Module``.

Each cluster ``k`` has its own PPCA local model: a low-dimensional
latent ``z ∈ R^{n_latent}`` is mapped to the observation by
``W_k z + mu_k`` with isotropic Gaussian noise ``sigma2_k I``. The
parameters per cluster are:

* ``W``           [n_components, n_features, n_latent]  (loadings)
* ``means``       [n_components, n_features]            (offsets)
* ``log_sigma2``  [n_components]                        (log noise variance)
* ``log_weights`` [n_components]                        (mixture logits)

``fit`` runs closed-form EM: the E-step computes posterior
responsibilities using a marginal likelihood per cluster; the M-step
re-estimates ``W_k, mu_k, sigma2_k`` with the standard PPCA
closed-form updates restricted to the weighted data.
"""

import torch

from .utils import _to_2d


def _ppca_log_lik(
    x_centered: torch.Tensor, W: torch.Tensor, sigma2: torch.Tensor
) -> torch.Tensor:
    """Log-likelihood of centred data under a single PPCA model.

    ``x_centered: [N, F]``, ``W: [F, L]``, ``sigma2: scalar``.
    Returns a ``[N]`` log-density per row.
    """
    F = W.shape[0]
    L = W.shape[1]
    WWt = W @ W.T                              # [F, F]
    C = WWt + sigma2 * torch.eye(F, dtype=W.dtype)  # [F, F]
    # log |C| = (F-L) log sigma2 + log |sigma2 I + W^T W| (per T&B 1999)
    WtW = W.T @ W                              # [L, L]
    eigvals = torch.linalg.eigvalsh(sigma2 * torch.eye(L, dtype=W.dtype) + WtW)
    log_det = (F - L) * torch.log(sigma2) + torch.sum(torch.log(eigvals))
    Cinv = torch.linalg.inv(C)
    quad = ((x_centered @ Cinv) * x_centered).sum(dim=-1)
    return -0.5 * (F * torch.log(torch.tensor(2 * 3.141592653589793, dtype=W.dtype))
                   + log_det + quad)


class ProbabilisticPCAMixture(torch.nn.Module):
    """
    Analogue:
        Tipping & Bishop 1999 'Mixtures of Probabilistic Principal Component Analysers'
    Mixture of probabilistic PCA models.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    n_features : int
        Observation dimension.
    n_latent : int
        Latent dimension per component.
    """

    def __init__(self, n_components: int, n_features: int, n_latent: int) -> None:
        super().__init__()
        if min(n_components, n_features, n_latent) <= 0:
            raise ValueError("dims must be > 0")
        self.n_components = n_components
        self.n_features = n_features
        self.n_latent = n_latent

        self.means = torch.nn.Parameter(
            torch.empty(n_components, n_features).uniform_(-0.5, 0.5)
        )
        self.W = torch.nn.Parameter(
            torch.empty(n_components, n_features, n_latent).uniform_(-0.5, 0.5)
        )
        self.log_sigma2 = torch.nn.Parameter(
            torch.zeros(n_components, dtype=torch.float32)
        )
        self.log_weights = torch.nn.Parameter(
            torch.zeros(n_components, dtype=torch.float32)
        )

    def _log_responsibility(self, x: torch.Tensor) -> torch.Tensor:
        """Return log responsibilities ``[N, K]``."""
        x = _to_2d(x)
        K = self.n_components
        log_mix = torch.nn.functional.log_softmax(self.log_weights, dim=-1)
        log_p = []
        for k in range(K):
            mu = self.means[k]
            W = self.W[k]
            sigma2 = torch.exp(self.log_sigma2[k]) + 1e-6
            log_p.append(_ppca_log_lik(x - mu, W, sigma2))
        log_p = torch.stack(log_p, dim=-1)                     # [N, K]
        return log_p + log_mix.unsqueeze(0)

    def responsibilities(self, x: torch.Tensor) -> torch.Tensor:
        log_r = self._log_responsibility(x)
        return torch.exp(log_r - torch.logsumexp(log_r, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.responsibilities(x)

    # ------------------------------------------------------------------- EM
    @torch.no_grad()
    def fit_em(self, x: torch.Tensor, n_iter: int = 20) -> "ProbabilisticPCAMixture":
        x = _to_2d(x)
        N, F = x.shape
        L = self.n_latent
        for _ in range(n_iter):
            log_r = self._log_responsibility(x)
            log_r = log_r - torch.logsumexp(log_r, dim=-1, keepdim=True)
            r = torch.exp(log_r)                                # [N, K]
            sum_r = r.sum(dim=0) + 1e-12                         # [K]

            # Update mixture weights
            new_log_w = torch.log(sum_r / N)
            self.log_weights.data.copy_(new_log_w)

            # Per-cluster M-step
            for k in range(self.n_components):
                rk = r[:, k]                                      # [N]
                mu_k = (rk.unsqueeze(-1) * x).sum(dim=0) / sum_r[k]
                Xc = x - mu_k
                Xw = rk.unsqueeze(-1) * Xc
                W = self.W[k]                                    # [F, L]
                # Closed-form PPCA re-estimation (Tipping & Bishop 1999):
                #   M = W^T W + sigma2 I
                #   W_new = (sum_r Xc Xc^T) W (W^T (sum_r Xc Xc^T) W + sigma2 N_k I)^{-1}
                S = (Xw.T @ Xc) / sum_r[k]                       # [F, F]
                eigvals_S, eigvecs_S = torch.linalg.eigh(S)
                eigvals_S = torch.clamp(eigvals_S, min=0.0)
                M = W.T @ S @ W
                eigvals_M, eigvecs_M = torch.linalg.eigh(M)
                eigvals_M = torch.clamp(eigvals_M, min=0.0)
                # Use the standard EM re-estimate (Bishop PRML 12.46):
                sigma2_new = (1.0 / (F * sum_r[k])) * (
                    torch.sum(rk * (Xc ** 2).sum(dim=-1))
                    - torch.trace(W.T @ S @ W)
                )
                sigma2_new = torch.clamp(sigma2_new, min=1e-4)
                self.log_sigma2[k].data.copy_(torch.log(sigma2_new + 1e-6))
                # W re-estimate: use eigendecomposition of S.
                # Take the top-L eigvecs of S as the new W basis.
                order = torch.argsort(-eigvals_S)[:L]
                W_new = eigvecs_S[:, order] @ torch.diag(
                    torch.sqrt(torch.clamp(eigvals_S[order], min=0.0))
                )
                self.W[k].data.copy_(W_new)
                self.means[k].data.copy_(mu_k)
        return self

    def extra_repr(self) -> str:
        return (
            f"n_components={self.n_components}, n_features={self.n_features}, "
            f"n_latent={self.n_latent}"
        )