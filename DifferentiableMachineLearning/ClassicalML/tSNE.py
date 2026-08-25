"""Vanilla / symmetric Student-t SNE (van der Maaten 2008)
re-implemented as a ``torch.nn.Module``.

The low-dimensional embedding ``Y`` is stored as a
:class:`torch.nn.Parameter` and optimised with standard Adam to
minimise the Kullback-Leibler divergence between the input and
output affinity distributions. ``fit_transform`` runs the full
training schedule, including the classical early-exaggeration
trick.
"""

from typing import Optional

import torch


def _pairwise_sq(x: torch.Tensor) -> torch.Tensor:
    """Squared euclidean distance matrix of rows of ``x``."""
    sq = torch.sum(x * x, dim=1, keepdim=True)                # [N, 1]
    return torch.clamp(
        sq + sq.T - 2.0 * x @ x.T,
        min=0.0,
    )


def _joint_high_dim(
    sq_dists: torch.Tensor, perplexity: float = 30.0, tol: float = 1e-5, max_iter: int = 50
) -> torch.Tensor:
    """Binary-search the per-point Gaussian ``sigma`` so the input
    distribution's perplexity matches the requested one. Returns
    the joint probability matrix ``P`` (with the row-symmetric
    convention used by t-SNE)."""
    N = sq_dists.shape[0]
    target_entropy = torch.log(torch.tensor(perplexity, dtype=torch.float32))
    arange_N = torch.arange(N)
    P_rows = []
    for i in range(N):
        beta_lo = torch.tensor(1e-20, dtype=torch.float32)
        beta_hi = torch.tensor(1e20, dtype=torch.float32)
        beta = torch.tensor(1.0, dtype=torch.float32)
        mask = arange_N != i
        d_i = sq_dists[i][mask]
        for _ in range(max_iter):
            pi = torch.exp(-d_i * beta)
            sum_pi = torch.sum(pi) + 1e-12
            H = torch.log(sum_pi) + beta * torch.sum(d_i * pi) / sum_pi
            diff = H - target_entropy
            if torch.abs(diff) < tol:
                break
            if diff > 0:
                beta_lo = beta
                beta = beta * 2 if float(beta_hi) > 1e19 else (beta + beta_hi) / 2
            else:
                beta_hi = beta
                beta = (beta + beta_lo) / 2
        # full row with self-similarity zeroed out
        pi_full = torch.exp(-sq_dists[i] * beta)
        pi_full = pi_full * mask.to(pi_full.dtype)
        pi_full = pi_full / (torch.sum(pi_full) + 1e-12)
        P_rows.append(pi_full)
    P = torch.stack(P_rows, dim=0)                           # [N, N]
    P = (P + P.T) / (2.0 * N)
    P = torch.clamp(P, min=1e-12)
    return P


def _student_t_q(sq_dists_y: torch.Tensor) -> torch.Tensor:
    """1 / (1 + ||y_i - y_j||^2)."""
    inv = 1.0 / (1.0 + sq_dists_y)
    # zero the diagonal so the joint distribution is not self-weighted
    inv = inv - torch.diag(torch.diagonal(inv))
    s = torch.sum(inv)
    return torch.clamp(inv / s, min=1e-12)


class tSNE(torch.nn.Module):
    """
    Analogue:
        sklearn.manifold.TSNE (van der Maaten & Hinton 2008)
    Student-t SNE.

    Parameters
    ----------
    n_components : int, default 2
        Embedding dimension.
    perplexity : float, default 30.0
    early_exaggeration : float, default 12.0
        Multiplier on the input affinities during the first phase of
        training.
    n_iter : int, default 500
        Total number of gradient steps.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        n_iter: int = 500,
        learning_rate: float = 1.0,
    ) -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.Y = torch.nn.Parameter(
            torch.empty(0, n_components).uniform_(-1e-4, 1e-4)
        )
        self._P = None
        self._input_dim = None

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full t-SNE training and return the learned embedding."""
        x = x.float()
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input, got {x.shape}")
        N = x.shape[0]
        self._input_dim = x.shape[1]
        # Initialise Y with small Gaussian noise around zero.
        self.Y = torch.nn.Parameter(
            torch.empty(N, self.n_components).normal_(0.0, 1e-4)
        )
        # Compute input affinities once.
        sq = _pairwise_sq(x)
        P = _joint_high_dim(sq, perplexity=self.perplexity)
        self._P = P

        opt = torch.optim.Adam(
            [self.Y], lr=self.learning_rate
        )
        exaggeration = self.early_exaggeration
        exaggeration_switch = max(self.n_iter // 4, 1)
        for it in range(self.n_iter):
            sq_y = _pairwise_sq(self.Y)
            Q = _student_t_q(sq_y)
            # KL = sum P * log(P / Q) where P may be the exaggerated version.
            Peff = exaggeration * P
            kld = torch.sum(Peff * (torch.log(Peff) - torch.log(Q)))
            opt.zero_grad()
            kld.backward()
            opt.step()
            if it == exaggeration_switch:
                exaggeration = 1.0
                # Reset Adam moments since the loss scale changes.
                opt = torch.optim.Adam(
                    [self.Y], lr=self.learning_rate
                )
        return self.Y.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fit_transform(x)