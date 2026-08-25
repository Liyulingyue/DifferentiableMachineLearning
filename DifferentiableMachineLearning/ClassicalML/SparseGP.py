"""Sparse / variational Gaussian Process regression (Titsias 2009;
Hensman et al. 2013) re-implemented as a ``torch.nn.Module``.

Analogue:
    GPflow's ``SVGP`` (Hensman et al. 2013); sklearn's
    ``GaussianProcessRegressor`` corresponds to the dense
    ``M=N`` limit.

The layer holds ``M`` learnable **inducing** locations
``Z: [M, dim]`` plus a variational Gaussian posterior over the
corresponding function values ``q(u) = N(m, S)``. The
marginal-likelihood ELBO is the standard Titsias bound,

    log p(y) >= sum_i log N(y_i | a_i, sigma^2)
                 - 0.5 * sigma^{-2} * tr(K_ff - Q_ff)

where ``Q_ff = K_fu K_uu^{-1} S K_uu^{-1} K_uf`` plus
``KL(q(u) || p(u))``. ``fit`` minimises the negative ELBO with
Adam and a small jitter on every covariance inverse to keep
``K_uu`` numerically well-conditioned.

Forward returns the posterior-predictive mean and (epistemic
plus aleatoric) standard deviation, both computed in closed
form from ``m, S, Z, X`` and the chosen kernel.
"""

import torch

from .KernelRidge import _linear, _poly, _rbf
from .utils import _to_2d


def _rbf_gamma(log_lengthscale: torch.Tensor, dim: int) -> float:
    ls = float(torch.exp(log_lengthscale.clamp(-2.0, 3.0)).item())
    return 1.0 / (2.0 * ls ** 2)


class SparseGP(torch.nn.Module):
    """Variational sparse Gaussian Process regression (Titsias 2009).

    Analogue:
        GPflow's ``SVGP`` (Hensman et al. 2013); sklearn's
        ``GaussianProcessRegressor`` corresponds to the dense
        ``M=N`` limit.

    Parameters
    ----------
    n_inducing : int
        Number of inducing points ``M``.
    dim : int
        Input feature dimension.
    n_train : int
        Number of training points (set by ``fit``; used for buffer
        bookkeeping).
    kernel : {"rbf", "linear", "polynomial"}
    log_lengthscale, log_outputscale, log_noise : float
        Initial kernel hyperparameters; outputscale and noise are
        learnable scalars; lengthscale is also learnable.
    """

    _KERNELS = {"rbf": _rbf, "linear": _linear, "polynomial": _poly}

    def __init__(
        self,
        n_inducing: int,
        dim: int,
        n_train: int,
        kernel: str = "rbf",
        log_lengthscale: float = 0.0,
        log_outputscale: float = 0.0,
        log_noise: float = -2.0,
    ) -> None:
        super().__init__()
        if kernel not in self._KERNELS:
            raise ValueError(f"Unknown kernel {kernel!r}")
        if min(n_inducing, dim, n_train) <= 0:
            raise ValueError("dims must be > 0")
        self.n_inducing = n_inducing
        self.dim = dim
        self.n_train = n_train
        self.kernel = kernel

        # Inducing locations: initialised by sub-sampling the
        # training set later in fit(); we begin with random locations.
        self.Z = torch.nn.Parameter(
            torch.empty(n_inducing, dim).uniform_(-1.0, 1.0)
        )
        self.log_lengthscale = torch.nn.Parameter(
            torch.tensor([log_lengthscale], dtype=torch.float32)
        )
        self.log_outputscale = torch.nn.Parameter(
            torch.tensor([log_outputscale], dtype=torch.float32)
        )
        self.log_noise = torch.nn.Parameter(
            torch.tensor([log_noise], dtype=torch.float32)
        )
        # Variational parameters over u.
        self.q_m = torch.nn.Parameter(
            torch.zeros(n_inducing, dtype=torch.float32)
        )
        self.q_log_diag = torch.nn.Parameter(
            torch.full((n_inducing,), -3.0, dtype=torch.float32)
        )
        self.register_buffer(
            "X_train", torch.zeros(n_train, dim, dtype=torch.float32)
        )
        self.register_buffer(
            "y_train", torch.zeros(n_train, dtype=torch.float32)
        )
        self._is_fitted = False

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ls2 = float(torch.exp(2.0 * self.log_lengthscale).item())
        gamma = 1.0 / (2.0 * max(min(ls2, 1e3), 1e-3))
        K = self._KERNELS[self.kernel](x, y, gamma)
        os = float(torch.exp(self.log_outputscale).item())
        return max(min(os, 1e3), 1e-3) * K

    @torch.no_grad()
    def fit_init(self, x: torch.Tensor, y: torch.Tensor) -> "SparseGP":
        x = _to_2d(x)
        y = y.reshape([-1])
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        if x.shape[0] != self.n_train:
            raise ValueError(
                f"Built for n_train={self.n_train}, got {x.shape[0]}"
            )
        self.X_train.data.copy_(x)
        self.y_train.data.copy_(y)
        # Sub-sample inducing locations from the training data.
        if x.shape[0] >= self.n_inducing:
            idx = torch.randperm(x.shape[0])[: self.n_inducing]
        else:
            idx = torch.arange(x.shape[0])
        self.Z.data.copy_(x[idx])
        self._is_fitted = True
        return self

    def elbo(self) -> torch.Tensor:
        """Negative ELBO for the Titsias bound.

        The bound is
            log p(y) >= log N(y | K_fu K_uu^{-1} m, sigma^2 I)
                    - 0.5 sigma^{-2} tr(K_ff - Q_ff)
                    - KL(q(u) || p(u))
        with ``Q_ff = K_fu K_uu^{-1} S K_uu^{-1} K_uf`` and
        ``S = diag(exp(2 q_log_diag))``.
        """
        x = self.X_train
        y = self.y_train
        noise = torch.exp(self.log_noise) ** 2
        K_uu = self._kernel(self.Z, self.Z) + 1e-4 * torch.eye(
            self.n_inducing, dtype=x.dtype
        )
        L = torch.linalg.cholesky(K_uu)
        K_fu = self._kernel(x, self.Z)                                # [N, M]
        # A = L^{-1} K_fu^T  (M x N), so that A^T K_fu^... is easy.
        A = torch.linalg.solve_triangular(L, K_fu.T, upper=False).T    # [N, M]
        mean = A @ self.q_m                                            # [N]
        K_ff_diag = torch.exp(self.log_outputscale) * torch.ones(
            x.shape[0], dtype=x.dtype
        )
        # tr(K_ff - Q_ff) = tr(K_ff) - tr(A^T A S_diag)
        S_diag = torch.exp(2.0 * self.q_log_diag)                     # [M]
        # diag(A^T A S_diag) = sum_m A[:, m]^2 S_diag[m]  but we want tr
        tr_Qff = torch.sum(A ** 2 * S_diag.unsqueeze(0))
        trace_term = (torch.sum(K_ff_diag) - tr_Qff) / noise
        quad = torch.sum((y - mean) ** 2) / noise
        const = float(x.shape[0]) * torch.log(
            2 * 3.141592653589793 * noise
        )
        # KL(q(u) || p(u)) where p(u) = N(0, outputscale^2 I).
        os2 = float(torch.exp(2.0 * self.log_outputscale).item())
        prior_var = max(min(os2, 1e3), 1e-3)
        var_q = torch.exp(2.0 * self.q_log_diag)
        kl = 0.5 * torch.sum(
            (var_q + self.q_m ** 2) / prior_var
            + torch.log(torch.tensor(prior_var, dtype=x.dtype))
            - 2.0 * self.q_log_diag
            - 1.0
        )
        elbo = -0.5 * (const + quad + trace_term) - kl
        return -elbo  # we minimise the negative ELBO

    def predict(
        self, x: torch.Tensor, return_std: bool = True
    ):
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got {x.shape[1]}"
            )
        if not self._is_fitted:
            raise RuntimeError("SparseGP is not fitted; call fit_init() first.")
        K_uu = self._kernel(self.Z, self.Z) + 1e-4 * torch.eye(
            self.n_inducing, dtype=x.dtype
        )
        L = torch.linalg.cholesky(K_uu)
        K_xu = self._kernel(x, self.Z)
        A = torch.linalg.solve_triangular(L, K_xu.T, upper=False).T    # [M_x, M]
        mean = A @ self.q_m
        if not return_std:
            return mean.unsqueeze(-1)
        K_xx_diag = torch.exp(self.log_outputscale) * torch.ones(
            x.shape[0], dtype=x.dtype
        )
        S_diag = torch.exp(2.0 * self.q_log_diag)
        # Predictive variance = K_xx - A A^T diag + noise
        var = K_xx_diag - (A ** 2 * S_diag.unsqueeze(0)).sum(dim=1)
        var = var + torch.exp(self.log_noise) ** 2
        var = torch.clamp(var, min=1e-10)
        std = torch.sqrt(var)
        return mean.unsqueeze(-1), std.unsqueeze(-1)

    def fit(
        self, n_outer: int = 500, lr: float = 5e-2
    ) -> "SparseGP":
        """Tune hyperparameters + variational parameters by Adam."""
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(n_outer):
            loss = self.elbo()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x, return_std=False)

    def extra_repr(self) -> str:
        return (
            f"n_inducing={self.n_inducing}, dim={self.dim}, "
            f"n_train={self.n_train}, kernel={self.kernel!r}"
        )