"""Gaussian Process regression re-implemented as a
``torch.nn.Module``.

Kernel hyperparameters (``log_lengthscale``, ``log_outputscale``,
``log_noise``) are stored as :class:`torch.nn.Parameter` so the
layer can be tuned with marginal-likelihood maximisation (Type-II
ML) via standard Adam / L-BFGS. The predictive distribution is
closed-form:

    K = k(X_train, X_train) + noise * I
    alpha = K^{-1} y
    mean(x*) = k(x*, X_train) @ alpha
    var(x*)  = k(x*, x*) - k(x*, X_train) @ K^{-1} @ k(X_train, x*)

The predictive mean is differentiable in ``alpha`` and therefore in
the kernel hyperparameters, so the layer can act as a probabilistic
Bayesian head on top of a learned feature extractor.
"""

from typing import Optional, Tuple

import torch

from .KernelRidge import _linear, _poly, _rbf
from .utils import _to_2d


def _matern_32(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Matérn-3/2 kernel: ``(1 + sqrt(3) d / l) exp(-sqrt(3) d / l)``."""
    sq_x = torch.sum(x * x, dim=1, keepdim=True)
    sq_y = torch.sum(y * y, dim=1)
    cross = x @ y.T
    d2 = torch.clamp(
        sq_x + sq_y.unsqueeze(0) - 2.0 * cross,
        min=0.0,
    )
    d = torch.sqrt(d2 + 1e-12)
    sqrt3 = float(torch.sqrt(torch.tensor(3.0)))
    g = torch.sqrt(torch.tensor(gamma, dtype=x.dtype))
    return (1.0 + sqrt3 * d * g) * torch.exp(-sqrt3 * d * g)


def _matern_52(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Matérn-5/2 kernel: ``(1 + sqrt(5) d / l + 5 d^2 / (3 l^2)) exp(-sqrt(5) d / l)``."""
    sq_x = torch.sum(x * x, dim=1, keepdim=True)
    sq_y = torch.sum(y * y, dim=1)
    cross = x @ y.T
    d2 = torch.clamp(
        sq_x + sq_y.unsqueeze(0) - 2.0 * cross,
        min=0.0,
    )
    d = torch.sqrt(d2 + 1e-12)
    sqrt5 = float(torch.sqrt(torch.tensor(5.0)))
    g = float(torch.sqrt(torch.tensor(gamma, dtype=x.dtype)).item())
    sg = sqrt5 * d * g
    return (1.0 + sg + sg ** 2 * (5.0 / 3.0)) * torch.exp(-sg)


class GaussianProcess(torch.nn.Module):
    """
    Analogue:
        sklearn.gaussian_process.GaussianProcessRegressor (Rasmussen & Williams 2006)
    Gaussian Process regression.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    n_train : int
        Number of stored training points; set by ``fit`` and
        matched against the data shape.
    kernel : {"rbf", "matern32", "matern52", "linear", "polynomial"}
    log_lengthscale : float
        Initial log-lengthscale. ``gamma = exp(-2 log_lengthscale)``
        so the RBF becomes ``exp(-||x-y||^2 / (2 l^2))``.
    log_outputscale : float
        Initial log-amplitude in front of the kernel.
    log_noise : float
        Initial log-observation-noise standard deviation.
    learnable_hyperparams : bool, default True
        If True, the three hyperparameters are :class:`torch.nn.Parameter`
        and can be tuned by ``fit``.
    """

    _KERNELS = {
        "rbf": _rbf,
        "matern32": _matern_32,
        "matern52": _matern_52,
        "linear": _linear,
        "polynomial": _poly,
    }

    def __init__(
        self,
        dim: int,
        n_train: int,
        kernel: str = "rbf",
        log_lengthscale: float = 0.0,
        log_outputscale: float = 0.0,
        log_noise: float = -3.0,
        learnable_hyperparams: bool = True,
    ) -> None:
        super().__init__()
        if kernel not in self._KERNELS:
            raise ValueError(f"Unknown kernel {kernel!r}")
        if dim <= 0 or n_train <= 0:
            raise ValueError("dim and n_train must be > 0")

        self.dim = dim
        self.n_train = n_train
        self.kernel = kernel

        if learnable_hyperparams:
            self.log_lengthscale = torch.nn.Parameter(
                torch.tensor([log_lengthscale], dtype=torch.float32)
            )
            self.log_outputscale = torch.nn.Parameter(
                torch.tensor([log_outputscale], dtype=torch.float32)
            )
            self.log_noise = torch.nn.Parameter(
                torch.tensor([log_noise], dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_lengthscale",
                torch.tensor([log_lengthscale], dtype=torch.float32),
            )
            self.register_buffer(
                "log_outputscale",
                torch.tensor([log_outputscale], dtype=torch.float32),
            )
            self.register_buffer(
                "log_noise",
                torch.tensor([log_noise], dtype=torch.float32),
            )

        self.register_buffer(
            "X_train", torch.zeros(n_train, dim, dtype=torch.float32)
        )
        self.register_buffer(
            "y_train", torch.zeros(n_train, dtype=torch.float32)
        )
        self.register_buffer(
            "alpha", torch.zeros(n_train, dtype=torch.float32)
        )
        self.register_buffer(
            "K_inv", torch.eye(n_train, dtype=torch.float32)
        )
        self._is_fitted = False

    def _gamma(self) -> float:
        # gamma = 1 / (2 l^2) for the RBF; equivalent to the inverse-square
        # lengthscale used by the existing _rbf helper. We clip the
        # lengthscale to [1e-3, 1e3] to keep K well-conditioned.
        ls = float(torch.exp(self.log_lengthscale).item())
        ls = max(min(ls, 1e3), 1e-3)
        return 1.0 / (2.0 * ls ** 2)

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        g = self._gamma()
        if self.kernel == "rbf":
            K = _rbf(x, y, g)
        elif self.kernel == "matern32":
            K = _matern_32(x, y, g)
        elif self.kernel == "matern52":
            K = _matern_52(x, y, g)
        elif self.kernel == "linear":
            K = _linear(x, y)
        else:
            K = _poly(x, y, g, c=1.0, d=3)
        os = float(torch.exp(self.log_outputscale).item())
        os = max(min(os, 1e3), 1e-3)
        return os * K

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "GaussianProcess":
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        y = y.reshape([-1]) if y.ndim > 1 else y
        n = x.shape[0]
        if n != self.n_train:
            raise ValueError(
                f"GP was built for n_train={self.n_train}, got {n}"
            )
        self.X_train.data.copy_(x)
        self.y_train.data.copy_(y)
        K = self._kernel(x, x) + torch.exp(self.log_noise) ** 2 * torch.eye(
            n, dtype=x.dtype
        )
        K_inv = torch.linalg.inv(K)
        self.K_inv.data.copy_(K_inv)
        self.alpha.data.copy_(K_inv @ y)
        self._is_fitted = True
        return self

    def neg_log_marginal_likelihood(self) -> torch.Tensor:
        """Differentiable negative log marginal likelihood. Use as a
        loss to tune the kernel hyperparameters by gradient descent."""
        n = self.X_train.shape[0]
        noise = torch.exp(self.log_noise).clamp(min=1e-6) ** 2
        K = self._kernel(self.X_train, self.X_train) + noise * torch.eye(
            n, dtype=self.X_train.dtype
        )
        # Add a small jitter to keep K numerically invertible during
        # hyperparameter tuning.
        K = K + 1e-4 * torch.eye(n, dtype=K.dtype)
        K_inv = torch.linalg.inv(K)
        alpha = K_inv @ self.y_train
        return 0.5 * (
            self.y_train @ alpha
            + torch.linalg.slogdet(K)[1]
            + n * torch.log(torch.tensor(2 * 3.141592653589793, dtype=K.dtype))
        )

    def predict(
        self, x: torch.Tensor, return_std: bool = False
    ):
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        if not self._is_fitted:
            raise RuntimeError("GP is not fitted; call fit() first.")
        K_xs = self._kernel(x, self.X_train)                     # [M, n]
        mean = K_xs @ self.alpha                                 # [M]
        if not return_std:
            return mean.unsqueeze(-1)
        K_ss_diag = torch.exp(self.log_outputscale) * torch.ones(
            x.shape[0], dtype=x.dtype
        )
        var = K_ss_diag - (K_xs @ self.K_inv * K_xs).sum(dim=1)
        var = torch.clamp(var, min=1e-10)
        std = torch.sqrt(var)
        return mean.unsqueeze(-1), std.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x, return_std=False)

    def forward_with_std(self, x: torch.Tensor):
        return self.predict(x, return_std=True)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_train={self.n_train}, kernel={self.kernel!r}, "
            f"fitted={self._is_fitted}"
        )