"""Kernel Ridge Regression re-implemented as a ``torch.nn.Module``.

The model is the dual form

    f(x) = sum_i alpha_i K(x_i, x)

where ``x_i`` are stored *support points* (the training set, or a
subset) and ``alpha`` are the dual coefficients. ``fit`` solves

    alpha = (K + lambda I)^{-1} y

in closed form. Supports RBF, linear, and polynomial kernels.
"""

from typing import Optional

import torch

from .utils import _to_2d


_KERNELS = {"rbf", "linear", "polynomial"}


def _rbf(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    sq_x = torch.sum(x * x, dim=1, keepdim=True)             # [n, 1]
    sq_y = torch.sum(y * y, dim=1)                           # [m]
    cross = x @ y.T                                            # [n, m]
    dist = torch.clamp(
        sq_x + sq_y.unsqueeze(0) - 2.0 * cross,
        min=0.0,
    )
    return torch.exp(-gamma * dist)


def _linear(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y.T


def _poly(x: torch.Tensor, y: torch.Tensor, gamma: float, c: float, d: int) -> torch.Tensor:
    return (gamma * (x @ y.T) + c) ** d


class KernelRidge(torch.nn.Module):
    """
    Analogue:
        sklearn.kernel_ridge.KernelRidge (Saunders et al. 1998)
    Dual-form kernel ridge regression.

    Parameters
    ----------
    n_support : int
        Number of stored support points (set by ``fit``).
    dim_in : int
        Input feature dimension.
    dim_out : int
        Output dimension (1 for scalar regression; >1 for multi-output).
    kernel : {"rbf", "linear", "polynomial"}
    gamma : float
        Kernel coefficient for RBF / polynomial.
    coef0 : float
        Polynomial bias term.
    degree : int
        Polynomial degree.
    alpha : float
        L2 regularisation on the dual coefficients.
    """

    def __init__(
        self,
        n_support: int,
        dim_in: int,
        dim_out: int = 1,
        kernel: str = "rbf",
        gamma: float = 1.0,
        coef0: float = 1.0,
        degree: int = 3,
        alpha: float = 1e-2,
    ) -> None:
        super().__init__()
        if kernel not in _KERNELS:
            raise ValueError(f"Unknown kernel {kernel!r}; pick from {_KERNELS}")
        if n_support <= 0 or dim_in <= 0 or dim_out <= 0:
            raise ValueError("n_support, dim_in, dim_out must all be > 0")

        self.n_support = n_support
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.alpha = alpha

        self.register_buffer(
            "support", torch.zeros(n_support, dim_in, dtype=torch.float32)
        )
        self.register_buffer(
            "dual_coef", torch.zeros(n_support, dim_out, dtype=torch.float32)
        )
        self.register_buffer(
            "y_train", torch.zeros(n_support, dim_out, dtype=torch.float32)
        )
        self._is_fitted = False

    def _K(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kernel == "rbf":
            return _rbf(x, y, self.gamma)
        if self.kernel == "linear":
            return _linear(x, y)
        return _poly(x, y, self.gamma, self.coef0, self.degree)

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "KernelRidge":
        x = _to_2d(x)
        y = _to_2d(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        if x.shape[1] != self.dim_in:
            raise ValueError(
                f"Expected input with {self.dim_in} features, got {x.shape[1]}"
            )
        if y.shape[1] != self.dim_out:
            raise ValueError(
                f"Expected y with {self.dim_out} outputs, got {y.shape[1]}"
            )
        n = x.shape[0]
        if n != self.n_support:
            raise ValueError(
                f"KernelRidge was built for n_support={self.n_support}, "
                f"got {n} training samples"
            )
        self.support.data.copy_(x)
        self.y_train.data.copy_(y)
        K = self._K(x, x)                                       # [n, n]
        K = K + self.alpha * torch.eye(n, dtype=K.dtype)
        self.dual_coef.data.copy_(torch.linalg.solve(K, y))
        self._is_fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_2d(x)
        if not self._is_fitted:
            raise RuntimeError("KernelRidge is not fitted; call fit() first.")
        if x.shape[1] != self.dim_in:
            raise ValueError(
                f"Expected input with {self.dim_in} features, got {x.shape[1]}"
            )
        K = self._K(x, self.support)                            # [batch, n]
        return K @ self.dual_coef                               # [batch, dim_out]

    def extra_repr(self) -> str:
        return (
            f"n_support={self.n_support}, dim_in={self.dim_in}, "
            f"dim_out={self.dim_out}, kernel={self.kernel!r}, "
            f"alpha={self.alpha}, fitted={self._is_fitted}"
        )