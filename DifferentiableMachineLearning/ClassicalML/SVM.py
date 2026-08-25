"""Least-Squares Support Vector Machine (LS-SVM, Suykens 1999)
re-implemented as a ``torch.nn.Module``.

The classical SVM dual is a quadratic program in the support
coefficients. The LS-SVM variant replaces the inequality constraint
with an equality constraint, turning the dual into a linear
system that can be solved in closed form with ``torch.linalg.solve``.
The resulting classifier is differentiable in the training set, so
it can act as a kernelised head in a deep pipeline.

For multi-class problems we use the one-vs-rest reduction: a list
of LS-SVMs, one per class.
"""

from typing import List, Optional, Tuple

import torch

from .KernelRidge import _linear, _poly, _rbf
from .utils import _to_2d


class _BinaryLSSVM(torch.nn.Module):
    """Binary LS-SVM with kernelised RBF / linear / polynomial."""

    _KERNELS = {"rbf": _rbf, "linear": _linear, "polynomial": _poly}

    def __init__(
        self,
        n_support: int,
        dim: int,
        kernel: str = "rbf",
        gamma: float = 1.0,
        coef0: float = 1.0,
        degree: int = 3,
        C: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel not in self._KERNELS:
            raise ValueError(f"Unknown kernel {kernel!r}")
        if n_support <= 0 or dim <= 0:
            raise ValueError("n_support and dim must be > 0")

        self.n_support = n_support
        self.dim = dim
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.C = C

        self.register_buffer(
            "support", torch.zeros(n_support, dim, dtype=torch.float32)
        )
        self.register_buffer(
            "dual_coef", torch.zeros(n_support, dtype=torch.float32)
        )
        self.register_buffer(
            "bias", torch.zeros(1, dtype=torch.float32)
        )
        self._is_fitted = False

    def _K(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kernel == "rbf":
            return _rbf(x, y, self.gamma)
        if self.kernel == "linear":
            return _linear(x, y)
        return _poly(x, y, self.gamma, self.coef0, self.degree)

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "_BinaryLSSVM":
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        if y.ndim == 2:
            y = y.squeeze(-1)
        n = x.shape[0]
        if n != self.n_support:
            raise ValueError(
                f"LS-SVM was built for n_support={self.n_support}, got {n}"
            )
        self.support.data.copy_(x)
        K = self._K(x, x)
        # LS-SVM linear system:
        #   [ K    1_n ] [ alpha ] = [ y ]
        #   [ 1_n^T  0  ] [ b     ]   [ 0 ]
        ones = torch.ones(n, 1, dtype=x.dtype)
        top = torch.cat([K + (1.0 / self.C) * torch.eye(n, dtype=x.dtype), ones], dim=1)
        bot = torch.cat([ones.T, torch.zeros(1, 1, dtype=x.dtype)], dim=1)
        A = torch.cat([top, bot], dim=0)
        b = torch.cat([y.to(x.dtype), torch.zeros(1, dtype=x.dtype)], dim=0)
        sol = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
        self.dual_coef.data.copy_(sol[:n])
        self.bias.data.copy_(sol[n:])
        self._is_fitted = True
        return self

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        K = self._K(x, self.support)
        return K @ self.dual_coef + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decision_function(x)


class SVM(torch.nn.Module):
    """
    Analogue:
        sklearn.svm.SVC, LS-SVM (Suykens & Vandewalle 1999) for the closed-form variant
    Kernelised LS-SVM with one-vs-rest multi-class extension.

    Parameters
    ----------
    n_support : int
        Number of stored support points (set by ``fit``).
    dim : int
        Input feature dimension.
    n_classes : int, default 2
        For >2 classes, an OvR ensemble is built and ``forward``
        returns the multi-class logits.
    kernel : {"rbf", "linear", "polynomial"}
    gamma, coef0, degree, C : kernel and regularisation parameters.
    """

    def __init__(
        self,
        n_support: int,
        dim: int,
        n_classes: int = 2,
        kernel: str = "rbf",
        gamma: float = 1.0,
        coef0: float = 1.0,
        degree: int = 3,
        C: float = 1.0,
    ) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = n_classes
        if n_classes == 2:
            self.binary = _BinaryLSSVM(
                n_support=n_support, dim=dim, kernel=kernel,
                gamma=gamma, coef0=coef0, degree=degree, C=C,
            )
            self.multi: Optional[torch.nn.ModuleList] = None
        else:
            self.multi = torch.nn.ModuleList([
                _BinaryLSSVM(
                    n_support=n_support, dim=dim, kernel=kernel,
                    gamma=gamma, coef0=coef0, degree=degree, C=C,
                )
                for _ in range(n_classes)
            ])
            self.binary = None

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "SVM":
        x = _to_2d(x)
        y = y.long()
        if y.ndim == 2:
            y = y.squeeze(-1)
        if self.n_classes == 2:
            binary_y = torch.where(y == 0, -1.0, 1.0)
            self.binary.fit(x, binary_y)
        else:
            for c in range(self.n_classes):
                binary_y = torch.where(y == c, 1.0, -1.0)
                self.multi[c].fit(x, binary_y)
        return self

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_classes == 2:
            return self.binary.decision_function(x)
        return torch.stack(
            [m.decision_function(x) for m in self.multi], dim=-1
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_classes == 2:
            return (self.binary.decision_function(x) > 0).long()
        return torch.argmax(self.decision_function(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decision_function(x)

    def extra_repr(self) -> str:
        return f"n_classes={self.n_classes}"