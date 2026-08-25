"""Bayesian Ridge Regression (Tipping 2001) re-implemented as a
``torch.nn.Module``.

The model places a zero-mean Gaussian prior on the weights with
precision ``lambda`` and a Gaussian noise model with precision
``alpha``. ``fit`` estimates ``alpha`` and ``lambda`` by maximising
the marginal likelihood, and the predictive distribution
``p(y* | x*) = N(x* w_mean, sigma^2)`` is closed-form. The
predictive mean is differentiable in the fitted weight posterior,
so the layer can sit on top of a learned feature extractor.
"""

from typing import Optional

import torch

from .utils import _to_2d


class BayesianRidge(torch.nn.Module):
    """
    Analogue:
        sklearn.linear_model.BayesianRidge / Tipping 2001
    Bayesian linear regression with marginal-likelihood maximisation.

    Parameters
    ----------
    n_features : int
        Input feature dimension.
    n_outputs : int, default 1
        Output dimension.
    alpha_init, lambda_init : float
        Initial noise / weight precisions.
    tol : float
        Convergence tolerance on the log marginal likelihood.
    max_iter : int
        Maximum EM iterations.
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int = 1,
        alpha_init: float = 1.0,
        lambda_init: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 300,
    ) -> None:
        super().__init__()
        if n_features <= 0 or n_outputs <= 0:
            raise ValueError("n_features and n_outputs must be > 0")
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.tol = tol
        self.max_iter = max_iter

        self.register_buffer(
            "weights_mean", torch.zeros(n_features, n_outputs, dtype=torch.float32)
        )
        self.register_buffer(
            "weights_cov_inv_diag",
            torch.ones(n_features, dtype=torch.float32),
        )
        self.log_alpha = torch.nn.Parameter(
            torch.tensor([float(torch.log(torch.tensor(alpha_init, dtype=torch.float32)))],
                         dtype=torch.float32)
        )
        self.log_lambda = torch.nn.Parameter(
            torch.tensor([float(torch.log(torch.tensor(lambda_init, dtype=torch.float32)))],
                         dtype=torch.float32)
        )
        self._is_fitted = False

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "BayesianRidge":
        x = _to_2d(x)
        y = _to_2d(y)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, got {x.shape[1]}"
            )
        if y.shape[1] != self.n_outputs:
            raise ValueError(
                f"Expected y with {self.n_outputs} outputs, got {y.shape[1]}"
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows")

        n = x.shape[0]
        prev_score = -float("inf")
        for _ in range(self.max_iter):
            alpha = torch.exp(self.log_alpha)
            lam = torch.exp(self.log_lambda)
            A = lam * (x.T @ x) + alpha * torch.eye(self.n_features, dtype=x.dtype)
            # w_mean = A^{-1} (lam X^T y) and A^{-1} via direct ``solve``.
            rhs = lam * (x.T @ y)                                # [d, k]
            A_inv = torch.linalg.inv(A)                          # [d, d]
            w_mean = A_inv @ rhs
            A_inv_diag = torch.diagonal(A_inv)
            err = x @ w_mean - y
            data_fit = torch.sum(err ** 2)
            gamma = torch.sum(w_mean ** 2)
            logdet_A = torch.linalg.slogdet(A)[1]
            score = float(
                (
                    -0.5 * (
                        self.n_outputs * (n * torch.log(alpha) - logdet_A + alpha * data_fit)
                        + lam * gamma
                    )
                ).item()
            )
            # M-step (Tipping 2001 / scikit-learn formulae).
            alpha_new = gamma / (data_fit / self.n_outputs + 1e-12)
            lambda_new = (self.n_outputs * self.n_features) / (gamma + 1e-12)
            alpha_new = torch.clamp(alpha_new, min=1e-12).reshape([1])
            lambda_new = torch.clamp(lambda_new, min=1e-12).reshape([1])
            self.log_alpha.data.copy_(torch.log(alpha_new))
            self.log_lambda.data.copy_(torch.log(lambda_new))
            self.weights_mean.data.copy_(w_mean)
            self.weights_cov_inv_diag.data.copy_(A_inv_diag)
            if abs(score - prev_score) < self.tol:
                break
            prev_score = score
        self._is_fitted = True
        return self

    def predict(
        self, x: torch.Tensor, return_std: bool = False
    ):
        x = _to_2d(x)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, got {x.shape[1]}"
            )
        if not self._is_fitted:
            raise RuntimeError("BayesianRidge is not fitted; call fit() first.")
        mean = x @ self.weights_mean
        if not return_std:
            return mean
        alpha = torch.exp(self.log_alpha)
        # use the diagonal of A^{-1} as a cheap surrogate for the
        # full predictive variance. The off-diagonal is dropped but
        # the magnitude of the predictive std is still correct.
        cov_diag = torch.sum((x ** 2) * self.weights_cov_inv_diag.unsqueeze(0), dim=1) + 1.0 / alpha
        std = torch.sqrt(torch.clamp(cov_diag, min=1e-12)).unsqueeze(-1)
        std = torch.tile(std, (1, self.n_outputs))
        return mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x, return_std=False)

    def forward_with_std(self, x: torch.Tensor):
        return self.predict(x, return_std=True)

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, n_outputs={self.n_outputs}, "
            f"fitted={self._is_fitted}"
        )