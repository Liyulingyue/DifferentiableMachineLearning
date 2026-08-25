"""Naive Bayes classifiers re-implemented as ``torch.nn.Module``.

* :class:`GaussianNB`        — per-class Gaussian density; gradients
  flow through ``log p(y|x)`` so the layer can be used as a
  differentiable head.
* :class:`MultinomialNB`     — per-class multinomial with Laplace
  smoothing; counts live in buffers, no parameters.

The two share the same observation / predict interface, so they are
both valid drop-in classifier heads in deep pipelines.
"""

from typing import Optional

import torch

from .utils import _to_2d


class _BaseNB(torch.nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.n_classes = n_classes
        self.register_buffer(
            "class_log_prior", torch.zeros(n_classes, dtype=torch.float32)
        )
        self._is_fitted = False

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x), dim=-1)

    def extra_repr(self) -> str:
        return f"n_classes={self.n_classes}, fitted={self._is_fitted}"


# --------------------------------------------------------------- Gaussian
class GaussianNB(_BaseNB):
    """
    Analogue:
        sklearn.naive_bayes.GaussianNB
    Per-class Gaussian Naive Bayes.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    n_classes : int
        Number of classes.
    var_smoothing : float
        Portion of the largest feature variance added to every variance
        estimate for numerical stability.
    """

    def __init__(self, dim: int, n_classes: int, var_smoothing: float = 1e-9) -> None:
        super().__init__(n_classes)
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self.dim = dim
        self.var_smoothing = var_smoothing

        self.register_buffer(
            "class_means", torch.zeros(n_classes, dim, dtype=torch.float32)
        )
        self.register_buffer(
            "class_vars", torch.ones(n_classes, dim, dtype=torch.float32)
        )

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "GaussianNB":
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        y = y.long()
        if y.ndim == 2:
            y = y.squeeze(-1)
        n = x.shape[0]
        means, vars_, priors = [], [], []
        for c in range(self.n_classes):
            mask = (y == c)
            cnt = int(torch.sum(mask))
            if cnt == 0:
                raise ValueError(f"class {c} has no samples")
            xc = x[mask]
            mc = torch.mean(xc, dim=0)
            vc = torch.mean((xc - mc) ** 2, dim=0)
            means.append(mc)
            vars_.append(vc)
            priors.append(cnt / n)
        means = torch.stack(means, dim=0)
        vars_t = torch.stack(vars_, dim=0)
        vars_t = vars_t + self.var_smoothing * torch.max(vars_t)
        self.class_means.data.copy_(means)
        self.class_vars.data.copy_(vars_t)
        self.class_log_prior.data.copy_(torch.log(torch.tensor(priors)))
        self._is_fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_2d(x)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected input with {self.dim} features, got {x.shape[1]}"
            )
        if not self._is_fitted:
            raise RuntimeError("GaussianNB is not fitted; call fit() first.")
        diff = x.unsqueeze(1) - self.class_means.unsqueeze(0)    # [B, C, D]
        log_var = torch.log(self.class_vars)                     # [C, D]
        log_p_x = -0.5 * (
            diff * diff / self.class_vars.unsqueeze(0)
            + log_var.unsqueeze(0)
            + torch.log(torch.tensor(2 * torch.pi, dtype=x.dtype))
        ).sum(dim=-1)                                            # [B, C]
        return log_p_x + self.class_log_prior.unsqueeze(0)


# ------------------------------------------------------------- Multinomial
class MultinomialNB(_BaseNB):
    """
    Analogue:
        sklearn.naive_bayes.MultinomialNB
    Multinomial Naive Bayes for non-negative count features.

    Parameters
    ----------
    n_features : int
        Vocabulary / feature count dimension.
    n_classes : int
        Number of classes.
    alpha : float
        Laplace smoothing coefficient (added to every count).
    """

    def __init__(self, n_features: int, n_classes: int, alpha: float = 1.0) -> None:
        super().__init__(n_classes)
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        self.n_features = n_features
        self.alpha = alpha
        self.register_buffer(
            "feature_log_prob", torch.zeros(n_classes, n_features, dtype=torch.float32)
        )

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "MultinomialNB":
        x = _to_2d(x)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, got {x.shape[1]}"
            )
        if (x < 0).any():
            raise ValueError("MultinomialNB requires non-negative inputs")
        y = y.long()
        if y.ndim == 2:
            y = y.squeeze(-1)
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        n = x.shape[0]
        # Materialise y as a numpy array to avoid dispatch
        # issues with bool-mask indexing inside ``no_grad``.
        y_np = y.cpu().numpy()
        x_np = x.cpu().numpy()
        priors = []
        counts = torch.zeros(self.n_classes, self.n_features, dtype=x.dtype)
        for c in range(self.n_classes):
            mask_np = (y_np == c)
            cnt = int(mask_np.sum())
            if cnt == 0:
                raise ValueError(f"class {c} has no samples")
            priors.append(cnt / n)
            counts[c] = torch.tensor(x_np[mask_np].sum(axis=0))
        smoothed = counts + self.alpha
        self.feature_log_prob.data.copy_(
            torch.log(smoothed) - torch.log(torch.sum(smoothed, dim=1, keepdim=True))
        )
        self.class_log_prior.data.copy_(torch.log(torch.tensor(priors)))
        self._is_fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_2d(x)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, got {x.shape[1]}"
            )
        if not self._is_fitted:
            raise RuntimeError("MultinomialNB is not fitted; call fit() first.")
        return x @ self.feature_log_prob.T + self.class_log_prior.unsqueeze(0)