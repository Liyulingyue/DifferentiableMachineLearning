"""ARMA(p, q) model:

    y(k) = -a_1 y(k-1) - ... - a_p y(k-p)
           + b_0 v(k) + b_1 v(k-1) + ... + b_q v(k-q)
    ``v`` is the residual / innovation series, stacked as the most-recent
    ``q + 1`` values along the feature axis.
"""

import torch

from .Autoregressive import Autoregressive


class ARMA(torch.nn.Module):
    """ARMA(p, q) model with a single exogenous-noise stream.

    Parameters
    ----------
    p : int
        AR order (lags of ``y``).
    q : int
        MA order (lags of residual ``v``). The model takes ``q + 1``
        most-recent residuals so that the ``b_0 v(k)`` term is included.
    """

    def __init__(self, p: int, q: int) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError(f"AR order p must be > 0, got {p}")
        if q < 0:
            raise ValueError(f"MA order q must be >= 0, got {q}")
        self.p, self.q = p, q
        self._block = Autoregressive(
            y_features=p, x_features=[], e_features=q + 1
        )

    def forward(
        self, y_history: torch.Tensor, v_history: torch.Tensor
    ) -> torch.Tensor:
        return self._block(y_history, v_history)