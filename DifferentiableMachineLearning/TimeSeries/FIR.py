"""FIR / MA(q) model:

    y(k) = b_0 u(k) + b_1 u(k-1) + ... + b_q u(k-q)

i.e. an ARMA with ``p = 0`` and a single exogenous input ``u``.
"""

import torch

from .Autoregressive import Autoregressive


class FIR(torch.nn.Module):
    """Finite Impulse Response (== MA(q)) model.

    Parameters
    ----------
    q : int
        Number of lags of ``u``. ``forward`` expects a tensor whose last
        dimension is ``q + 1``.
    """

    def __init__(self, q: int) -> None:
        super().__init__()
        if q < 0:
            raise ValueError(f"FIR order q must be >= 0, got {q}")
        self.q = q
        self._block = Autoregressive(
            y_features=0, x_features=[q + 1], e_features=0
        )

    def forward(self, u_history: torch.Tensor) -> torch.Tensor:
        return self._block(u_history)