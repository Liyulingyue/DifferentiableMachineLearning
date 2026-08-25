"""Autoregressive model ``AR(p)``:

    y(k) = -a_1 y(k-1) - ... - a_p y(k-p)
"""

import torch

from .Autoregressive import Autoregressive


class AR(torch.nn.Module):
    """Pure AR(p) model.

    Parameters
    ----------
    p : int
        Number of lags. ``forward`` expects a tensor whose last
        dimension is ``p`` (i.e. the ``p`` most-recent observations
        stacked along the feature axis).
    """

    def __init__(self, p: int) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError(f"AR order p must be > 0, got {p}")
        self.p = p
        self._block = Autoregressive(y_features=p, x_features=[], e_features=0)

    def forward(self, y_history: torch.Tensor) -> torch.Tensor:
        return self._block(y_history)